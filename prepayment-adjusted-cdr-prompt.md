# Prepayment-Adjusted CDR Denominator

## Problem

`compute_pool_assumptions()` in `src/cashflow_engine.py` estimates the performing balance for each trailing month using `calc_balance()` — which assumes zero prepayments. A loan that has been prepaying $100/month extra for 2 years has a lower actual balance than the schedule predicts. This overstates the denominator → understates MDR → understates CDR.

## Fix

For loans where we can observe the actual current balance (`out_prncp`), compute the cumulative prepayment at the snapshot, spread it evenly across the loan's age, and subtract the prorated amount from the scheduled balance at each historical month.

**Why this is valid for CDR but NOT CPR:**
- CDR numerator is directly observed (which loans defaulted, their exposure). Only the denominator is estimated. One degree of freedom.
- CPR would require estimating both the numerator (monthly prepayment amount) and the denominator (balance) from the same uniform-spread assumption. Two degrees of freedom — errors compound rather than cancel.
- The single-month CPR we currently use has actual observed values on both sides (`last_pymnt_amnt` drives the numerator, `out_prncp` drives the denominator via beginning balance reconstruction). It's noisier (one month vs twelve) but both inputs are real.
- CPR is computed from **Current loans only** — delinquent loans are behind on payments, not prepaying. Including them would contaminate the prepayment signal. This is already implemented and must not change.

## Code Change — `src/cashflow_engine.py`

Replace the CDR section of `compute_pool_assumptions()` (lines 55–106). The CPR, loss severity, and return dict are unchanged.

### Current code (lines 55–106):

```python
    # --- CDR (Conditional, trailing 12-month) ---
    snapshot = pd.Timestamp('2019-03-01')
    issue_dates = pd.to_datetime(df_all['issue_d'])

    # Parse default_month and payoff_month once
    default_month = pd.to_datetime(df_all['default_month'], errors='coerce')
    payoff_month = pd.to_datetime(df_all['payoff_month'], errors='coerce')

    monthly_mdrs = []
    for m in range(1, 13):  # trailing 12 months: Apr 2018 – Mar 2019
        month_start = snapshot - pd.DateOffset(months=m)
        month_end = month_start + pd.DateOffset(months=1)

        # Defaults in this month: loans whose default_month falls in [month_start, month_end)
        defaults_mask = (default_month >= month_start) & (default_month < month_end)
        default_upb = (
            df_all.loc[defaults_mask, 'funded_amnt'] -
            df_all.loc[defaults_mask, 'total_rec_prncp']
        ).clip(lower=0).sum()

        # Performing balance at start of month:
        # originated before month_start, not yet defaulted, not yet paid off
        originated = issue_dates <= month_start
        not_defaulted = default_month.isna() | (default_month > month_start)
        not_paid_off = payoff_month.isna() | (payoff_month > month_start)
        performing_mask = originated & not_defaulted & not_paid_off

        perf_idx = performing_mask.values.nonzero()[0]

        if len(perf_idx) > 0:
            funded = df_all['funded_amnt'].values[perf_idx].astype(np.float64)
            rates = df_all['int_rate'].values[perf_idx].astype(np.float64)
            terms = df_all['term_months'].values[perf_idx].astype(np.float64)

            perf_issue = issue_dates.values[perf_idx]
            td = np.datetime64(month_start) - perf_issue
            age_at_month = np.round(
                td.astype('timedelta64[D]').astype(np.float64) / 30.44
            ).astype(int).clip(min=0)

            pmt = calc_monthly_payment(funded, rates, terms)
            est_balance, _, _ = calc_balance(funded, rates, pmt,
                                             age_at_month.astype(np.float64))
            performing_balance = np.clip(est_balance, 0, None).sum()
        else:
            performing_balance = 0.0

        mdr = default_upb / performing_balance if performing_balance > 0 else 0.0
        monthly_mdrs.append(mdr)

    avg_mdr = float(np.mean(monthly_mdrs))
    cdr = 1 - (1 - avg_mdr) ** 12
```

### Replacement code:

```python
    # --- CDR (Conditional, trailing 12-month) ---
    snapshot = pd.Timestamp('2019-03-01')
    issue_dates = pd.to_datetime(df_all['issue_d'])

    # Parse default_month and payoff_month once
    default_month = pd.to_datetime(df_all['default_month'], errors='coerce')
    payoff_month = pd.to_datetime(df_all['payoff_month'], errors='coerce')

    # --- Precompute per-loan monthly prepayment rate for balance adjustment ---
    # For active loans (still performing at snapshot), we can observe the actual
    # balance (out_prncp) vs the scheduled balance.  Spreading the cumulative
    # prepayment evenly across the loan's age gives a better estimate of what
    # the balance was at each historical month.
    #
    # For Fully Paid / Charged Off loans we don't have a reliable calibration
    # endpoint, so monthly_prepaid stays 0 (unadjusted schedule).

    all_funded = df_all['funded_amnt'].values.astype(np.float64)
    all_rates = df_all['int_rate'].values.astype(np.float64)
    all_terms = df_all['term_months'].values.astype(np.float64)
    all_out_prncp = df_all['out_prncp'].values.astype(np.float64)

    # Age at snapshot for every loan
    td_snapshot = np.datetime64(snapshot) - issue_dates.values
    age_at_snapshot = np.round(
        td_snapshot.astype('timedelta64[D]').astype(np.float64) / 30.44
    ).astype(int).clip(min=0)

    # Scheduled balance at snapshot (zero-prepayment assumption)
    all_pmt = calc_monthly_payment(all_funded, all_rates, all_terms)
    sched_balance_at_snapshot, _, _ = calc_balance(
        all_funded, all_rates, all_pmt, age_at_snapshot.astype(np.float64)
    )
    sched_balance_at_snapshot = np.clip(sched_balance_at_snapshot, 0, None)

    # Cumulative prepayment = scheduled balance - actual balance (for active loans)
    active_statuses = {'Current', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)'}
    is_active = df_all['loan_status'].isin(active_statuses).values
    has_age = age_at_snapshot > 0

    cumulative_prepaid = np.clip(sched_balance_at_snapshot - all_out_prncp, 0, None)
    monthly_prepaid = np.zeros(len(df_all), dtype=np.float64)
    adjustable = is_active & has_age
    monthly_prepaid[adjustable] = cumulative_prepaid[adjustable] / age_at_snapshot[adjustable]

    # --- Compute trailing 12-month MDRs ---
    monthly_mdrs = []
    for m in range(1, 13):  # trailing 12 months: Apr 2018 – Mar 2019
        month_start = snapshot - pd.DateOffset(months=m)
        month_end = month_start + pd.DateOffset(months=1)

        # Defaults in this month: loans whose default_month falls in [month_start, month_end)
        defaults_mask = (default_month >= month_start) & (default_month < month_end)
        default_upb = (
            df_all.loc[defaults_mask, 'funded_amnt'] -
            df_all.loc[defaults_mask, 'total_rec_prncp']
        ).clip(lower=0).sum()

        # Performing balance at start of month:
        # originated before month_start, not yet defaulted, not yet paid off
        originated = issue_dates <= month_start
        not_defaulted = default_month.isna() | (default_month > month_start)
        not_paid_off = payoff_month.isna() | (payoff_month > month_start)
        performing_mask = originated & not_defaulted & not_paid_off

        perf_idx = performing_mask.values.nonzero()[0]

        if len(perf_idx) > 0:
            funded = all_funded[perf_idx]
            rates = all_rates[perf_idx]
            terms = all_terms[perf_idx]

            perf_issue = issue_dates.values[perf_idx]
            td = np.datetime64(month_start) - perf_issue
            age_at_month = np.round(
                td.astype('timedelta64[D]').astype(np.float64) / 30.44
            ).astype(int).clip(min=0)

            pmt = calc_monthly_payment(funded, rates, terms)
            sched_balance, _, _ = calc_balance(funded, rates, pmt,
                                               age_at_month.astype(np.float64))
            sched_balance = np.clip(sched_balance, 0, None)

            # Subtract prorated cumulative prepayment at this historical month
            cum_prepaid_at_month = monthly_prepaid[perf_idx] * age_at_month
            adj_balance = np.clip(sched_balance - cum_prepaid_at_month, 0, None)

            performing_balance = adj_balance.sum()
        else:
            performing_balance = 0.0

        mdr = default_upb / performing_balance if performing_balance > 0 else 0.0
        monthly_mdrs.append(mdr)

    avg_mdr = float(np.mean(monthly_mdrs))
    cdr = 1 - (1 - avg_mdr) ** 12
```

### Key differences from the old code:
1. **New pre-loop block** computes `monthly_prepaid` per loan (only for active loans with age > 0)
2. **Inside the loop**, `cum_prepaid_at_month = monthly_prepaid[perf_idx] * age_at_month` is subtracted from the scheduled balance
3. Reuses pre-computed `all_funded`, `all_rates`, `all_terms` arrays instead of re-slicing each iteration (minor perf improvement)
4. **Numerator is unchanged** — default_upb still comes directly from observed data
5. **CPR and loss severity are unchanged** — only the CDR denominator is affected

### What NOT to change:
- The CPR section — single-month observed, **Current loans only**, not estimated
- The loss severity section
- The return dict
- The docstring (update it to mention the prepayment adjustment — see below)

### Docstring update:

Add the following **after** the existing CDR description line (`annualized via CDR = 1 - (1 - avg_MDR)^12.`). Do NOT duplicate the existing text:

```
    The performing balance denominator is prepayment-adjusted: for loans still
    active at the snapshot, we observe the gap between the scheduled balance
    (from amortization) and the actual balance (out_prncp), spread the
    cumulative prepayment evenly across the loan's age, and subtract the
    prorated amount at each historical month. This avoids overstating the
    denominator (and thus understating CDR) for pools with significant
    prepayment activity. Loans that defaulted or paid off before the snapshot
    use the unadjusted amortization schedule (no calibration endpoint available).

    CPR is NOT prepayment-adjusted — it uses actual observed values from the
    last payment (Current loans only). See CPR section below.
```

---

## Validator Update — `.claude/agents/financial-validator.md`

After this CDR change is applied, update Step 9 (CDR Computation). Replace the **Performing balance reconstruction** section with:

```markdown
**Performing balance — prepayment-adjusted**:
- For active loans (Current + In Grace + Late) at the snapshot with age > 0:
  - `sched_balance_at_snapshot = calc_balance(funded_amnt, int_rate, pmt, age_at_snapshot)`
  - `cumulative_prepaid = max(sched_balance_at_snapshot − out_prncp, 0)`
  - `monthly_prepaid = cumulative_prepaid / age_at_snapshot`
- For Fully Paid / Charged Off loans: `monthly_prepaid = 0` (no calibration endpoint)
- For each historical month M:
  - `sched_balance_at_M = calc_balance(funded_amnt, int_rate, pmt, age_at_M)`
  - `cum_prepaid_at_M = monthly_prepaid × age_at_M`
  - `adj_balance_at_M = max(sched_balance_at_M − cum_prepaid_at_M, 0)`
  - `performing_balance = Σ adj_balance_at_M`
- Verify: adjusted performing balance ≤ unadjusted performing balance (always — prepayments only reduce it)
- Verify: for a pool with zero prepayment (out_prncp = sched_balance for all active loans), adjusted = unadjusted
- Verify: monthly_prepaid ≥ 0 for all loans
- Verify: cum_prepaid_at_M ≤ sched_balance_at_M (prepayment can't exceed the balance)
```

And add to Critical Rules:

```markdown
- CDR denominator uses prepayment-adjusted balances; CPR does NOT (single-month observed values, Current loans only)
- The CDR adjustment only applies to loans still active at the snapshot — Fully Paid and Charged Off loans use unadjusted amortization
```

---

## Writeup Update — `writeup.md`

After this CDR change is applied, update Section 2 MDR denominator. Replace:

```markdown
For each performing loan, we calculate its scheduled balance at that historical date using the amortization formula (known funded amount, rate, and term):

age = months between issue_d and month_start
payment = PV × r / (1 − (1 + r)^(−n))
balance_at_month = calc_balance(funded_amnt, int_rate, payment, age)
performing_balance = Σ balance_at_month
```

With:

```markdown
For each performing loan, we calculate its scheduled balance at that historical date, then adjust for observed prepayment behavior. Using the unadjusted amortization schedule would overstate balances for loans that have been prepaying → understate MDR → understate CDR. For loans still active at the snapshot, we calibrate from the observed endpoint:

sched_balance_at_snapshot = calc_balance(funded_amnt, int_rate, payment, age_at_snapshot)
cumulative_prepaid = max(sched_balance_at_snapshot − out_prncp, 0)
monthly_prepaid = cumulative_prepaid / age_at_snapshot

Then for each historical month M:

sched_balance_at_M = calc_balance(funded_amnt, int_rate, payment, age_at_M)
cum_prepaid_at_M = monthly_prepaid × age_at_M
adj_balance_at_M = max(sched_balance_at_M − cum_prepaid_at_M, 0)
performing_balance = Σ adj_balance_at_M

For Fully Paid / Charged Off loans, no actual balance is observable at the snapshot, so we use the unadjusted amortization schedule.

This adjustment is valid for the CDR denominator because the numerator (default exposure) is directly observed — one degree of freedom. It is NOT applied to CPR, where both numerator (monthly prepayment) and denominator (balance) would need to be estimated from the same uniform-spread assumption — two degrees of freedom, errors compound. The single-month CPR uses actual observed values on both sides (Current loans only).
```

---

## docs/calculations.md Update

In Section 3 → CDR → step 2, replace:

```markdown
2. Estimate performing balance at start of month:
   - Loans originated before month_start
   - Not yet defaulted (default_month is null or > month_start)
   - Not yet paid off (payoff_month is null or > month_start)
   - Balance estimated via amortization formula: calc_balance(funded_amnt, int_rate, pmt, age)
```

With:

```markdown
2. Estimate performing balance at start of month (prepayment-adjusted):
   - Loans originated before month_start
   - Not yet defaulted (default_month is null or > month_start)
   - Not yet paid off (payoff_month is null or > month_start)
   - Scheduled balance via amortization: sched_bal = calc_balance(funded_amnt, int_rate, pmt, age)
   - For active loans at snapshot: adjust by prorated cumulative prepayment:
     monthly_prepaid = max(sched_bal_at_snapshot − out_prncp, 0) / age_at_snapshot
     adj_bal = max(sched_bal − monthly_prepaid × age, 0)
   - For Fully Paid / Charged Off loans: use unadjusted sched_bal (no calibration endpoint)
```

---

## Test Updates — `tests/test_cashflow_engine.py`

Add a test verifying the prepayment adjustment:

```python
def test_cdr_prepayment_adjusted_denominator():
    """CDR denominator should be lower when loans have prepaid."""
    # Create a minimal df_all with two Current loans:
    # Loan 1: no prepayment (out_prncp = scheduled balance)
    # Loan 2: has prepaid (out_prncp < scheduled balance)
    #
    # The adjusted performing balance should be less than the
    # unadjusted (pure amortization) balance, resulting in a
    # higher MDR and higher CDR.
    #
    # Implementation: run compute_pool_assumptions twice — once with
    # out_prncp = scheduled balance (no adjustment), once with
    # out_prncp < scheduled balance (adjustment kicks in).
    # Assert: CDR with adjustment > CDR without adjustment.
    pass  # Implement with synthetic data matching your fixture patterns
```

---

## Verification Steps

After applying the change:

1. `pytest tests/ -v` — all existing tests pass
2. Run the dashboard, verify Tab 2 CDR values are slightly higher than before (smaller denominator → higher MDR → higher CDR)
3. Spot-check: for a pool with zero prepayments (all `out_prncp` = scheduled balance), CDR should be identical to pre-change values
4. Spot-check: `monthly_prepaid` is 0 for all Fully Paid and Charged Off loans
5. Run the financial validator agent — all steps pass including the new Step 9 checks
6. Verify CPR values are **unchanged** — this change does not touch the CPR path (Current loans only, single-month observed)
