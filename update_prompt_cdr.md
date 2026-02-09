# Update Instructions — Conditional CDR (12-month trailing)

Update the CDR calculation to use a conditional methodology: observe monthly default rates over a trailing 12-month window, average, annualize. CPR is UNCHANGED — keep the existing single-month calc_amort backsolve. Do not change anything else.

---

## Update 1: CLAUDE.md — Update CDR lines in Domain Terminology / Rate Conversions

Find the existing CDR line(s). They may look like:
```
- **CDR (Cumulative Default Rate)**: `CDR = sum(funded_amnt - total_rec_prncp for Charged Off) / sum(funded_amnt for all loans in cohort)`
```

Replace with these three lines:
```markdown
- **MDR (Monthly Default Rate, observed)**: For a given calendar month M: `MDR_M = defaults_in_month_M / performing_balance_at_start_of_month_M`. This parallels SMM for prepayments.
- **CDR (Conditional Default Rate)**: `CDR = 1 - (1 - avg_MDR)^12` where `avg_MDR` is the average of 12 trailing monthly MDRs (Apr 2018 – Mar 2019). This is the PRIMARY rate used in cash flow projections. Uses the dv01 conditional methodology.
- **Cumulative Default Rate (reference only)**: `cumulative_default_rate = sum(defaulted_upb) / sum(funded_amnt)`. Raw lifetime default rate. Displayed for reference but NOT used in cash flow engine. Do NOT call this "CDR" — CDR refers only to the conditional (annualized) rate.
```

Also find the existing MDR projection line:
```
- **MDR (Monthly Default Rate)**: `MDR = 1 - (1 - CDR)^(1/12)`
```

Replace with:
```markdown
- **MDR (Monthly Default Rate, for projections)**: `MDR = 1 - (1 - CDR)^(1/12)`. The round-trip is: observe monthly MDRs → average → annualize to CDR → cash flow engine converts back to MDR via this formula.
```

---

## Update 2: CLAUDE.md — Replace CDR calculation in compute_pool_assumptions

Find the CDR calculation block inside the `compute_pool_assumptions` function spec. It may contain references to `CDR = defaulted_upb / total_originated_upb` and/or WALA-based annualization. Replace the ENTIRE CDR calculation block with:

```markdown
**CDR calculation — Conditional, trailing 12-month average** (uses `df_all`):

This uses the dv01 conditional methodology. Requires `reconstruct_loan_timeline()` to have been called on `df_all` first (for `default_month` and `payoff_month` columns).

```python
from src.amortization import calc_balance, calc_monthly_payment

snapshot = pd.Timestamp('2019-03-01')

monthly_mdrs = []
for m in range(1, 13):  # trailing 12 months: Apr 2018 – Mar 2019
    month_start = snapshot - pd.DateOffset(months=m)
    month_end = month_start + pd.DateOffset(months=1)

    # --- Defaults in this month ---
    # Loans whose reconstructed default_month falls in [month_start, month_end)
    defaults_mask = (
        (df_all['default_month'] >= month_start) &
        (df_all['default_month'] < month_end)
    )
    default_upb = (
        df_all.loc[defaults_mask, 'funded_amnt'] -
        df_all.loc[defaults_mask, 'total_rec_prncp']
    ).clip(lower=0).sum()

    # --- Performing balance at start of month ---
    # Loans originated before month_start, not yet defaulted, not yet paid off
    originated = df_all['issue_d'] <= month_start
    not_defaulted = df_all['default_month'].isna() | (df_all['default_month'] > month_start)
    not_paid_off = df_all['payoff_month'].isna() | (df_all['payoff_month'] > month_start)
    performing_mask = originated & not_defaulted & not_paid_off

    # Estimate each performing loan's balance using amortization formula
    # This is a forward-solve approximation (assumes on-schedule payments),
    # which is reasonable at pool level for the denominator
    performing = df_all.loc[performing_mask].copy()
    performing['age_at_month'] = (
        (month_start - performing['issue_d']).dt.days / 30.44
    ).round().astype(int).clip(lower=0)

    monthly_payment = calc_monthly_payment(
        performing['funded_amnt'].values,
        performing['int_rate'].values,
        performing['term_months'].values
    )
    est_balance, _, _ = calc_balance(
        performing['funded_amnt'].values,
        performing['int_rate'].values,
        monthly_payment,
        performing['age_at_month'].values
    )
    performing_balance = np.clip(est_balance, 0, None).sum()

    # --- Monthly default rate ---
    mdr = default_upb / performing_balance if performing_balance > 0 else 0.0
    monthly_mdrs.append(mdr)

# Average MDR, annualize
avg_mdr = np.mean(monthly_mdrs)
CDR = 1 - (1 - avg_mdr) ** 12

# Cumulative default rate for reference display only (NOT called CDR)
charged_off = df_all[df_all['loan_status'] == 'Charged Off']
defaulted_upb_cum = (charged_off['funded_amnt'] - charged_off['total_rec_prncp']).clip(lower=0).sum()
total_originated = df_all['funded_amnt'].sum()
cumulative_default_rate = defaulted_upb_cum / total_originated if total_originated > 0 else 0.0
```
```

---

## Update 3: CLAUDE.md — Update compute_pool_assumptions Returns

Find the existing Returns line for `compute_pool_assumptions`. Replace it with:

```markdown
**Returns**: `{'cdr': float, 'cumulative_default_rate': float, 'avg_mdr': float, 'monthly_mdrs': list[float], 'cpr': float, 'loss_severity': float, 'recovery_rate': float}`

Where:
- `cdr` = conditional CDR (12-month trailing average MDR, annualized) — PRIMARY, used in cash flow projections
- `cumulative_default_rate` = raw lifetime cumulative default rate — for display/reference only, NOT called CDR
- `avg_mdr` = average monthly default rate (un-annualized) — for display
- `monthly_mdrs` = list of 12 individual monthly MDRs — for display (shows trend/volatility)
- `cpr` = single-month CPR from calc_amort (UNCHANGED)
- `loss_severity`, `recovery_rate` = unchanged

**Dependencies**: `reconstruct_loan_timeline()` must have been called on `df_all` first (for `default_month`, `payoff_month`). `calc_amort()` must have been called on `df_current` (for CPR, unchanged).
```

---

## Update 4: CLAUDE.md — Update Tab 2 metric cards

Find the Tab 2 section. Replace the line about showing base assumptions:
```markdown
- Show computed base assumptions (CDR, CPR, loss severity) as metric cards
```

With:
```markdown
- Show computed base assumptions as metric cards:
  - **CDR (Conditional, 12mo trailing)**: the annualized rate driving projections
  - **Cumulative Default Rate**: raw lifetime rate — shown in smaller text or tooltip as reference (do NOT label as "CDR")
  - **Avg MDR**: un-annualized monthly default rate for transparency
  - **CPR**: single-month from calc_amort (unchanged)
  - **Loss Severity** and **Recovery Rate**: unchanged
```

---

## Update 5: CLAUDE.md — Update Critical Rules

Find this line:
```markdown
- **CDR is computed from ALL loans in the cohort.**
```

Replace with:
```markdown
- **CDR is computed from ALL loans in the cohort** using the dv01 conditional methodology: trailing 12-month average MDR, annualized. Cumulative default rate (NOT called CDR) retained for display only.
- **`compute_pool_assumptions()` requires `reconstruct_loan_timeline()`** on df_all first (for `default_month` and `payoff_month` columns used in conditional CDR).
```

---

## Update 6: CLAUDE.md — Update testing requirements

Find the existing `compute_pool_assumptions` test line(s) in the testing section and replace with:

```markdown
- `compute_pool_assumptions` CDR:
  - Create synthetic cohort: 1000 loans originated Jan 2017. In each of the 12 trailing months (Apr 2018 – Mar 2019), 5 loans default each month with $10K exposure each. Performing balance ~$8M/month. Verify avg_MDR ≈ $50K/$8M ≈ 0.625%/month, CDR = 1-(1-0.00625)^12 ≈ 7.24%
  - Zero defaults in all 12 months: all MDRs = 0, CDR = 0
  - Verify monthly_mdrs list has exactly 12 elements
  - Verify CDR round-trip: CDR → MDR via projection formula ≈ avg_mdr from observations
  - Verify cumulative_default_rate is also returned (different value from conditional CDR, and not labeled as CDR)
```

---

## Update 7: .claude/agents/financial-validator.md — Update CDR tests

Find the existing `**CDR ↔ MDR**:` section (and any `**CDR annualization**:` section if present). Replace ALL CDR-related test sections with:

```markdown
**CDR — conditional, trailing 12-month**:
- 12 monthly MDRs all equal to 0.005 (0.5%/month) → avg_MDR = 0.005 → CDR = 1-(1-0.005)^12 = 0.05841 (5.84%)
- 12 monthly MDRs varying between 0.003 and 0.007 → avg_MDR = 0.005 → CDR = 5.84% (same — averaging smooths volatility)
- Zero defaults in all 12 months → all MDRs = 0 → CDR = 0
- Verify round-trip: CDR → MDR = 1-(1-CDR)^(1/12) ≈ avg_MDR. Example: CDR = 0.05841 → MDR = 1-(1-0.05841)^(1/12) ≈ 0.005 ✓
- Verify conditional CDR ≠ cumulative default rate (they measure different things; only the conditional rate is called CDR)
- Verify each monthly MDR is non-negative and ≤ 1.0
- CDR = 0.00 → MDR = 0.00 (boundary case)
- CDR = 1.00 → MDR = 1.00 (boundary case)
```

---

That's it. Seven targeted edits across two files. CPR is completely unchanged (single-month backsolve from calc_amort). Only CDR changes — from cumulative/WALA to conditional 12-month trailing.
