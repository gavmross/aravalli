# Calculations & Formula Reference

This document defines every financial formula used in the Lending Club Portfolio Investment Analysis Tool. All formulas are implemented in `src/cashflow_engine.py`, `src/scenario_analysis.py`, and `src/amortization.py`.

---

## Table of Contents

1. [Standard Amortization](#1-standard-amortization)
2. [Rate Conversions](#2-rate-conversions)
3. [Pool-Level Assumptions](#3-pool-level-assumptions)
4. [Pool-Level Weighted Averages](#4-pool-level-weighted-averages)
5. [Monthly Cash Flow Projection](#5-monthly-cash-flow-projection)
6. [IRR Calculation](#6-irr-calculation)
7. [Price Solver](#7-price-solver)
8. [Scenario Analysis](#8-scenario-analysis)
9. [State-Transition Cash Flow Model](#9-state-transition-cash-flow-model)

---

## 1. Standard Amortization

Implemented in `src/amortization.py`.

### Monthly Payment (PMT)

```
PMT = PV * r / (1 - (1 + r)^(-n))
```

| Variable | Definition | Example |
|----------|-----------|---------|
| `PV` | Loan principal (funded_amnt) | $10,000 |
| `r` | Monthly interest rate = `int_rate / 12` | 0.10/12 = 0.008333 |
| `n` | Loan term in months (term_months) | 36 |
| `PMT` | Fixed monthly payment | $322.67 |

**Edge case**: If `int_rate = 0`, then `PMT = PV / n` (simple division).

**Worked example**: $10,000 at 10% annual for 36 months:
```
r = 0.10 / 12 = 0.008333
PMT = 10000 * 0.008333 / (1 - 1.008333^(-36)) = $322.67
```

### Remaining Balance After k Payments

```
Balance_k = PV * [(1+r)^n - (1+r)^k] / [(1+r)^n - 1]
```

| Variable | Definition | Example |
|----------|-----------|---------|
| `k` | Number of payments made | 12 |
| `Balance_k` | Outstanding principal after k payments | $6,915.72 |

**Worked example**: $10,000 at 10% for 36 months, after 12 payments:
```
Balance_12 = 10000 * [1.008333^36 - 1.008333^12] / [1.008333^36 - 1] = $6,915.72
```

### Payment Number Calculation

```
payments_made = months_between(issue_d, last_pymnt_d)
```

Calculated as `(last_pymnt_d.year - issue_d.year) * 12 + (last_pymnt_d.month - issue_d.month)`.

### Last Payment Period Decomposition

For each loan, the last payment is decomposed into scheduled and unscheduled components:

```
beginning_balance = (out_prncp + last_pymnt_amnt) / (1 + monthly_rate)
scheduled_interest = beginning_balance * monthly_rate
scheduled_principal = PMT - scheduled_interest
actual_principal = beginning_balance - out_prncp
unscheduled_principal = max(actual_principal - scheduled_principal, 0)
```

The `unscheduled_principal` represents prepayment — principal paid beyond the scheduled amortization amount.

---

## 2. Rate Conversions

### CPR <-> SMM (Prepayment Rates)

**SMM (Single Monthly Mortality)** is the monthly prepayment fraction. **CPR (Conditional Prepayment Rate)** is the annualized equivalent.

```
CPR = 1 - (1 - SMM)^12        # Monthly to annual
SMM = 1 - (1 - CPR)^(1/12)    # Annual to monthly (used in projections)
```

**Worked example**:
- SMM = 0.01 (1% of performing balance prepays monthly)
- CPR = 1 - (1 - 0.01)^12 = 1 - 0.99^12 = 1 - 0.8864 = **11.36%**
- Reverse: SMM = 1 - (1 - 0.1136)^(1/12) = 1 - 0.8864^(1/12) = **0.01** (verified)

### CDR <-> MDR (Default Rates)

**CDR (Conditional Default Rate)** is the annualized default rate derived from observed monthly default rates using the dv01 conditional methodology. **MDR (Monthly Default Rate)** is the monthly rate used in projections.

**Observed MDR** (for a given calendar month M):
```
MDR_M = defaults_in_month_M / performing_balance_at_start_of_month_M
```

**CDR from observed MDRs** (trailing 12-month average):
```
avg_MDR = mean(MDR_1, MDR_2, ..., MDR_12)    # Apr 2018 – Mar 2019
CDR = 1 - (1 - avg_MDR)^12
```

**MDR from CDR** (for projections):
```
MDR = 1 - (1 - CDR)^(1/12)
```

The round-trip is: observe monthly MDRs → average → annualize to CDR → cash flow engine converts back to MDR.

**Worked example**:
- 12 monthly MDRs all equal to 0.005 (0.5%/month)
- avg_MDR = 0.005
- CDR = 1 - (1 - 0.005)^12 = 1 - 0.995^12 = **5.84%**
- Round-trip: MDR = 1 - (1 - 0.0584)^(1/12) = 1 - 0.9416^(1/12) = **0.005** (verified)

**Important**: Do NOT use `CDR / 12` as an approximation. The compounding formula is exact and must be used.

**Note**: The **Cumulative Default Rate** (`sum(defaulted_upb) / sum(funded_amnt)`) is a separate metric retained for display/reference only. It is NOT called CDR and is NOT used in cash flow projections.

---

## 3. Pool-Level Assumptions

Implemented in `src/cashflow_engine.py: compute_pool_assumptions()`.

### CDR (Conditional Default Rate)

Computed from **ALL loans** in the filtered strata using the dv01 conditional methodology.

**Requires**: `reconstruct_loan_timeline()` must have been called on `df_all` first (for `default_month` and `payoff_month` columns).

For each of 12 trailing months (Apr 2018 – Mar 2019):

```
1. Identify defaults: loans whose default_month falls in [month_start, month_end)
   default_upb = sum(funded_amnt - total_rec_prncp)  [for defaults, clipped ≥ 0]

2. Estimate performing balance at start of month:
   - Loans originated before month_start
   - Not yet defaulted (default_month is null or > month_start)
   - Not yet paid off (payoff_month is null or > month_start)
   - Balance estimated via amortization formula: calc_balance(funded_amnt, int_rate, pmt, age)

3. MDR_M = default_upb / performing_balance
```

Then annualize:
```
avg_MDR = mean(MDR_1, MDR_2, ..., MDR_12)
CDR = 1 - (1 - avg_MDR)^12
```

**Cumulative Default Rate** (reference only, NOT called CDR):
```
cumulative_default_rate = sum(defaulted_upb) / sum(funded_amnt)    [all Charged Off loans]
```

This raw lifetime rate is displayed for reference but is NOT used in cash flow projections.

### CPR (Conditional Prepayment Rate)

Computed from **all active loans**: Current loans with March 2019 last payment date plus all delinquent loans (In Grace + Late 16-30 + Late 31-120) regardless of last payment date:

```
total_beginning_balance = sum(last_pmt_beginning_balance)
total_scheduled_principal = sum(last_pmt_scheduled_principal)
total_unscheduled_principal = sum(last_pmt_unscheduled_principal)
denominator = total_beginning_balance - total_scheduled_principal
SMM = total_unscheduled_principal / denominator
CPR = 1 - (1 - SMM)^12
```

Only loans with `last_pmt_beginning_balance > 0` are included (avoids division by zero for loans with no prior balance).

**Rationale**: CPR is computed from the most recent payment period of performing loans. The pool-level SMM aggregates across all Current loans rather than averaging individual SMMs, which would not properly weight by balance.

### Loss Severity

Computed from **Charged Off loans** in the filtered strata with positive exposure:

```
exposure = funded_amnt - total_rec_prncp                          [per Charged Off loan]
capped_recoveries = min(recoveries, exposure)                     [cap at 100% recovery]
capped_upb_lost = clip(exposure - capped_recoveries, min=0)       [floor at zero]
loss_severity = sum(capped_upb_lost) / sum(exposure)
recovery_rate = sum(capped_recoveries) / sum(exposure)
```

**Identity**: `loss_severity + recovery_rate = 1.0` (always holds when recoveries are capped).

**Why cap recoveries?** Four loans in the dataset have `recoveries > exposure`, which would make `recovery_rate > 1.0` and break the identity. Capping ensures the model is economically sensible.

**Worked example**:
- Loan: funded_amnt = $10,000, total_rec_prncp = $3,000, recoveries = $1,500
- Exposure = $10,000 - $3,000 = $7,000
- Capped recoveries = min($1,500, $7,000) = $1,500
- UPB lost = $7,000 - $1,500 = $5,500
- Loss severity = $5,500 / $7,000 = **78.57%**
- Recovery rate = $1,500 / $7,000 = **21.43%**
- Sum = 100% (verified)

---

## 4. Pool-Level Weighted Averages

Implemented in `src/cashflow_engine.py: compute_pool_characteristics()`. All computed from **active loans**: Current with March 2019 last payment date plus all delinquent loans (In Grace + Late) regardless of last payment date.

### WAC (Weighted Average Coupon)

```
WAC = sum(int_rate * out_prncp) / sum(out_prncp)
```

Weighted by current outstanding principal, not original funded amount. Result is a decimal (e.g., 0.1269 = 12.69%).

### WAM (Weighted Average Maturity)

```
WAM = round(sum(updated_remaining_term * out_prncp) / sum(out_prncp))
```

- `updated_remaining_term` is computed by `calc_amort()` as `term_months - payments_made`
- Rounded to nearest integer (months)
- Determines how many months the cash flow projection runs

### Monthly Payment

```
monthly_payment = sum(installment)
```

Sum of all individual loan installments for the active March 2019 pool.

---

## 5. Monthly Cash Flow Projection (Simple Model)

> **Note**: Sections 5-8 describe the Simple (flat CDR/CPR) model implemented in `project_cashflows()`, `solve_price()`, `build_scenarios()`, and `compare_scenarios()`. These functions exist in the codebase but are **not currently used by the dashboard**, which uses the State-Transition model described in [Section 9](#9-state-transition-cash-flow-model-advanced). These sections are retained for reference.

Implemented in `src/cashflow_engine.py: project_cashflows()`.

This engine projects monthly cash flows at the **pool-level aggregate** — one balance, one WAC, one WAM per run. It does NOT model individual loans.

### Inputs

| Parameter | Source | Example |
|-----------|--------|---------|
| `total_upb` | `compute_pool_characteristics()` | $50,000,000 |
| `wac` | `compute_pool_characteristics()` | 0.1269 (12.69%) |
| `wam` | `compute_pool_characteristics()` | 32 months |
| `monthly_payment` | `compute_pool_characteristics()` | $1,800,000 |
| `cdr` | `compute_pool_assumptions()` or user input | 0.10 (10%) |
| `cpr` | `compute_pool_assumptions()` or user input | 0.12 (12%) |
| `loss_severity` | `compute_pool_assumptions()` | 0.88 (88%) |
| `purchase_price` | User input (sidebar slider) | 0.95 (95 cents on dollar) |

### Pre-Loop Setup

```
MDR = 1 - (1 - CDR)^(1/12)     # Monthly default rate
SMM = 1 - (1 - CPR)^(1/12)     # Single monthly mortality
monthly_rate = WAC / 12          # Monthly interest rate
```

### Monthly Loop (t = 1 to WAM)

The projection starts at t=0 (March 2019), with first cash flow at t=1 (April 2019).

For each month `t`:

```
1. beginning_balance = previous ending_balance (or total_upb at t=1)

2. DEFAULTS (applied first)
   defaults = beginning_balance * MDR
   loss = defaults * loss_severity
   recovery = defaults * (1 - loss_severity)

3. PERFORMING BALANCE
   performing_balance = beginning_balance - defaults

4. INTEREST (on performing balance only)
   interest = performing_balance * monthly_rate

5. SCHEDULED PRINCIPAL
   scheduled_principal = monthly_payment - interest
   scheduled_principal = min(scheduled_principal, performing_balance)
   scheduled_principal = max(scheduled_principal, 0)

6. PREPAYMENTS
   prepayments = (performing_balance - scheduled_principal) * SMM

7. TOTALS
   total_principal = scheduled_principal + prepayments
   ending_balance = performing_balance - total_principal
   ending_balance = max(ending_balance, 0)
   total_cashflow = interest + total_principal + recovery
```

The loop terminates early if `beginning_balance <= 0`.

### Key Design Decisions

- **Defaults happen first**: This is the standard convention in structured finance. Defaults reduce the balance before interest and amortization are calculated.
- **Interest on performing balance**: Only non-defaulted principal accrues interest. Defaulted principal produces loss and recovery.
- **Fixed monthly payment**: The `monthly_payment` from pool characteristics is used throughout. As the balance decreases, the scheduled principal portion increases and interest decreases (standard amortization behavior).
- **Prepayments after scheduled principal**: SMM is applied to the remaining balance after scheduled amortization, preventing double-counting.
- **Balance floor at zero**: Floating-point arithmetic can produce tiny negative balances; these are floored at zero.

### Output Columns

| Column | Definition |
|--------|-----------|
| `month` | Period number (1, 2, ..., WAM) |
| `date` | Calendar date (April 2019, May 2019, ...) |
| `beginning_balance` | Balance at start of period |
| `defaults` | Principal defaulting this period |
| `loss` | Unrecoverable portion of defaults |
| `recovery` | Recovered portion of defaults |
| `interest` | Interest earned on performing balance |
| `scheduled_principal` | Amortization principal |
| `prepayments` | Voluntary prepayments |
| `total_principal` | scheduled_principal + prepayments |
| `ending_balance` | Balance at end of period |
| `total_cashflow` | interest + total_principal + recovery |

---

## 6. IRR Calculation

Implemented in `src/cashflow_engine.py: calculate_irr()`.

### Cash Flow Construction

```
cf[0] = -(purchase_price * total_upb)    # Initial outlay (negative)
cf[1] = total_cashflow for month 1        # First inflow
cf[2] = total_cashflow for month 2
...
cf[N] = total_cashflow for month N        # Final inflow
```

### IRR Computation

```
monthly_irr = npf.irr(cf)                                # numpy_financial.irr
annual_irr = (1 + monthly_irr)^12 - 1                    # Compound annualization
```

**Verification**: After computing, NPV at the monthly IRR is checked:
```
npv = npf.npv(monthly_irr, cf)
assert |npv| < max(1.0, |cf[0]| * 1e-6)                  # Should be approximately zero
```

**Important**: Annualization uses compounding `(1 + r)^12 - 1`, NOT simple multiplication `r * 12`. The simple method would understate the true annual return.

**Worked example**:
- Cash flows: [-100, 60, 60]
- monthly_irr = npf.irr([-100, 60, 60]) = 0.01031
- annual_irr = (1 + 0.01031)^12 - 1 = **13.07%**
- NPV check: -100 + 60/1.01031 + 60/1.01031^2 = ~$0 (verified)

---

## 7. Price Solver

Implemented in `src/cashflow_engine.py: solve_price()`.

### Problem Statement

Given pool characteristics, CDR, CPR, loss severity, and a target IRR, find the purchase price (as a fraction of UPB) that achieves that IRR.

### Method

Uses `scipy.optimize.brentq` (Brent's method), a bracketed root-finding algorithm.

```
objective(price) = calculate_irr(project_cashflows(..., price), ..., price) - target_irr

solved_price = brentq(objective, 0.50, 1.50, xtol=1e-6)
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Lower bound | 0.50 | 50 cents on the dollar |
| Upper bound | 1.50 | 150 cents on the dollar (premium) |
| Tolerance | 1e-6 | Six decimal places of precision |

**Intuition**: Higher purchase price implies lower IRR (you pay more, earn less). The objective function is monotonically decreasing in price, so Brent's method converges efficiently.

If no root exists in [0.50, 1.50] (i.e., the target IRR is unachievable), the function returns `None`.

---

## 8. Scenario Analysis

Implemented in `src/scenario_analysis.py`.

### Multiplicative Shifts

Scenarios are built by applying **multiplicative** adjustments to the base-case CDR and CPR. Loss severity remains **fixed** across all scenarios.

| Scenario | CDR Formula | CPR Formula | Loss Severity |
|----------|------------|------------|---------------|
| Base | CDR_base | CPR_base | LS_base |
| Stress | CDR_base * (1 + stress_pct) | CPR_base * (1 - stress_pct) | LS_base |
| Upside | CDR_base * (1 - upside_pct) | CPR_base * (1 + upside_pct) | LS_base |

Default `stress_pct` = `upside_pct` = 0.15 (15%), adjustable via the dashboard slider (range 5% to 50%).

**Why multiplicative, not additive?** Multiplicative shifts maintain proportional relationships across cohorts. A cohort with 5% base CDR gets a 5.75% stress CDR (+15%), while a cohort with 15% base CDR gets 17.25%. Additive shifts would move both by the same absolute amount, which doesn't make economic sense.

**Why stress CPR decreases**: In a stress scenario, defaults rise and prepayments fall. Borrowers who would normally refinance (prepay) may be unable to due to tightening credit conditions. This is the standard convention in structured finance stress testing.

**Worked example** (stress_pct = 0.15):
- Base: CDR = 8%, CPR = 12%, Loss Severity = 85%
- Stress: CDR = 8% * 1.15 = **9.20%**, CPR = 12% * 0.85 = **10.20%**, Loss Severity = **85%**
- Upside: CDR = 8% * 0.85 = **6.80%**, CPR = 12% * 1.15 = **13.80%**, Loss Severity = **85%**

### Scenario Comparison

For each scenario, the engine:
1. Runs `project_cashflows()` with the scenario's CDR, CPR, and loss severity
2. Computes IRR via `calculate_irr()`
3. Aggregates: total interest, total principal, total losses, total recoveries
4. Computes **Weighted Average Life (WAL)**:

```
WAL = sum(month * total_principal) / sum(total_principal) / 12
```

WAL is expressed in **years**. It represents the average time (weighted by principal received) until principal is returned to the investor.

### Expected Ordering

Under normal conditions:
- **Stress IRR < Base IRR < Upside IRR** (higher defaults hurt returns)
- **Stress losses > Base losses > Upside losses** (more defaults = more losses)
- **Stress WAL > Base WAL > Upside WAL** (slower prepayments extend life)

---

## 9. State-Transition Cash Flow Model

Implemented in `src/cashflow_engine.py: project_cashflows_transition()`, with transition probabilities from `src/portfolio_analytics.py: compute_age_transition_probabilities()`.

This is the active model used by the dashboard. It uses the same pool-level WAC, WAM, and monthly_payment as the Simple model (Sections 5-8) but determines defaults and prepayments via age-specific transition probabilities rather than flat CDR/CPR rates.

### 7 States

| State | Type | Description |
|-------|------|-------------|
| Current | Transient | Performing loans |
| Delinquent (0-30) | Transient | Missed 1 payment (Grace + Late 16-30 combined) |
| Late_1 | Transient | 1st month of Late (31-120) |
| Late_2 | Transient | 2nd month of Late (31-120) |
| Late_3 | Transient | 3rd month of Late (31-120) |
| Charged Off | Absorbing | Terminal default |
| Fully Paid | Absorbing | Prepaid or matured |

### Delinquency Pipeline

Defaults flow through a 5-month pipeline instead of hitting immediately:

```
Current → Delinquent (0-30) → Late_1 → Late_2 → Late_3 → Charged Off
```

**Key consequence**: For an all-Current pool, the first defaults appear at month 5 (not month 1).

### Transition Probabilities

Probabilities are empirically derived from the dataset using `compute_age_transition_probabilities(df, bucket_size=1, states='7state')`. Each row contains:

```
(from_status, age) → {to_current_pct, to_delinquent_0_30_pct, to_late_1_pct,
                       to_late_2_pct, to_late_3_pct, to_charged_off_pct,
                       to_fully_paid_pct}
```

- Probabilities are looked up by individual loan age (monthly granularity)
- For ages beyond the data range, the nearest available age's probabilities are used
- Each row sums to 1.0

### Pool State

The pool is tracked as `{state: {loan_age: upb}}`:

```python
pool_state = {
    'states': {
        'Current': {10: 500000, 20: 300000, 30: 200000},
        'Delinquent (0-30)': {},
        'Late_1': {}, 'Late_2': {}, 'Late_3': {},
        'Charged Off': {}, 'Fully Paid': {},
    },
    'total_upb': 1000000,
    'wac': 0.1269,
    'wam': 32,
    'monthly_payment': 35000,
}
```

LC statuses are mapped to model states:
- Current → Current
- In Grace Period / Late (16-30 days) → Delinquent (0-30)
- Late (31-120 days) → Late_1, Late_2, or Late_3 (based on months since entering late status)

### Monthly Loop

For each month t = 1 to num_months:

```
1. TRANSITION: For each (state, age) with UPB > 0:
   - Look up transition probs for (from_status, age)
   - Multiply UPB by each probability → distribute to new states at age+1

2. CASH FLOWS:
   - interest = total_current_upb × (WAC / 12)
   - sched_principal = monthly_payment - interest  (capped ≥ 0, ≤ current_upb)
   - prepayments = Σ Current[age] × p(Current→Fully Paid | age)
   - new_defaults = Σ Late_3[age] × p(Late_3→Charged Off | age)
     + any other state→Charged Off transitions
   - losses = new_defaults × loss_severity
   - recoveries = new_defaults × recovery_rate
   - total_cashflow = interest + sched_principal + prepayments + recoveries

3. UPDATE STATE:
   - New Current[age+1] = stays + all cured
   - Subtract sched_principal proportionally from Current buckets
   - Roll other states forward (Delinq→Late_1, Late_1→Late_2, etc.)
   - Accumulate defaults and payoffs
```

### Prepayment Rate Override

The empirical `Current → Fully Paid` transition probabilities from the historical data include **all** payoffs — both voluntary prepayments and loans reaching maturity. This overstates the forward-looking prepayment rate for currently performing loans (many of which are mid-term).

Before projection, the engine calls `adjust_prepayment_rates(age_probs, cpr)` which:

1. Converts the pool-level CPR to a monthly SMM: `SMM = 1 - (1 - CPR)^(1/12)`
2. Replaces all `Current → Fully Paid` probabilities with this constant SMM
3. Adds the difference back to `Current → Current` to maintain row sum = 1.0

This means the **state-transition model drives defaults only** (through the delinquency pipeline), while **prepayments use the same constant CPR** as the Simple model. The CPR is derived from the pool's historical single-month prepayment behavior (from `compute_pool_assumptions()`), and the user can override it on Tab 2.

**Example**: CPR = 1.49% → SMM = 1 - (1 - 0.0149)^(1/12) = 0.00125 (0.125%/month). This replaces the empirical ~2.8%/month rate.

### Key Design Decisions

- **Interest from Current UPB only**: Only performing (Current) loans accrue interest.
- **Prepayments use CPR-derived SMM, not empirical transition rates**: Forward-looking prepayment behavior is driven by the constant pool-level CPR, not the historical `Current → Fully Paid` rate (which includes maturity payoffs).
- **Defaults from Late_3→Charged Off**: Primary default path (other state→Charged Off transitions also captured).
- **Losses NOT subtracted from investor cashflow**: `total_cashflow = interest + sched_principal + prepayments + recoveries`. Losses are tracked but not deducted (investor receives recovery portion).
- **Same pool-level WAC/WAM/monthly_payment as Simple model**: Only the default/prepayment mechanism differs.

### Price Solver (Transition)

Implemented in `src/cashflow_engine.py: solve_price_transition()`.

Since the projection is independent of purchase price (price only affects the initial outlay in IRR), the engine projects cash flows once and then uses brentq to solve for price:

```python
cf_df = project_cashflows_transition(...)  # Project once
def objective(price):
    return calculate_irr(cf_df, pool_chars, price) - target_irr
solved_price = brentq(objective, 0.50, 1.50)
```

### Transition Scenario Analysis

Implemented in `src/scenario_analysis.py: build_scenarios_transition()`.

Instead of shifting CDR/CPR, scenarios shift the transition probabilities themselves:

**Stress** (per row of the transition matrix):
- `Current→Delinquent` × `(1 + stress_pct)` — more delinquency
- `Current→Fully Paid` × `(1 - stress_pct)` — less prepayment
- All cure rates (any non-Current state → Current) × `(1 - stress_pct)` — harder to cure
- `Late_3→Charged Off` is NOT directly stressed — it increases mechanically via re-normalization
- After multipliers: clamp all probs to [0, 1], then adjust residual column to make row sum to 1

**Upside**: opposite multipliers.

**Loss severity**: FIXED across all scenarios (same as Simple model).

**Expected ordering**: Stress IRR < Base IRR < Upside IRR.

---

## Assumptions & Limitations

1. **Pool-level aggregate model**: Cash flows are projected for the pool as a single entity (one balance, one WAC, one WAM). Individual loan behavior is not modeled.

2. **Static CDR/CPR**: Default and prepayment rates are constant throughout the projection. In reality, these rates vary over the life of the pool (seasoning effects, economic cycles).

3. **No reinvestment**: Cash flows received by the investor are not reinvested. IRR inherently assumes reinvestment at the IRR rate.

4. **Historical base case**: Base CDR, CPR, and loss severity are derived from the historical performance of the filtered cohort. Past performance does not guarantee future results.

5. **Fixed monthly payment**: The pool's monthly payment stays constant throughout the projection, matching the sum of individual loan installments at origination. As the pool balance decreases, the interest-to-principal ratio shifts naturally.

6. **March 2019 snapshot**: All projections are forward-looking from March 2019, the snapshot date of the Lending Club data. Dates in the output start at April 2019 (t=1).

7. **No credit enhancement**: The model assumes the investor holds the entire pool directly with no tranching, overcollateralization, or other structural protections.
