# Calculations Guide

A comprehensive reference for every calculation in the codebase: what it is, the exact formula, where it lives in the code, what data it operates on, and the key assumptions behind it.

---

## Table of Contents

1. [Loan-Level Amortization](#1-loan-level-amortization)
2. [Pool-Level Weighted Averages](#2-pool-level-weighted-averages)
3. [Conditional Default Rate (CDR)](#3-conditional-default-rate-cdr)
4. [Conditional Prepayment Rate (CPR)](#4-conditional-prepayment-rate-cpr)
5. [Loss Severity & Recovery Rate](#5-loss-severity--recovery-rate)
6. [Cumulative Default Rate](#6-cumulative-default-rate)
7. [Timeline Reconstruction (Backsolve)](#7-timeline-reconstruction-backsolve)
8. [Transition Probabilities](#8-transition-probabilities)
9. [State-Transition Cash Flow Projection](#9-state-transition-cash-flow-projection)
10. [Simple (Flat CDR/CPR) Cash Flow Projection](#10-simple-flat-cdrcpr-cash-flow-projection)
11. [IRR Calculation](#11-irr-calculation)
12. [Price Solver](#12-price-solver)
13. [Scenario Analysis](#13-scenario-analysis)
14. [Credit Metrics](#14-credit-metrics)
15. [Performance Metrics](#15-performance-metrics)
16. [Transition Matrix (Flow-Based)](#16-transition-matrix-flow-based)
17. [Display-Only Analytics](#17-display-only-analytics)

---

## Populations

Before diving into formulas, it's critical to understand which loans feed into which calculations. Every metric operates on one of three populations:

| Population | Definition | Used For |
|-----------|-----------|---------|
| **All Loans** | Every loan matching the sidebar filter (all statuses) | CDR, cumulative default rate, loss severity, credit metrics, performance metrics, transition matrix |
| **Active Loans** | Current (with `last_pymnt_d = 2019-03-01`) + In Grace Period + Late (16-30 days) + Late (31-120 days) (regardless of last payment date) | WAC, WAM, WALA, pool UPB, monthly payment, cash flow projections |
| **CPR Loans** | Current (`last_pymnt_d = 2019-03-01`) + Fully Paid (`last_pymnt_d = 2019-03-01`) | CPR, pct_prepaid (delinquent loans don't prepay) |
| **Charged Off Loans** | Loans with `loan_status = 'Charged Off'` within the filter | Loss severity, recovery rate |

The key distinctions:
- **Active loans** define the investable pool for cash flow projections. Current loans must have a March 2019 last payment date (proving they were actively paying). Delinquent loans are included regardless of their last payment date because they are, by definition, behind on payments.
- **CPR uses Current + Fully Paid (March 2019) loans** because prepayment is a behavior of performing borrowers. Fully Paid loans with a March 2019 last payment date paid off during the observation month — their payoff above scheduled principal is a prepayment event. Delinquent loans are behind on scheduled payments — they are not prepaying.

---

## 1. Loan-Level Amortization

**File**: `src/amortization.py`
**Function**: `calc_amort()` (line 102)
**Population**: All loans (run once on the full dataset)

### Monthly Payment (PMT)

```
PMT = P * r(1+r)^n / ((1+r)^n - 1)
```

Where: P = funded amount, r = int_rate / 12, n = term_months

**Code**: `calc_monthly_payment()` (line 12)

**Example**: $10,000 at 10.78% for 36 months
- r = 0.1078 / 12 = 0.008983
- PMT = 10000 * 0.008983 * 1.008983^36 / (1.008983^36 - 1) = **$325.17**

### Remaining Balance After k Payments

Computed iteratively in `calc_balance()` (line 39). For each payment period:
1. Interest = remaining_balance * monthly_rate
2. Principal = min(monthly_payment - interest, remaining_balance)
3. New balance = remaining_balance - principal

### Columns Added by `calc_amort()`

| Column | Formula |
|--------|---------|
| `orig_exp_monthly_payment` | PMT formula above |
| `orig_exp_payments_made` | Months between `issue_d` and `last_pymnt_d` |
| `orig_exp_balance` | Iterative balance after payments_made periods |
| `orig_exp_principal_paid` | Cumulative principal paid through payments_made |
| `last_pmt_beginning_balance` | `(out_prncp + last_pymnt_amnt) / (1 + monthly_rate)` |
| `last_pmt_scheduled_principal` | `installment - (beginning_balance * monthly_rate)` |
| `last_pmt_unscheduled_principal` | `actual_principal - scheduled_principal` (prepayment) |
| `last_pmt_smm` | `unscheduled_principal / (beginning_balance - scheduled_principal)` |
| `last_pmt_cpr` | `1 - (1 - SMM)^12` |
| `updated_remaining_term` | `-ln(1 - r*balance/payment) / ln(1+r)`, rounded up |

---

## 2. Pool-Level Weighted Averages

**File**: `src/cashflow_engine.py`
**Function**: `compute_pool_characteristics()` (line 163)
**Population**: Active loans — Current with `last_pymnt_d = 2019-03-01` plus all delinquent (In Grace Period + Late 16-30 + Late 31-120) regardless of last payment date

### WAC (Weighted Average Coupon)

```
WAC = sum(int_rate * out_prncp) / sum(out_prncp)
```

Weights by current outstanding principal, not original funded amount. Result is a decimal (e.g., 0.1269 = 12.69%).

### WAM (Weighted Average Maturity)

```
WAM = round(sum(updated_remaining_term * out_prncp) / sum(out_prncp))
```

`updated_remaining_term` comes from `calc_amort()`. Result is rounded to the nearest integer (months). This determines how many months the cash flow projection runs.

### WALA (Weighted Average Loan Age)

```
WALA = sum(orig_exp_payments_made * out_prncp) / sum(out_prncp)
```

Computed in `calculate_credit_metrics()` (`src/portfolio_analytics.py`, line 220). Weights by current UPB for Current loans.

### Monthly Payment (Pool-Level)

```
monthly_payment = sum(installment)
```

Simple sum of all individual loan installments in the active pool.

---

## 3. Conditional Default Rate (CDR)

**File**: `src/cashflow_engine.py`
**Function**: `compute_pool_assumptions()` (line 28)
**Population**: All loans in the filtered strata (for defaults and performing balance); requires `reconstruct_loan_timeline()` columns

This is the most complex calculation in the codebase. It uses the **dv01 conditional methodology**: observe monthly default rates over the trailing 12 months, average them, then annualize.

### Step-by-Step

**Step 1: Compute Monthly Default Rates (MDR) for each of the 12 trailing months (April 2018 - March 2019)**

For each month M in the trailing window: 

```
MDR_M = default_UPB_in_month_M / performing_balance_at_start_of_month_M
```

- **default_UPB_in_month_M**: Sum of `(funded_amnt - total_rec_prncp)` for loans whose `default_month` falls within month M. `default_month` is reconstructed by `reconstruct_loan_timeline()` as `last_pymnt_d + 5 months` for Charged Off loans (delinquent_month + 4 months).

- **performing_balance_at_start_of_month_M**: For every loan that was (a) originated before month M, (b) not yet defaulted, and (c) not yet paid off — estimate the outstanding balance using the standard amortization formula based on loan age at that point. This is computed by calling `calc_monthly_payment()` and `calc_balance()` for each performing loan. (Lines 82-98)

**Step 2: Average the 12 monthly MDRs**

```
avg_MDR = mean(MDR_1, MDR_2, ..., MDR_12)
```

**Step 3: Annualize to CDR**

```
CDR = 1 - (1 - avg_MDR)^12
```

This is **not** simply `avg_MDR * 12`. The compounding formula accounts for the shrinking pool.

### Key Assumptions

1. **Trailing window**: April 2018 through March 2019 (12 months ending at the snapshot date).
2. **Performing balance is estimated with prepayment adjustment**. We reconstruct each loan's expected balance at the start of each month using the amortization formula, then adjust downward for active loans where we can observe `out_prncp < scheduled_balance` at the snapshot (spreading the cumulative prepayment evenly across the loan's age). Fully Paid and Charged Off loans use the unadjusted schedule (no calibration endpoint). This avoids overstating the denominator when loans have been prepaying.
3. **Default timing is backsolve-derived**: `default_month = delinquent_month + 4 months`. This assumes the standard Lending Club timeline of ~120 days from first missed payment to charge-off.
4. **Cohort-specific**: CDR is computed for whatever strata the user has filtered to (e.g., "Grade B, 36-month, 2018Q1"), not a global average.
5. **Round-trip property**: When the cash flow engine needs a monthly rate, it converts back: `MDR = 1 - (1 - CDR)^(1/12)` which recovers approximately the avg_MDR.

### Verification

The function returns `monthly_mdrs` (list of 12 MDRs) and `avg_mdr` alongside the annualized `cdr`, allowing verification that the round-trip holds.

---

## 4. Conditional Prepayment Rate (CPR)

**File**: `src/cashflow_engine.py`
**Function**: `compute_pool_assumptions()` (line 116)
**Also**: `src/portfolio_analytics.py`, `calculate_performance_metrics()` (line 369) — same logic
**Population**: Current loans only (`loan_status = 'Current'` with `last_pymnt_d = 2019-03-01`). Delinquent loans are excluded — they are behind on payments, not prepaying.

### Formula

**Step 1: Compute pool-level SMM (Single Monthly Mortality)**

```
SMM = total_unscheduled_principal / (total_beginning_balance - total_scheduled_principal)
```

Where (aggregated across all active loans with `last_pmt_beginning_balance > 0`):
- `total_unscheduled_principal` = sum of `last_pmt_unscheduled_principal` (prepayment amounts from `calc_amort()`)
- `total_beginning_balance` = sum of `last_pmt_beginning_balance`
- `total_scheduled_principal` = sum of `last_pmt_scheduled_principal`

**Step 2: Annualize to CPR**

```
CPR = 1 - (1 - SMM)^12
```

### Key Assumptions

1. **Current + Fully Paid (March 2019) loans**: CPR is computed from Current loans plus Fully Paid loans, both with March 2019 last payment date. Fully Paid March 2019 loans paid off during the observation month — their payoff above scheduled principal is a prepayment event. Delinquent loans (In Grace, Late 16-30, Late 31-120) are excluded because they are behind on scheduled payments — they are not prepaying.
2. **Single-period observation**: This CPR is based on the most recent payment period only (the last month in the dataset). It is not a multi-month average like CDR.
3. **Pool-level aggregation**: SMM is computed at the pool level (sum of unscheduled principal / sum of eligible balances), not as an average of individual loan SMMs.
4. **Prepayment definition**: Unscheduled principal = actual principal paid minus the amortization-scheduled principal. This captures voluntary prepayments and extra payments, but not full payoffs of matured loans (those are already Fully Paid).
5. **Transition model usage**: In the dashboard's state-transition model, the empirical CPR is used to replace the historical Current → Fully Paid transition rate via `adjust_prepayment_rates()` (`src/cashflow_engine.py`, line 403). This prevents overstating prepayments from maturity payoffs.

---

## 5. Loss Severity & Recovery Rate

**File**: `src/cashflow_engine.py`
**Function**: `compute_pool_assumptions()` (line 132)
**Also**: `src/portfolio_analytics.py`, `calculate_performance_metrics()` (line 411)
**Population**: Charged Off loans only

### Formula

```
exposure = funded_amnt - total_rec_prncp          (what was still owed)
capped_recoveries = min(recoveries, exposure)      (cap at 100%)
net_loss = exposure - capped_recoveries

loss_severity = sum(net_loss) / sum(exposure)       (for loans with exposure > 0)
recovery_rate = sum(capped_recoveries) / sum(exposure)
```

**Identity**: `loss_severity + recovery_rate = 1.0` (exactly, by construction)

### Why Cap Recoveries?

A small number of loans (4 in the full dataset) have `recoveries > exposure`, meaning the servicer collected more than the remaining balance. Without capping, recovery_rate would exceed 100% and the identity would break. Capping at exposure ensures the math stays clean.

### Key Assumptions

1. **Loss severity is FIXED across all scenarios.** Stress and upside scenarios change transition probabilities (or CDR/CPR), but loss severity stays constant. The rationale: loss severity is a function of collateral recovery, not credit deterioration speed.
2. **Cohort-specific**: Like CDR, loss severity reflects the filtered strata's actual experience.

---

## 6. Cumulative Default Rate

**File**: `src/cashflow_engine.py`
**Function**: `compute_pool_assumptions()` (line 108)
**Population**: All loans

```
cumulative_default_rate = sum(funded_amnt - total_rec_prncp for Charged Off) / sum(funded_amnt for ALL)
```

This is the raw lifetime default rate. It is returned for display/reference only and is **not** used in cash flow projections. The conditional CDR (Section 3) is the rate that drives the cash flow engine.

Do **not** call this "CDR" — CDR refers exclusively to the conditional (annualized) rate.

---

## 7. Timeline Reconstruction (Backsolve)

**File**: `src/portfolio_analytics.py`
**Function**: `reconstruct_loan_timeline()` (line 734)
**Population**: All loans

Since we have a single point-in-time snapshot (March 2019) rather than monthly servicing tapes, we reconstruct approximate monthly loan statuses using backsolve logic based on Lending Club's deterministic delinquency progression.

### Reconstruction Rules

| Transition | Timing | Column |
|-----------|--------|--------|
| First missed payment (Delinquent 0-30) | `last_pymnt_d + 1 month` | `delinquent_month` |
| Late (31-120 days) | `delinquent_month + 1 month` | `late_31_120_month` |
| Late_1 | Same as `late_31_120_month` | `late_1_month` |
| Late_2 | `late_31_120_month + 1 month` | `late_2_month` |
| Late_3 | `late_31_120_month + 2 months` | `late_3_month` |
| Charged Off | `delinquent_month + 4 months` | `default_month` |
| Fully Paid | `last_pymnt_d` | `payoff_month` |

### Loan Age

```
loan_age_months = round((snapshot_date - issue_d) / 30.44)
```

Capped at `term_months - 1` so a 36-month loan has ages 0-35.

### State Determination at a Given Age

`get_loan_status_at_age()` (line 859) determines each loan's status at any historical age. In **7-state mode**, delinquent states are transient (1 month each):

- **Priority** (highest wins): Charged Off > Fully Paid > Late_3 > Late_2 > Late_1 > Delinquent (0-30) > Current
- Charged Off and Fully Paid are **absorbing** (once entered, permanent from that age forward)
- Delinquent (0-30), Late_1, Late_2, Late_3 are each exactly 1 month in duration

### Key Assumptions

1. Lending Club's progression is **deterministic**: Current → Delinquent (0-30) → Late (31-120) → Charged Off takes exactly 5 months.
2. **Cured loans** are identified via `curr_paid_late1_flag`: loans that reached Late (16-30 days) but returned to Current or Fully Paid.
3. Late (31-120) loans **cannot cure** — they either stay late or charge off.
4. The data only supports backsolve; there are no actual monthly servicing records.

---

## 8. Transition Probabilities

**File**: `src/portfolio_analytics.py`
**Function**: `compute_age_transition_probabilities()` (line 942)
**Population**: All loans (historical transitions needed)

### How Probabilities Are Computed

For every pair of consecutive loan ages (N, N+1), the function:

1. Calls `get_loan_status_at_age(df, N)` and `get_loan_status_at_age(df, N+1)` for all loans
2. Counts transitions: how many loans went from state X at age N to state Y at age N+1
3. Groups by age bucket (default: 1 month for 7-state model)
4. Normalizes each row to probabilities (counts / total observations for that from_status at that age)

### The 7 States

```
Current → Delinquent (0-30) → Late_1 → Late_2 → Late_3 → Charged Off
                                                          → Fully Paid (absorbing)
```

### Prepayment Rate Adjustment

The raw empirical Current → Fully Paid transition rate includes both voluntary prepayments and natural maturity payoffs. Since matured loans inflate this rate, the dashboard calls `adjust_prepayment_rates()` (`src/cashflow_engine.py`, line 403) to replace it with the CPR-derived SMM:

```
SMM = 1 - (1 - CPR)^(1/12)
```

The difference is shifted into Current → Current to maintain row sum = 1.0.

### Output

A DataFrame with columns: `age_bucket`, `from_status`, `to_current_pct`, `to_delinquent_0_30_pct`, `to_late_1_pct`, `to_late_2_pct`, `to_late_3_pct`, `to_charged_off_pct`, `to_fully_paid_pct`, `observation_count`.

### Key Assumptions

1. Probabilities are **age-specific**: a 6-month-old loan has different transition rates than a 24-month-old loan.
2. Probabilities are **empirical**: derived from the actual historical behavior of all loans in the cohort.
3. The 5-month default pipeline (Current → Delinquent → Late_1 → Late_2 → Late_3 → Charged Off) means an all-Current pool will have zero defaults for the first 4 months of projection.

---

## 9. State-Transition Cash Flow Projection

**File**: `src/cashflow_engine.py`
**Function**: `project_cashflows_transition()` (line 616)
**Population**: Active loans via `build_pool_state()` — Current with `last_pymnt_d = 2019-03-01` plus all delinquent (In Grace Period + Late 16-30 + Late 31-120) regardless of last payment date

This is the **primary model used by the dashboard** (Tabs 2 and 3). It tracks pool UPB across all 7 states with age-specific transition probabilities.

### Pool State Construction

`build_pool_state()` (line 448) maps each active loan to one of the 7 model states, grouped by loan age:

| LC Status | Model State |
|----------|------------|
| Current | Current |
| In Grace Period | Delinquent (0-30) |
| Late (16-30 days) | Delinquent (0-30) |
| Late (31-120 days) | Late_1, Late_2, or Late_3 (based on months in late status) |

The output is `{state: {age: total_UPB}}`.

### Monthly Loop (t = 1 to num_months)

For each month:

**1. Apply transitions**: For every (state, age) bucket with UPB > 0, look up transition probabilities for that (from_status, age). Distribute UPB to destination states:

```
for each (state, age, upb):
    probs = lookup(from_status=state, age=age)
    for to_state, p in probs:
        flow = upb * p
        if to_state == 'Charged Off': → new_defaults += flow
        elif to_state == 'Fully Paid' and state == 'Current': → prepayments += flow
        else: → new_states[to_state][age+1] += flow
```

**2. Compute scheduled principal** (from Current UPB only):

```
fraction = current_upb / notional_upb
scaled_payment = pool_monthly_payment * fraction
interest = current_upb * monthly_rate
sched_principal = max(0, min(scaled_payment - interest, current_upb))
```

The `notional_upb` tracks standard amortization independently of transitions. When loans leave Current, the payment scales down proportionally.

**3. Apply defaults**:

```
losses = new_defaults * loss_severity
recoveries = new_defaults * recovery_rate
```

**4. Total cash flow to investor**:

```
total_cashflow = interest + sched_principal + prepayments + recoveries
```

### Key Assumptions

1. **Defaults flow through a 5-month pipeline**: Current → Delinquent → Late_1 → Late_2 → Late_3 → Charged Off. An all-Current pool has zero defaults for months 1-4.
2. **Interest is earned only on Current UPB**, not on delinquent balances.
3. **Prepayments come only from Current → Fully Paid transitions.**
4. **Loss severity and recovery rate are applied immediately** when a loan reaches Charged Off.
5. **Age probabilities nearest-match**: If no probability exists for a specific age, the nearest available age (capped at the maximum observed) is used.

---

## 10. Simple (Flat CDR/CPR) Cash Flow Projection

**File**: `src/cashflow_engine.py`
**Function**: `project_cashflows()` (line 197)
**Population**: Active loans — Current with `last_pymnt_d = 2019-03-01` plus all delinquent (In Grace Period + Late 16-30 + Late 31-120) regardless of last payment date (via pool characteristics)

This is the simpler model retained in the codebase for reference/testing. It is **not used by the dashboard**.

### Pre-Loop Setup

```
MDR = 1 - (1 - CDR)^(1/12)
SMM = 1 - (1 - CPR)^(1/12)
monthly_rate = WAC / 12
```

### Monthly Loop (t = 1 to WAM)

```
For each month t:
    beginning_balance = previous ending_balance (or total_upb for t=1)

    defaults = beginning_balance * MDR
    loss = defaults * loss_severity
    recovery = defaults * (1 - loss_severity)

    performing_balance = beginning_balance - defaults
    interest = performing_balance * monthly_rate

    scheduled_principal = min(max(monthly_payment - interest, 0), performing_balance)
    prepayments = (performing_balance - scheduled_principal) * SMM

    total_principal = scheduled_principal + prepayments
    ending_balance = max(performing_balance - total_principal, 0)

    total_cashflow = interest + total_principal + recovery
```

### Key Differences from State-Transition Model

- Defaults hit the pool **immediately** each month (no pipeline delay)
- Single CDR/CPR rate applied uniformly (not age-specific)
- No delinquency state tracking

---

## 11. IRR Calculation

**File**: `src/cashflow_engine.py`
**Function**: `calculate_irr()` (line 300)

### Formula

```
cf[0] = -(purchase_price * total_upb)    # Initial outlay (negative)
cf[1..N] = total_cashflow per month       # Monthly inflows

monthly_irr = npf.irr(cf)                 # numpy_financial root-finder
annual_irr = (1 + monthly_irr)^12 - 1     # Compound annualization
```

**Important**: Annualization uses compounding `(1 + r)^12 - 1`, not simple multiplication `r * 12`.

### Verification

After computing IRR, the function checks that NPV at the solved rate is approximately zero:

```
npv_check = npf.npv(monthly_irr, cf)
```

---

## 12. Price Solver

**File**: `src/cashflow_engine.py`
**Functions**: `solve_price()` (line 342), `solve_price_transition()` (line 801)

### Problem

Given a target IRR (e.g., 12%), find the purchase price (as a fraction of UPB) that achieves that return.

### Method

Uses `scipy.optimize.brentq` (bisection root-finder) on the interval [0.50, 1.50]:

```
objective(price) = IRR(price) - target_irr
solved_price = brentq(objective, 0.50, 1.50, xtol=1e-6)
```

### Optimization for Transition Model

`solve_price_transition()` projects cash flows **once** (since cash flows don't depend on purchase price — price only affects the initial outlay in the IRR calculation). Then it solves for price by varying only the initial outlay. This is much faster than re-projecting for every trial price.

### Intuition

Higher price → lower IRR (you pay more, earn less). Lower price → higher IRR. If the objective doesn't cross zero in [0.50, 1.50], the target IRR is unachievable and the function returns `None`.

---

## 13. Scenario Analysis

### State-Transition Model (Dashboard)

**File**: `src/scenario_analysis.py`
**Function**: `build_scenarios_transition()` (line 162)

Applies multiplicative shifts to **transition probabilities**, not to CDR/CPR directly.

| What Changes | Stress (shift_sign = +1) | Upside (shift_sign = -1) |
|-------------|-------------------------|--------------------------|
| Current → Delinquent | `× (1 + stress_pct)` | `× (1 - upside_pct)` |
| Current → Fully Paid | `× (1 - stress_pct)` | `× (1 + upside_pct)` |
| Cure rates (non-Current → Current) | `× (1 - stress_pct)` | `× (1 + upside_pct)` |
| Late_3 → Charged Off | **NOT directly stressed** | **NOT directly stressed** |
| Loss severity | **FIXED** | **FIXED** |

After multipliers are applied, each row is re-normalized so probabilities sum to 1.0. The residual (difference from 1.0) is absorbed by:
- Current → Current (for Current rows)
- The "roll forward" column (e.g., Delinquent → Late_1, Late_1 → Late_2, etc.)

**Late_3 → Charged Off increases mechanically** under stress: when cure rates decrease, less UPB exits the pipeline, so more accumulates and reaches Charged Off via re-normalization.

Default stress/upside percentage: 15% (user-adjustable via slider).

### Simple Model (Reference Only)

**File**: `src/scenario_analysis.py`
**Function**: `build_scenarios()` (line 52)

```
Stress: CDR × (1 + stress_pct), CPR × (1 - stress_pct)
Upside: CDR × (1 - upside_pct), CPR × (1 + upside_pct)
Loss severity: FIXED across all scenarios
```

### Scenario Comparison

`compare_scenarios_transition()` (line 270) runs `project_cashflows_transition()` for each scenario's probability set and computes IRR + summary metrics.

---

## 14. Credit Metrics

**File**: `src/portfolio_analytics.py`
**Function**: `calculate_credit_metrics()` (line 12)
**Population**: All loans, stratified by a chosen column

### Origination Metrics (weighted by `funded_amnt`)

| Metric | Formula |
|--------|---------|
| orig_wac | `sum(int_rate * funded_amnt) / sum(funded_amnt)` |
| orig_wam | `sum(term_months * funded_amnt) / sum(funded_amnt)` |
| orig_avg_fico | `sum(original_fico * funded_amnt) / sum(funded_amnt)` |
| orig_avg_dti | `sum(dti_clean * funded_amnt) / sum(funded_amnt)` |

### Active Metrics (weighted by `out_prncp`, Current loans only)

| Metric | Formula |
|--------|---------|
| curr_wac | `sum(int_rate * out_prncp) / sum(out_prncp)` |
| curr_wam | `sum(updated_remaining_term * out_prncp) / sum(out_prncp)` |
| curr_wala | `sum(orig_exp_payments_made * out_prncp) / sum(out_prncp)` |
| curr_avg_fico | `sum(latest_fico * out_prncp) / sum(out_prncp)` |

### UPB Breakdown

Active UPB is broken into Current, In Grace Period, Late (16-30), Late (31-120) as percentages of active total.

---

## 15. Performance Metrics

**File**: `src/portfolio_analytics.py`
**Function**: `calculate_performance_metrics()` (line 257)
**Population**: All loans, grouped by vintage (`issue_quarter`)

Per vintage, computes:

| Metric | Formula |
|--------|---------|
| pct_defaulted_upb | `sum(funded_amnt - total_rec_prncp for Charged Off) / sum(funded_amnt)` |
| pool_cpr | Same as Section 4 (CPR), computed from active loans |
| pct_prepaid | `sum(total_rec_prncp - orig_exp_principal_paid for active) / sum(orig_exp_principal_paid for active)` |
| loss_severity | Same as Section 5 |
| recovery_rate | Same as Section 5 |

---

## 16. Transition Matrix (Flow-Based)

**File**: `src/portfolio_analytics.py`
**Function**: `calculate_transition_matrix()` (line 499)
**Population**: All loans

This is a **different** calculation from the age-specific transition probabilities (Section 8). It's a cumulative lifetime flow analysis showing what fraction of loans progressed through each delinquency stage.

### Flow Logic

All loans start as Current. The matrix traces:

| From | To | How Counted |
|------|----|-------------|
| Current → Fully Paid (clean) | Fully Paid with `curr_paid_late1_flag = 0` | Never reached Late 16-30 |
| Current → Still Current (clean) | Current with `curr_paid_late1_flag = 0` | Never reached Late 16-30 |
| Current → Delinquent | Everyone else | Entered the delinquency pipeline |
| Late 16-30 → Cured | `curr_paid_late1_flag = 1` | Reached late then recovered |
| Late 31-120 → Charged Off | `loan_status = 'Charged Off'` | Terminal default |

---

## 17. Display-Only Analytics

These functions produce visualizations and tables in Tab 1 but do **not** feed into the cash flow engine.

### Dollar-Flow Transition Matrix

**Function**: `compute_pool_transition_matrix()` (line 1099)

Applies age-specific transition probabilities (from Section 8) to the current pool's actual UPB distribution by age bucket, producing expected dollar flows for the next month.

### Default Timing Distribution

**Function**: `compute_default_timing()` (line 1231)

Shows at what loan age (months since origination) defaults occur. Uses `default_age` from the timeline reconstruction.

### Loan Age Status Distribution

**Function**: `compute_loan_age_status_matrix()` (line 1301)

Cross-sectional snapshot showing the percentage of loans in each status (Current, Fully Paid, Charged Off, Late/Grace) at each age bucket.

---

## Formula Quick Reference

| Rate | Formula | Direction |
|------|---------|-----------|
| SMM → CPR | `CPR = 1 - (1 - SMM)^12` | Monthly → Annual |
| CPR → SMM | `SMM = 1 - (1 - CPR)^(1/12)` | Annual → Monthly |
| MDR → CDR | `CDR = 1 - (1 - MDR)^12` | Monthly → Annual |
| CDR → MDR | `MDR = 1 - (1 - CDR)^(1/12)` | Annual → Monthly |
| Monthly IRR → Annual | `annual = (1 + monthly)^12 - 1` | Compound |
| Loss + Recovery | `loss_severity + recovery_rate = 1.0` | Identity |

---

## File Map

| Calculation | File | Key Function(s) |
|------------|------|-----------------|
| Amortization (loan-level) | `src/amortization.py` | `calc_amort()`, `calc_monthly_payment()`, `calc_balance()` |
| WAC, WAM, pool UPB | `src/cashflow_engine.py` | `compute_pool_characteristics()` |
| CDR, CPR, loss severity | `src/cashflow_engine.py` | `compute_pool_assumptions()` |
| Timeline reconstruction | `src/portfolio_analytics.py` | `reconstruct_loan_timeline()` |
| Transition probabilities | `src/portfolio_analytics.py` | `compute_age_transition_probabilities()`, `get_loan_status_at_age()` |
| Cash flow projection (transition) | `src/cashflow_engine.py` | `build_pool_state()`, `project_cashflows_transition()` |
| Cash flow projection (simple) | `src/cashflow_engine.py` | `project_cashflows()` |
| IRR | `src/cashflow_engine.py` | `calculate_irr()` |
| Price solver | `src/cashflow_engine.py` | `solve_price()`, `solve_price_transition()` |
| Scenario analysis | `src/scenario_analysis.py` | `build_scenarios_transition()`, `compare_scenarios_transition()` |
| Credit metrics | `src/portfolio_analytics.py` | `calculate_credit_metrics()` |
| Performance metrics | `src/portfolio_analytics.py` | `calculate_performance_metrics()` |
| Transition matrix (flow) | `src/portfolio_analytics.py` | `calculate_transition_matrix()` |
