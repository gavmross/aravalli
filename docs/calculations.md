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

### CDR -> MDR (Default Rates)

**CDR (Cumulative Default Rate)** is the historical annual default fraction. **MDR (Monthly Default Rate)** is the monthly equivalent used in projections.

```
MDR = 1 - (1 - CDR)^(1/12)
```

**Worked example**:
- CDR = 0.10 (10% annual default rate)
- MDR = 1 - (1 - 0.10)^(1/12) = 1 - 0.90^(1/12) = **0.877%** monthly
- Sanity check: 1 - (1 - 0.00877)^12 = 0.10 (verified)

**Important**: Do NOT use `CDR / 12` as an approximation. The compounding formula is exact and must be used.

---

## 3. Pool-Level Assumptions

Implemented in `src/cashflow_engine.py: compute_pool_assumptions()`.

### CDR (Cumulative Default Rate)

Computed from **ALL loans** in the filtered strata:

```
defaulted_upb = sum(funded_amnt - total_rec_prncp)   [for Charged Off loans only]
total_originated_upb = sum(funded_amnt)               [for ALL loans in cohort]
CDR = defaulted_upb / total_originated_upb
```

| Variable | Source | Scope |
|----------|--------|-------|
| `funded_amnt` | Original loan amount | All loans |
| `total_rec_prncp` | Principal received to date | Charged Off loans |
| `defaulted_upb` | Unrecovered principal at charge-off | Charged Off loans |

**Rationale**: CDR uses all loans in the denominator (including Current, Fully Paid, and Charged Off) to represent the true default rate for the cohort.

### CPR (Conditional Prepayment Rate)

Computed from **Current loans with March 2019 last payment date only**:

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

Implemented in `src/cashflow_engine.py: compute_pool_characteristics()`. All computed from **Current loans with March 2019 last payment date only**.

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

Sum of all individual loan installments for the Current March 2019 pool.

---

## 5. Monthly Cash Flow Projection

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

## Assumptions & Limitations

1. **Pool-level aggregate model**: Cash flows are projected for the pool as a single entity (one balance, one WAC, one WAM). Individual loan behavior is not modeled.

2. **Static CDR/CPR**: Default and prepayment rates are constant throughout the projection. In reality, these rates vary over the life of the pool (seasoning effects, economic cycles).

3. **No reinvestment**: Cash flows received by the investor are not reinvested. IRR inherently assumes reinvestment at the IRR rate.

4. **Historical base case**: Base CDR, CPR, and loss severity are derived from the historical performance of the filtered cohort. Past performance does not guarantee future results.

5. **Fixed monthly payment**: The pool's monthly payment stays constant throughout the projection, matching the sum of individual loan installments at origination. As the pool balance decreases, the interest-to-principal ratio shifts naturally.

6. **March 2019 snapshot**: All projections are forward-looking from March 2019, the snapshot date of the Lending Club data. Dates in the output start at April 2019 (t=1).

7. **No credit enhancement**: The model assumes the investor holds the entire pool directly with no tranching, overcollateralization, or other structural protections.
