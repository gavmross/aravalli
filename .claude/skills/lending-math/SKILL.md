# Skill: Consumer Lending Financial Math

Reference this skill whenever implementing, modifying, or validating financial calculations in `src/cashflow_engine.py`, `src/scenario_analysis.py`, or `src/amortization.py`.

---

## Rate Conversions

### SMM ↔ CPR (Prepayment)

**SMM (Single Monthly Mortality)** — the fraction of the pool that prepays in a single month:

```
SMM = unscheduled_principal / (beginning_balance - scheduled_principal)
```

**CPR (Conditional Prepayment Rate)** — annualized prepayment rate:

```
CPR = 1 - (1 - SMM)^12
```

**Inverse** (used in cash flow projections when CPR is the input):

```
SMM = 1 - (1 - CPR)^(1/12)
```

**Worked example:**
- If SMM = 0.01 (1% of pool prepays this month)
- CPR = 1 - (1 - 0.01)^12 = 1 - 0.99^12 = 1 - 0.8864 = 0.1136 → **11.36% annual prepayment rate**
- Reverse check: SMM = 1 - (1 - 0.1136)^(1/12) = 1 - 0.8864^(1/12) = 1 - 0.99 = **0.01** ✓

### CDR → MDR (Default)

**CDR (Cumulative Default Rate)** — fraction of original UPB that has defaulted:

```
CDR = sum(funded_amnt - total_rec_prncp for Charged Off loans) / sum(funded_amnt for ALL loans in cohort)
```

**MDR (Monthly Default Rate)** — derived from annual CDR for use in projections:

```
MDR = 1 - (1 - CDR)^(1/12)
```

**Worked example:**
- If CDR = 0.10 (10% annual default rate)
- MDR = 1 - (1 - 0.10)^(1/12) = 1 - 0.90^(1/12) = 1 - 0.99123 = **0.00877** (0.877% monthly)
- Sanity check: 1 - (1 - 0.00877)^12 ≈ 0.10 ✓

---

## Loss Metrics

### Loss Severity & Recovery Rate

These are computed from **Charged Off loans only**, with recovery amounts capped at exposure.

**Step-by-step calculation:**

```python
# 1. Identify Charged Off loans
charged_off = df[df['loan_status'] == 'Charged Off']

# 2. Calculate exposure (what was still owed at charge-off)
exposure = charged_off['funded_amnt'] - charged_off['total_rec_prncp']

# 3. Keep only positive exposure loans
mask = exposure > 0
exposure = exposure[mask]
recoveries = charged_off.loc[mask, 'recoveries']

# 4. CRITICAL: Cap recoveries at exposure (can't recover more than was lost)
capped_recoveries = np.minimum(recoveries, exposure)

# 5. Calculate net loss after capped recoveries
capped_upb_lost = (exposure - capped_recoveries).clip(lower=0)

# 6. Aggregate
loss_severity = capped_upb_lost.sum() / exposure.sum()
recovery_rate = capped_recoveries.sum() / exposure.sum()

# 7. Verify: loss_severity + recovery_rate == 1.0 (within floating point)
assert abs(loss_severity + recovery_rate - 1.0) < 1e-6
```

**Why cap recoveries?** A small number of loans (4 in the dataset) have `recoveries > exposure`, which would make recovery rate exceed 100% and break the loss_severity + recovery_rate = 1.0 identity.

**Worked example:**
- Loan: funded_amnt = $10,000, total_rec_prncp = $3,000, recoveries = $1,500
- Exposure = $10,000 - $3,000 = $7,000
- Capped recoveries = min($1,500, $7,000) = $1,500
- UPB lost = $7,000 - $1,500 = $5,500
- Loss severity = $5,500 / $7,000 = **78.57%**
- Recovery rate = $1,500 / $7,000 = **21.43%**
- Sum = 100.00% ✓

---

## Pool-Level Weighted Averages

All weighted averages for cash flow projections use **active loans**: Current with March 2019 last payment date plus all delinquent (In Grace + Late) regardless of last payment date.

### WAC (Weighted Average Coupon)

```
WAC = sum(int_rate × out_prncp) / sum(out_prncp)
```

- `int_rate` is stored as a decimal (e.g., 0.1078 for 10.78%)
- Weight is current outstanding principal (`out_prncp`), NOT original funded amount
- Result is a decimal; display as percentage by multiplying by 100

### WAM (Weighted Average Maturity)

```
WAM = round(sum(updated_remaining_term × out_prncp) / sum(out_prncp))
```

- `updated_remaining_term` is computed by `calc_amort()` — it's the original term minus payments made
- Result is rounded to nearest integer (months)
- This determines how many months the cash flow projection runs

### WALA (Weighted Average Loan Age)

```
WALA = sum(orig_exp_payments_made × out_prncp) / sum(out_prncp)
```

- `orig_exp_payments_made` is computed by `calc_amort()` — months between issue date and snapshot

### Average FICO

```
avg_fico_original = sum(original_fico × funded_amnt) / sum(funded_amnt)
avg_fico_current  = sum(latest_fico × out_prncp) / sum(out_prncp)
```

- `original_fico` = average of `fico_range_high` and `fico_range_low`
- `latest_fico` = average of `last_fico_range_high` and `last_fico_range_low`

### Average DTI

```
avg_dti = sum(dti_clean × funded_amnt) / sum(funded_amnt)
```

- `dti_clean` uses `dti_joint` for joint applications, `dti` for individual

---

## Monthly Cash Flow Projection Loop

This is the core engine in `project_cashflows()`. It operates at **pool-level aggregate** — one balance, one WAC, one WAM.

### Inputs

| Parameter | Source | Example |
|-----------|--------|---------|
| `total_upb` | `compute_pool_characteristics()` → `sum(out_prncp)` | $50,000,000 |
| `wac` | `compute_pool_characteristics()` → weighted avg `int_rate` | 0.1269 (12.69%) |
| `wam` | `compute_pool_characteristics()` → weighted avg remaining term | 32 months |
| `monthly_payment` | `compute_pool_characteristics()` → `sum(installment)` | $1,800,000 |
| `cdr` | User input or `compute_pool_assumptions()` | 0.10 (10%) |
| `cpr` | User input or `compute_pool_assumptions()` | 0.12 (12%) |
| `loss_severity` | `compute_pool_assumptions()` | 0.88 (88%) |
| `purchase_price` | User input | 0.95 (95 cents on dollar) |

### Pre-loop Setup

```python
MDR = 1 - (1 - cdr) ** (1/12)
SMM = 1 - (1 - cpr) ** (1/12)
monthly_rate = wac / 12
```

### Monthly Loop (t = 1 to WAM, or until balance ≤ 0)

```
For each month t:
    beginning_balance = previous ending_balance  (or total_upb for t=1)

    # 1. Defaults happen first
    defaults = beginning_balance × MDR
    loss = defaults × loss_severity
    recovery = defaults × (1 - loss_severity)

    # 2. Performing balance after removing defaults
    performing_balance = beginning_balance - defaults

    # 3. Interest on performing balance
    interest = performing_balance × monthly_rate

    # 4. Scheduled principal (from regular amortization payment)
    scheduled_principal = monthly_payment - interest
    scheduled_principal = min(scheduled_principal, performing_balance)  # can't exceed balance

    # 5. Prepayments on remaining balance after scheduled principal
    prepayments = (performing_balance - scheduled_principal) × SMM

    # 6. Total principal and ending balance
    total_principal = scheduled_principal + prepayments
    ending_balance = performing_balance - total_principal
    ending_balance = max(ending_balance, 0)  # floor at zero

    # 7. Total cash flow to investor
    total_cashflow = interest + total_principal + recovery
```

### Edge Cases

- **Balance goes to zero early**: Stop the loop. Remaining months have zero cash flows.
- **Scheduled principal exceeds performing balance**: Cap at performing balance. This happens in the final months when the remaining balance is small.
- **Final month**: If ending balance is tiny (< $1), set to zero and include in total principal.
- **CDR = 0**: MDR = 0, no defaults, no losses, no recoveries. Pure amortization + prepayments.
- **CDR = 1**: All principal defaults immediately. Only recovery cash flows remain.
- **CPR = 0**: SMM = 0, no prepayments. Pure amortization + defaults.
- **CPR = 1**: All performing balance prepays immediately. One large cash flow.

### Output DataFrame Columns

`month`, `date`, `beginning_balance`, `defaults`, `loss`, `recovery`, `interest`, `scheduled_principal`, `prepayments`, `total_principal`, `ending_balance`, `total_cashflow`

---

## IRR Calculation

### `calculate_irr(cashflows_df, pool_chars, purchase_price)`

```python
import numpy_financial as npf

# Build cash flow array
cf = np.zeros(len(cashflows_df) + 1)
cf[0] = -(purchase_price * pool_chars['total_upb'])  # initial outlay
cf[1:] = cashflows_df['total_cashflow'].values        # monthly inflows

# Compute monthly IRR
monthly_irr = npf.irr(cf)

# Annualize
annual_irr = (1 + monthly_irr) ** 12 - 1

# Verify: NPV at this rate should ≈ 0
npv_check = npf.npv(monthly_irr, cf)
assert abs(npv_check) < 1.0, f"IRR verification failed: NPV = {npv_check}"
```

**Worked example (simple case):**
- Cash flows: [-100, 60, 60]
- Monthly IRR = 0.01031 (via npf.irr)
- NPV at that rate: -100 + 60/(1.01031) + 60/(1.01031)^2 ≈ 0 ✓
- Annual IRR = (1.01031)^12 - 1 = **13.07%**

**Another test case:**
- Cash flows: [-100, 110] (one-period, 10% return)
- Monthly IRR = 0.10
- Annual IRR = (1.10)^12 - 1 = **213.8%** (because it's a one-month investment)

---

## Price Solver

### `solve_price(pool_chars, target_irr, cdr, cpr, loss_severity)`

Use `scipy.optimize.brentq` to find the purchase price that yields the target IRR.

```python
from scipy.optimize import brentq

def objective(price):
    cf_df = project_cashflows(pool_chars, cdr, cpr, loss_severity, price)
    irr = calculate_irr(cf_df, pool_chars, price)
    return irr - target_irr

# Search between 0.01 and 2.00 (1% to 200% of UPB)
solved_price = brentq(objective, 0.01, 2.00, xtol=1e-6)
```

**Round-trip verification:** After solving, always verify:
```python
verify_cf = project_cashflows(pool_chars, cdr, cpr, loss_severity, solved_price)
verify_irr = calculate_irr(verify_cf, pool_chars, solved_price)
assert abs(verify_irr - target_irr) < 0.0001  # within 1bp
```

**Intuition:** Higher price → lower IRR (you pay more, earn less). Lower price → higher IRR. If the objective function doesn't cross zero in [0.01, 2.00], the target IRR is unachievable.

---

## Scenario Analysis

### Multiplicative Shifts

Scenarios are built by applying multipliers to the **base case CDR and CPR**. Loss severity stays FIXED across all scenarios.

| Scenario | CDR Multiplier | CPR Multiplier |
|----------|---------------|----------------|
| Base     | 1.0           | 1.0            |
| Stress   | 1.5 (default) | 0.75 (default) |
| Upside   | 0.5 (default) | 1.25 (default) |

**Worked example:**
- Base CDR = 8%, Base CPR = 12%, Loss Severity = 85%
- Stress: CDR = 8% × 1.5 = **12%**, CPR = 12% × 0.75 = **9%**, Loss Severity = **85%** (unchanged)
- Upside: CDR = 8% × 0.5 = **4%**, CPR = 12% × 1.25 = **15%**, Loss Severity = **85%** (unchanged)

**Why multiplicative, not additive?** Multiplicative shifts maintain proportional relationships across cohorts. A cohort with 5% base CDR gets a 7.5% stress CDR, while a cohort with 15% base CDR gets 22.5%. Additive would shift both by the same absolute amount, which doesn't make economic sense.

**User-adjustable multipliers:** The Streamlit dashboard lets users change the stress/upside multipliers via sliders. The defaults above are starting points.

### Base Case Derivation

Base case CDR and CPR come from `compute_pool_assumptions()`, which uses **historical data for the filtered cohort**:

- **CDR**: From ALL loans in the strata (need Charged Off loans to calculate default rate)
- **CPR**: From active loans — Current with March 2019 last payment date plus all delinquent (In Grace + Late) regardless of last payment date
- **Loss Severity**: From Charged Off loans in the strata (with capped recoveries)

These are cohort-specific, not global averages. If the user filters to "Grade B, 36-month, 2016Q4", the base case reflects that specific cohort's historical performance.

---

## Standard Amortization Reference

For verifying amortization calculations in `calc_amort()`:

### Monthly Payment

```
PMT = PV × r / (1 - (1 + r)^(-n))
```
Where: PV = funded_amnt, r = int_rate/12, n = term_months

**Test case:** $10,000 at 10% for 36 months → r = 0.10/12 = 0.008333
PMT = 10000 × 0.008333 / (1 - 1.008333^(-36)) = **$322.67**

### Remaining Balance After k Payments

```
Balance_k = PV × [(1+r)^n - (1+r)^k] / [(1+r)^n - 1]
```

**Test case:** $10,000 at 10% for 36 months, after 12 payments:
Balance_12 = 10000 × [1.008333^36 - 1.008333^12] / [1.008333^36 - 1] = **$6,915.72**

---

## Common Mistakes to Avoid

1. **Using CDR directly as monthly rate** — Always convert: MDR = 1 - (1 - CDR)^(1/12). Using CDR/12 is wrong.
2. **Forgetting to cap recoveries** — Without the cap, 4 loans break the loss_severity + recovery_rate = 1.0 identity.
3. **Interest on full balance instead of performing balance** — Interest should be on (beginning_balance - defaults), not beginning_balance.
4. **Not flooring ending balance at zero** — Floating point can produce tiny negative balances.
5. **Using funded_amnt instead of out_prncp for current pool metrics** — funded_amnt is the original amount; out_prncp is what's still outstanding.
6. **Confusing original vs current WAC** — Original WAC weights by funded_amnt; current WAC weights by out_prncp. Cash flow projections use current WAC.
7. **Annualizing IRR as monthly_irr × 12** — Wrong. Use (1 + monthly_irr)^12 - 1 for compounding.
8. **Applying CPR to cohort-level SMM** — The CPR provided to `compute_pool_assumptions()` should come from active loans (Current + In Grace + Late) with March 2019 last payment only, not all loans.
