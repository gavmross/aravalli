# Lending Club Portfolio Investment Analysis — Writeup

## 1. Data Issues and How We Handled Them

**Source**: 2,260,701 loans, 151 columns, March 2019 snapshot. Final database: 2,255,494 loans (~46 columns) after dropping 5,207 rows (0.23%).

**Status misclassifications** — the most impactful issue:

- **4,537 "Current" loans with zero balance** → reclassified as Fully Paid. Leaving them inflates the active pool and distorts cash flow projections.
- **168 delinquent loans (Grace / Late 16-30 / Late 31-120) with zero balance** → reclassified as Fully Paid. Zero principal means the loan is done.
- **6 "Current" loans with stale payment dates** (last payment well before Feb 2019) → dropped. Inconsistent with active status on a March 2019 snapshot.

**Late fee signal for the transition matrix:**

- Lending Club charges late fees only at Late (16-30 days)
- Late fee on a now-Current or Fully Paid loan = evidence the borrower was delinquent and cured
- Cleaned out sub-$15 fees (rounding artifacts), then flagged 42,433 loans (`curr_paid_late1_flag`)
- Because I don't have a reliable way to estimate how delinquent these loans went, I just assumed they were late 1 (16-30 days)
- This flag is the sole input for estimating cure rates from Late (16-30)

**Other cleaning:**

- Dropped 2,737 "Does not meet the credit policy" loans (legacy underwriting, non-representative)
- Parsed date strings, converted `int_rate` from percentage to decimal, averaged FICO ranges to midpoints
- Created `dti_clean` using joint DTI for joint applications, individual DTI otherwise
- Computed `maturity_month = issue_d + term_months`

**No monthly tapes** — single point-in-time snapshot. We reconstruct approximate monthly statuses via backsolve, leveraging Lending Club's deterministic delinquency progression:

```
Current → Grace → Late (16-30) → Late (31-120) → Charged Off
```

This produces `default_month` and `payoff_month` per loan — the foundation for age-specific transition probabilities and the conditional default rate calculation.

---

## 2. Metric and Cash Flow Definitions

### Default Rates

**MDR (Monthly Default Rate)** — for each trailing month M (Apr 2018 – Mar 2019):

```
default_UPB = Σ (funded_amnt − total_rec_prncp)    [loans defaulting in month M, clipped ≥ 0]
```

Performing balance is reconstructed (no monthly tapes). A loan is performing at the start of month M if:

- `issue_d ≤ month_start` (originated)
- `default_month` is null or `> month_start` (not yet defaulted)
- `payoff_month` is null or `> month_start` (not yet paid off)

Each performing loan's balance is estimated via amortization:

```
age = months between issue_d and month_start
payment = PV × r / (1 − (1 + r)^(−n))
estimated_balance = calc_balance(funded_amnt, int_rate, payment, age)
performing_balance = Σ estimated_balance
```

```
MDR_M = default_UPB / performing_balance
```

**CDR (Conditional Default Rate)** — annualized from trailing 12-month average:

```
avg_MDR = mean(MDR_1, MDR_2, ..., MDR_12)
CDR = 1 − (1 − avg_MDR)^12
```

Cash flow engine converts back: `MDR = 1 − (1 − CDR)^(1/12)`.

Cumulative default rate (`Σ defaulted_UPB / Σ funded_amnt`) is displayed for reference only — NOT used in projections, NOT called CDR.

### Prepayment Rates

**SMM (Single Monthly Mortality)** — derived from last payment period. Per-loan components:

```
beginning_balance = (out_prncp + last_pymnt_amnt) / (1 + monthly_rate)
interest = beginning_balance × monthly_rate
actual_principal = last_pymnt_amnt − interest
scheduled_principal = installment − interest
unscheduled_principal = actual_principal − scheduled_principal
```

Pool-level aggregation:

```
SMM = Σ unscheduled_principal / (Σ beginning_balance − Σ scheduled_principal)
```

**CPR (Conditional Prepayment Rate)** — annualized:

```
CPR = 1 − (1 − SMM)^12
```

Cash flow engine converts back: `SMM = 1 − (1 − CPR)^(1/12)`.

### Loss Severity

From Charged Off loans with positive exposure:

```
exposure = funded_amnt − total_rec_prncp
capped_recoveries = min(recoveries, exposure)          [4 loans exceed — cap at 100%]
loss_severity = Σ (exposure − capped_recoveries) / Σ exposure
recovery_rate = 1 − loss_severity
```

### Pool Characteristics

From all active loans (Current + In Grace + Late) with March 2019 last payment:

```
WAC = Σ (int_rate × out_prncp) / Σ out_prncp
WAM = round(Σ (remaining_term × out_prncp) / Σ out_prncp)
monthly_payment = Σ installment
```

### Cash Flow Engine — 7-State Transition Model

Pool tracked across 7 states with age-specific empirical transition probabilities:

| State | Type |
|-------|------|
| Current | Transient |
| Delinquent (0-30) | Transient |
| Late_1, Late_2, Late_3 | Transient |
| Charged Off | Absorbing |
| Fully Paid | Absorbing |

Defaults flow through a 5-month pipeline:

```
Current → Delinquent (0-30) → Late_1 → Late_2 → Late_3 → Charged Off
```

For an all-Current pool, first defaults appear at **month 5**.

**Prepayment override**: empirical `Current → Fully Paid` includes maturity payoffs (overstates forward prepayment). We replace it with constant SMM from pool CPR, adjust `Current → Current` to keep row sums = 1.0. Result: transition model drives defaults, CPR drives prepayments.

### IRR and Price Solver

```
cash_flows[0] = −(purchase_price × total_UPB)
cash_flows[1..N] = interest + sched_principal + prepayments + recoveries
monthly_irr = npf.irr(cash_flows)
annual_irr = (1 + monthly_irr)^12 − 1
```

Price solver: Brent's method finds purchase price where `IRR(price) = target_IRR`.

---

## 3. Key Insights About the Portfolio

**Credit concentration is mid-grade:**

- Grades B + C = 1.31M loans (58%)
- Grades A–C = 77%+
- Near-prime — default behavior highly sensitive to economic conditions

**Vintage drives everything:**

- Older vintages (2007–2014) largely run off → clean historical data
- 2017–2018 dominate the active pool with less seasoning → observed defaults understate ultimate losses

**Loss severity is structurally high:**

- ~75–85% loss severity on Charged Off loans (15–25 cents recovery per dollar)
- Unsecured consumer credit — no collateral
- CDR assumptions have outsized impact on returns

**Delinquency pipeline is fast and one-directional:**

- Only ~1.9% of Current/Fully Paid loans show evidence of ever being delinquent
- Past Late (16-30), cure rates are very low — vast majority charge off
- Late (31-120) is effectively a terminal path

**Pool at March 2019:**

- 821,602 Current loans (36%), ~86,000 in delinquency stages (including 52,171 reclassified from Current to In Grace Period)
- Active UPB > $9.5B
- Predominantly performing, but meaningful tail risk from delinquent pipeline + high loss severity

---

## 4. Scenario Recommendations for Stress Testing

### Framework

We stress **transition probabilities**, not flat CDR/CPR. This stresses the underlying mechanism rather than applying a top-level haircut.

**Stress shifts** (multiplicative, default ±15%, adjustable 5–50%):

- `Current → Delinquent` × `(1 + stress_pct)` — more missed payments
- `Current → Fully Paid` × `(1 − stress_pct)` — less refinancing
- All cure rates (non-Current → Current) × `(1 − stress_pct)` — harder to recover
- `Late_3 → Charged Off` NOT directly stressed — increases mechanically via re-normalization
- **Loss severity held constant** — structurally low recovery doesn't vary much with the cycle

**Upside**: opposite multipliers.

### Why Multiplicative, Not Additive

- 5% base CDR × 1.15 → 5.75% stress
- 15% base CDR × 1.15 → 17.25% stress
- Preserves proportional relationships — higher-risk pools are more sensitive to deterioration

### Recommended Stress Magnitudes

| Magnitude | Represents | Question It Answers |
|-----------|-----------|---------------------|
| ±15% | Mild recession | Does the investment have margin of safety? |
| ±25–30% | Labor market weakness | Does IRR still exceed cost of capital? |
| ±40–50% | Severe (2008-level) | Is the downside bounded? |

### Why the Upside Matters

- Establishes the **ceiling** on returns
- Narrow stress-to-upside IRR spread → insensitive to credit, base case is reliable
- Wide spread → directional bet on credit, price accordingly