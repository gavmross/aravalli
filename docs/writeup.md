# Lending Club Portfolio Investment Analysis — Writeup

## 1. Data Issues and How I Handled Them

The raw Lending Club dataset contains 2,260,701 rows and 151 columns — a March 2019 snapshot of every loan originated between 2007 and 2018. I dropped 5,207 rows (0.23%) across five cleaning steps. The most consequential decisions were not the row drops but the reclassifications and column engineering.

**Status mismatches were the biggest issue.** 
 - 4,537 loans were labeled "Current" but had zero outstanding principal. Another 168 loans in delinquent statuses (Grace Period, Late 16-30, Late 31-120) also had zero balances. I reclassified all of these as Fully Paid. Leaving them in their original status would have inflated delinquency rates and corrupted the transition matrix. 
 - 6 "Current" loans had last payment dates well before February 2019, which is inconsistent with active status on a March 2019 snapshot. I dropped these rather than guess their true status.

**Late fees required careful handling** to build the transition matrix. 
 - Lending Club only charges late fees once a borrower reaches Late (16-30 days), which means the presence of a late fee on a now-Current or Fully Paid loan is evidence that it was delinquent at some point and then cured. I used this signal (after cleaning out sub-$15 fees that appeared to be rounding artifacts or partial reversals) to flag 42,433 loans as historically delinquent. 
 - This `curr_paid_late1_flag` is the key input for estimating cure rates from the Late (16-30) bucket — without it, the transition matrix would have no way to distinguish loans that were always current from those that recovered from delinquency.

**I also dropped 2,737 "Does not meet the credit policy" loans** (legacy originations under a discontinued underwriting standard). Including them would skew metrics for evaluating the portfolio as it exists today.

 - Other cleaning was more routine: parsing date strings, converting interest rates from percentage to decimal form, creating averaged FICO scores from the reported ranges, handling joint application DTI, and computing a `maturity_month` for remaining-term calculations. 
 - The final database is 2,255,494 loans across ~46 columns (29 raw, 17 engineered).

One structural limitation of the data is that it is a single point-in-time snapshot — there are no monthly servicing tapes showing how each loan's status evolved over time. I worked around this by backsolveing approximate monthly statuses from the snapshot fields: Lending Club's delinquency progression is deterministic (Current → Grace → Late 16-30 → Late 31-120 → Charged Off), so given a loan's current status and last payment date, I can reconstruct when it entered each stage and further drill down how many months delinquent a loan is (rather than just knowing it is 31-120 days late). This backsolve is the foundation for the loan age-specific transition probabilities and the default timing analysis.

---

## 2. Metric and Cash Flow Definitions

**CDR (Conditional Default Rate)** 
 - Because I was able to reconstruct the timeline of when a loan progressed through delinqunet stages, I was able to use a trailing 12-month average. 
 - For each month from April 2018 through March 2019, I compute the Monthly Default Rate (MDR) as the UPB of newly defaulted loans divided by the estimated performing balance at the start of that month.
 - For each performing loan, I calculate its scheduled balance at that historical date, then adjust for observed prepayment behavior. Using the unadjusted amortization schedule would overstate balances for loans that have been prepaying → understate MDR → understate CDR. For loans still active at the snapshot, I calibrate from the observed endpoint:
   - `sched_balance_at_snapshot = calc_balance(funded_amnt, int_rate, payment, age_at_snapshot)`
   - `cumulative_prepaid = max(sched_balance_at_snapshot − out_prncp, 0)`
   - `monthly_prepaid = cumulative_prepaid / age_at_snapshot`
   - Then for each historical month M: `adj_balance = max(sched_balance − monthly_prepaid × age_at_M, 0)`
   - For Fully Paid / Charged Off loans, no actual balance is observable at the snapshot, so I use the unadjusted amortization schedule.
   - This adjustment is valid for the CDR denominator because the numerator (default exposure) is directly observed — one degree of freedom. It is NOT applied to CPR, where both numerator (monthly prepayment amount) and denominator (balance) would need to be estimated from the same uniform-spread assumption — two degrees of freedom, errors compound. The single-month CPR uses actual observed values on both sides (Current + Fully Paid March 2019 loans).
 - I average the 12 MDRs and annualize: `CDR = 1 − (1 − avg_MDR)^12`. This is the standard conditional approach used in structured finance (sometimes called the "dv01" methodology). It captures the portfolio's recent default behavior rather than its lifetime cumulative experience, which is more relevant for forward-looking projections. I separately display the cumulative default rate as a reference metric but do not use it in cash flow projections.

**CPR (Conditional Prepayment Rate)** is derived from pool-level prepayment behavior in the most recent payment period. I compute the aggregate SMM as total unscheduled principal divided by (total beginning balance minus total scheduled principal) across all active loans, then annualize: `CPR = 1 − (1 − SMM)^12`. This pool-level aggregation properly weights by balance — averaging individual loan SMMs would not.

**Loss severity** is computed from Charged Off loans: `(exposure − capped_recoveries) / exposure`, where exposure is `funded_amnt − total_rec_prncp` and recoveries are capped at exposure to prevent a handful of anomalous loans from producing recovery rates above 100%.

**The cash flow engine uses a 7-state transition model.** Rather than applying flat CDR/CPR rates each month, I track the pool across seven states — Current, Delinquent (0-30), Late_1, Late_2, Late_3, Charged Off, and Fully Paid — with empirically derived, age-specific transition probabilities at monthly granularity. Defaults flow through a realistic 5-month pipeline (Current → Delinquent → Late_1 → Late_2 → Late_3 → Charged Off), meaning for a fully current pool, the first defaults don't appear until month 5. This better reflects how consumer loan delinquencies actually progress than a model where defaults hit immediately each month.

One important design decision: the historical `Current → Fully Paid` transition rate includes both voluntary prepayments and loans reaching scheduled maturity, which would overstate forward-looking prepayment speed for a pool of mid-term loans. I replace the empirical rate with a constant SMM derived from the observed pool-level CPR, adding the difference back to the `Current → Current` rate to keep row probabilities summing to 1.0. This means the transition model drives the timing and magnitude of defaults through the delinquency pipeline, while prepayments are governed by the pool-level CPR — a hybrid approach that leverages the strengths of both methods.

**IRR** is computed as the annualized rate: the initial outlay is `purchase_price × total_UPB` (negative), followed by monthly inflows of interest, scheduled principal, prepayments, and recoveries. I use `numpy_financial.irr` for the monthly rate and annualize via `(1 + monthly_irr)^12 − 1`. The price solver inverts this — given a target IRR, it uses Brent's method to find the purchase price that achieves it.

---

## 3. Key Insights About the Portfolio

**The portfolio is heavily concentrated in mid-grade credit.** Grades B and C together account for 1.31 million loans (58% of the pool), while Grades A through C represent over 77%. This is not a deep subprime portfolio, but it is not prime either — it sits squarely in the near-prime space, which means default behavior is highly sensitive to economic conditions.

**Vintage matters enormously.** Older vintages (2007–2014) have largely run off — most loans are either Fully Paid or Charged Off, providing clean historical performance data. Recent vintages (2017–2018) dominate the active pool and have less performance history, which introduces uncertainty into forward projections. The 2018 vintages in particular are still early in their life cycle, so their observed default rates understate what they will ultimately experience.

**Loss severity is high.** Across the pool, loss severity on Charged Off loans runs around 75-85%, meaning recoveries are only 15-25 cents on the dollar of exposure. This is characteristic of unsecured consumer credit — there is no collateral to liquidate. It also means that the CDR assumption has outsized impact on returns: even a modest increase in defaults produces large losses because almost nothing is recovered.

**The delinquency pipeline is relatively fast-moving.** The transition matrix shows that once a loan enters Grace Period, the probability of progressing to Late (16-30) is substantial, and cure rates from Late (16-30) back to Current are modest (the `curr_paid_late1_flag` analysis shows only ~1.9% of Current/Fully Paid loans ever experienced a late fee). Once a loan reaches Late (31-120), it very rarely cures — the vast majority charge off. This deterministic pipeline is what makes the state-transition model work well: there are limited paths and the probabilities are empirically stable.

**The pool as of March 2019 is large and primarily performing.** 821,602 loans are Current (36% of the total), with another ~86,000 in various stages of delinquency (including 52,171 reclassified from Current to In Grace Period because their last payment was February 2019, meaning they missed the March 2019 cycle). Total active UPB exceeds $9.5 billion. The investor is buying a seasoned, predominantly performing book — but with meaningful tail risk from the delinquent pipeline and the high loss severity on any loans that do default.

---

## 4. Scenario Recommendations for Stress Testing

Our scenario framework uses multiplicative shifts to transition probabilities rather than simple CDR/CPR adjustments. This is more realistic because it stresses the underlying mechanism — the probability of a Current loan becoming delinquent, the probability of a delinquent loan curing — rather than applying a blunt top-level haircut. The default stress/upside magnitude is ±15%, adjustable from 5% to 50%.

**Stress scenario logic:** I increase `Current → Delinquent` transition probabilities by the stress factor (more loans miss payments), decrease all cure rates by the same factor (fewer recoveries from delinquency), and decrease `Current → Fully Paid` rates (less refinancing activity in a tight credit environment). I do not directly stress the `Late_3 → Charged Off` probability — it increases mechanically through re-normalization as cure rates decline, which is more realistic than an arbitrary bump. Loss severity is held constant across scenarios because recovery rates on unsecured consumer loans are structurally low and don't vary much with the economic cycle.

**Why multiplicative rather than additive shifts?** A cohort with a 5% base CDR gets a 5.75% stress CDR at +15%, while a 15% base CDR cohort gets 17.25%. Additive shifts would move both by the same absolute amount (e.g., +2%), which doesn't reflect the reality that higher-risk pools are more sensitive to economic deterioration. The multiplicative approach preserves proportional relationships across strata.

**Recommended stress magnitudes and what they represent:**

A 15% stress (the default) is a moderate deterioration — roughly the kind of credit tightening you might see in a mild recession. It is a reasonable base stress for evaluating whether the investment has adequate margin of safety at a given purchase price.

A 25-30% stress represents a more severe downturn, comparable to meaningful labor market weakness. At these levels, the investor should expect IRR compression of several hundred basis points and should evaluate whether the return still exceeds their cost of capital.

A 40-50% stress is a severe scenario — approaching 2008-level disruption for consumer credit. This is the "how bad can it get" test. If the investment still produces a positive return (or the loss is contained) at this level, the downside risk is well understood.

**Why the upside scenario matters too:** It is not just about quantifying risk. The upside scenario (lower delinquency inflows, faster prepayments, higher cure rates) establishes the realistic ceiling on returns. If the spread between stress and upside IRR is narrow, the investment is insensitive to credit conditions and the base case is dependable. If the spread is wide, the investor is making a directional bet on credit performance and should price accordingly.

The scenario framework is designed to give an investor three things: a base case grounded in historical performance, a stress case that quantifies downside, and an upside case that bounds the opportunity. Together they define the range of outcomes and let the investor decide whether the risk-return profile at a given purchase price is acceptable.
