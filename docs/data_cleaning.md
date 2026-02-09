# Data Cleaning Documentation

This document describes every cleaning step applied by `scripts/export_to_sqlite.py` to transform the raw Lending Club CSV into the analysis-ready `data/loans.db` SQLite database.

---

## Source Data

- **File**: `data/accepted_2007_to_2018Q4.csv`
- **Raw rows**: 2,260,701
- **Raw columns**: 151
- **Snapshot date**: March 2019

---

## Pipeline Summary

| Metric | Value |
|--------|-------|
| Initial rows | 2,260,701 |
| Final rows | 2,255,494 |
| Total rows dropped | 5,207 (0.23%) |
| Initial columns | 151 |
| Final columns | ~46 |

---

## Cleaning Steps

### Step 1 — Drop non-loan rows

**Action**: Drop rows where `id` is non-numeric.

**Rationale**: The Lending Club CSV contains trailing summary/policy rows that are not actual loan records. These rows have non-numeric values in the `id` column.

**Impact**: Dropped **33** rows. Remaining: 2,260,668.

---

### Step 2 — Keep only used columns (whitelist)

**Action**: Keep only the ~29 columns required by the analytics pipeline; drop everything else.

**Rationale**: The raw CSV has 151 columns, but the dashboard, analytics functions, and cash flow engine only reference ~29 of them. Keeping all columns inflated the SQLite database to over 1 GB. A whitelist approach keeps only what is needed, reducing the database to ~200 MB.

**Columns kept (29)**:
`id`, `loan_status`, `funded_amnt`, `int_rate`, `term`, `installment`, `grade`, `sub_grade`, `purpose`, `addr_state`, `home_ownership`, `issue_d`, `last_pymnt_d`, `next_pymnt_d`, `last_credit_pull_d`, `out_prncp`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`, `last_pymnt_amnt`, `recoveries`, `collection_recovery_fee`, `fico_range_high`, `fico_range_low`, `last_fico_range_high`, `last_fico_range_low`, `dti`, `dti_joint`, `annual_inc`, `annual_inc_joint`, `application_type`

**Impact**: 151 columns → 31 columns (29 raw + `term` kept temporarily). No rows dropped. Additional engineered columns are added in later steps, bringing the final count to ~46.

---

### Step 3 — Convert interest rate to decimal

**Action**: Convert `int_rate` from percentage form (e.g., 10.78) to decimal form (e.g., 0.1078) by dividing by 100.

**Rationale**: All downstream formulas (amortization, WAC, cash flow projections) expect rates in decimal form.

**Validation**: After conversion, `int_rate` range is 0.0531 – 0.3099 (5.31% – 30.99%).

**Impact**: No rows dropped.

---

### Step 4 — Parse date columns

**Action**: Parse `issue_d`, `last_pymnt_d`, `next_pymnt_d`, and `last_credit_pull_d` from string (e.g., "Dec-2016") to datetime.

**Rationale**: Date arithmetic (maturity calculation, payment period counting, filtering by date) requires proper datetime types.

**Non-null counts after parsing**:
- `issue_d`: 2,260,668
- `last_pymnt_d`: 2,258,241
- `next_pymnt_d`: 915,358
- `last_credit_pull_d`: 2,260,596

**Impact**: No rows dropped.

---

### Step 5 — Extract term_months

**Action**: Extract integer term from the `term` string column (e.g., " 36 months" → 36).

**Rationale**: Numeric term is required for amortization calculations and maturity computation.

**Validation**: Only two values exist: **36** and **60**.

**Impact**: No rows dropped.

---

### Step 6 — Create maturity_month

**Action**: Compute `maturity_month = issue_d + term_months` (as a monthly period).

**Rationale**: Needed for remaining-term calculations and to identify loans nearing maturity.

**Impact**: No rows dropped.

---

### Step 7 — Drop rows with missing last_pymnt_d

**Action**: Drop rows where `last_pymnt_d` is null.

**Rationale**: The last payment date is essential for amortization calculations (`calc_amort` uses it as the as-of date), CPR computation, and identifying the cash flow population. Loans without this field cannot be analyzed. Inspection showed most of these also lack `next_pymnt_d`, suggesting they are incomplete records.

**Impact**: Dropped **2,427** rows. Remaining: 2,258,241.

---

### Step 8 — Drop "Does not meet the credit policy" loans

**Action**: Drop loans with `loan_status` starting with "Does not meet the credit policy".

**Rationale**: These are legacy loans originated under a different credit policy that Lending Club no longer uses. They represent a non-representative population (< 0.3% of loans) and would skew metrics for a portfolio being evaluated for purchase as of 2019.

**Impact**: Dropped **2,737** rows (0.12%). Remaining: 2,255,504.

---

### Step 9 — Reclassify zero-balance Current loans as Fully Paid

**Action**: Change `loan_status` from "Current" to "Fully Paid" where `out_prncp == 0`.

**Rationale**: A loan with zero outstanding principal has been fully repaid regardless of its recorded status. Investigation showed these loans had recent last payment dates (Feb/March 2019) and their last payments were large enough to pay off the balance. Keeping them as "Current" would inflate the active portfolio and distort cash flow projections.

**Impact**: **4,537** loans reclassified. No rows dropped.

---

### Step 10 — Drop Current loans with stale payment dates

**Action**: Drop Current loans where `last_pymnt_d` is not February 2019 or March 2019.

**Rationale**: The data snapshot is as of March 2019. Current loans should have recent payment activity. Loans with last payments before February 2019 are anomalous — they may be in an unrecorded delinquency state. The population is negligible (6 loans).

**Impact**: Dropped **6** rows. Remaining: 2,255,498.

---

### Step 11 — Set negative late fees to zero

**Action**: Set `total_rec_late_fee` to 0 where the value is negative.

**Rationale**: Negative late fees are data errors — late fees cannot be refunded below zero. All 8 affected values were very small negatives (essentially zero due to float rounding).

**Impact**: **8** values corrected. No rows dropped.

---

### Step 12 — Set small late fees to zero (global, all statuses)

**Action**: Round `total_rec_late_fee` to 2 decimal places, then set to 0 where the value is between $0 and $15 (exclusive).

**Rationale**: Lending Club's minimum late fee is the greater of 5% of the monthly payment or $15. Values below $15 are either data artifacts, rounding errors, or partial fee reversals. Setting them to zero prevents spurious late-fee flags.

**Impact**: **675** values zeroed. No rows dropped.

---

### Step 13 — Create current_late_fee_flag

**Action**: Create `current_late_fee_flag = 1` for Current loans with `total_rec_late_fee > 0` (after cleaning).

**Rationale**: Identifies Current loans that incurred late fees at some point during their life, indicating the borrower was delinquent at least once. Useful for credit quality segmentation within the performing pool.

**Impact**: **18,419** Current loans flagged.

---

### Step 14 — Reclassify zero-balance delinquent loans as Fully Paid

**Action**: For loans with `loan_status` in {In Grace Period, Late (16-30 days), Late (31-120 days)} and `out_prncp == 0`, reclassify as "Fully Paid". Create boolean flags to preserve the original status:
- `grace_to_paid_flag`
- `late1_to_paid_flag`
- `late2_to_paid_flag`

**Rationale**: These borrowers had zero outstanding principal and no recoveries, indicating the loan was fully repaid despite the delinquent status label. Many had large final payments exceeding their installment, consistent with a payoff. Keeping them as delinquent would inflate delinquency rates and distort the transition matrix.

**Counts**:
- In Grace Period → Fully Paid: **107**
- Late (16-30 days) → Fully Paid: **35**
- Late (31-120 days) → Fully Paid: **26**
- **Total reclassified: 168**

**Impact**: No rows dropped.

---

### Step 15 — Create upb_lost for Charged Off loans

**Action**: Create `upb_lost = -(funded_amnt - total_rec_prncp - recoveries)` for Charged Off loans; 0 for all others.

**Rationale**: Represents the net principal loss on defaulted loans. Stored as a negative value (loss convention). Used for loss severity calculations and portfolio-level loss metrics.

**Statistics**:
- Charged Off loans: **266,246**
- Average `upb_lost`: **-$9,928.42**

**Impact**: No rows dropped.

---

### Step 16 — Create joint application columns

**Action**: Create three engineered columns:
- `joint_app_flag`: 1 if `application_type != "Individual"`, else 0
- `dti_clean`: uses `dti_joint` for joint applications, `dti` for individual
- `annual_inc_clean`: uses `annual_inc_joint` for joint applications, `annual_inc` for individual

**Rationale**: Joint applications report household-level DTI and income, which are more relevant for credit analysis than the primary borrower's individual figures. The clean columns provide a single consistent metric regardless of application type.

**Statistics**: **120,581** loans flagged as joint applications (5.35%).

**Impact**: No rows dropped.

---

### Step 17 — Create FICO score columns

**Action**: Create averaged FICO columns:
- `original_fico = (fico_range_high + fico_range_low) / 2`
- `latest_fico = (last_fico_range_high + last_fico_range_low) / 2`

**Rationale**: Lending Club reports FICO as a range (e.g., 685–689). The midpoint is the standard convention for analysis.

**Validation**:
- `original_fico` range: 627 – 848
- `latest_fico` range: 0 – 848 (0 indicates missing FICO pull data)

**Impact**: No rows dropped.

---

### Step 18 — Clean DTI

**Action**: Set negative `dti_clean` values to 0, then drop rows where `dti_clean` is null.

**Rationale**: Negative DTI values are data errors (DTI cannot be negative). Null DTI values occur for a handful of joint applications where neither individual nor joint DTI was reported. Since DTI is used in weighted-average credit metrics, these rows cannot be included.

**Statistics**:
- Negative values set to 0: **1**
- Null rows dropped: **4**

**Impact**: Dropped **4** rows. Remaining: 2,255,494.

---

### Step 19 — Create date feature columns

**Action**: Create two columns from `issue_d`:
- `issue_quarter`: vintage quarter (e.g., "2016Q3"), using `pd.to_period("Q")`
- `issue_month_year`: formatted as "Dec-2016", using `strftime("%b-%Y")`

**Rationale**: `issue_quarter` is the primary vintage stratification axis used throughout the dashboard. `issue_month_year` provides a human-readable label for display purposes.

**Validation**: `issue_quarter` range is **2007Q2 – 2018Q4**.

**Impact**: No rows dropped.

---

### Step 20 — Create curr_paid_late1_flag

**Action**: Create `curr_paid_late1_flag = 1` for loans where `loan_status` is "Fully Paid" or "Current" AND `total_rec_late_fee > 0` (after all late-fee cleaning).

**Rationale**: This flag identifies loans that reached at least Late (16-30 days) status during their life (since that's when Lending Club charges late fees) but subsequently cured to Current or paid off. It is the key input to the delinquency transition matrix — it allows us to estimate the cure rate from the Late (16-30) bucket.

**Statistics**: **42,433** loans flagged (1.88%).

**Impact**: No rows dropped.

---

## Final Database Statistics

### Loan Status Distribution

| Status | Count | Percentage |
|--------|------:|----------:|
| Fully Paid | 1,081,453 | 47.95% |
| Current | 873,773 | 38.74% |
| Charged Off | 266,246 | 11.80% |
| Late (31-120 days) | 21,339 | 0.95% |
| In Grace Period | 8,329 | 0.37% |
| Late (16-30 days) | 4,314 | 0.19% |
| Default | 40 | 0.00% |

### Grade Distribution

| Grade | Count |
|-------|------:|
| A | 432,757 |
| B | 662,776 |
| C | 648,787 |
| D | 323,164 |
| E | 134,708 |
| F | 41,408 |
| G | 11,894 |

### Cash Flow Projection Population

The cash flow projection engine (Tabs 2 and 3 of the dashboard) operates on **Current loans with `last_pymnt_d = 2019-03-01`** only.

| Metric | Value |
|--------|------:|
| March 2019 Current loans | 821,602 |
| Total outstanding UPB | $8,576,562,742 |

### Engineered Columns Summary

| Column | Non-zero count | Description |
|--------|---------------:|-------------|
| `current_late_fee_flag` | 18,419 | Current loans with historical late fees |
| `curr_paid_late1_flag` | 42,433 | Current/Fully Paid loans that reached Late (16-30) |
| `grace_to_paid_flag` | 107 | In Grace Period reclassified to Fully Paid |
| `late1_to_paid_flag` | 35 | Late (16-30) reclassified to Fully Paid |
| `late2_to_paid_flag` | 26 | Late (31-120) reclassified to Fully Paid |
| `joint_app_flag` | 120,581 | Joint/Direct Pay applications |

### Row Drop Waterfall

| Step | Rows Dropped | Cumulative Dropped | Remaining |
|------|-------------:|-------------------:|----------:|
| Raw data | — | — | 2,260,701 |
| Step 1: Non-numeric IDs | 33 | 33 | 2,260,668 |
| Step 7: Null last_pymnt_d | 2,427 | 2,460 | 2,258,241 |
| Step 8: Credit policy loans | 2,737 | 5,197 | 2,255,504 |
| Step 10: Stale Current loans | 6 | 5,203 | 2,255,498 |
| Step 18: Null dti_clean | 4 | 5,207 | 2,255,494 |
| **Total** | **5,207** | | **2,255,494** |

---

## Notes

- **40 Default loans** remain in the database. These are loans in default but not yet charged off. They represent 0.002% of the population and are not used by any analytics function (which filter by specific statuses).
- **`int_rate`** is stored as a decimal (e.g., 0.1078 for 10.78%). All downstream code assumes this format.
- **`calc_amort()` columns** (e.g., `orig_exp_*`, `last_pmt_*`, `next_pmt_*`, `updated_remaining_term`) are NOT stored in the database. They are computed at runtime when the dashboard loads data.
- **Date columns** are stored as strings in `YYYY-MM-DD` format in SQLite.
