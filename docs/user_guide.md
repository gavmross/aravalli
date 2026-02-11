# User Guide

This guide covers setup, launching the dashboard, and a detailed walkthrough of every feature.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Generating the Database](#2-generating-the-database)
3. [Running Tests](#3-running-tests)
4. [Launching the Dashboard](#4-launching-the-dashboard)
5. [Dashboard Walkthrough](#5-dashboard-walkthrough)
   - [Sidebar Controls](#sidebar-controls)
   - [Tab 1: Portfolio Analytics](#tab-1-portfolio-analytics)
   - [Tab 2: Cash Flow Projection](#tab-2-cash-flow-projection)
   - [Tab 3: Scenario Analysis](#tab-3-scenario-analysis)
6. [Codebase Overview](#6-codebase-overview)
7. [Common Workflows](#7-common-workflows)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Environment Setup

### Prerequisites

- Python 3.11 or higher
- The raw Lending Club CSV file (`accepted_2007_to_2018Q4.csv`) placed in the `data/` directory

### Create Virtual Environment

```bash
# Create the virtual environment
python -m venv .env

# Activate it
# macOS/Linux:
source .env/bin/activate

# Windows (Command Prompt):
.env\Scripts\activate

# Windows (PowerShell):
.env\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: pandas, numpy, scipy, numpy-financial, streamlit, plotly, openpyxl, and pytest.

---

## 2. Generating the Database

The raw CSV must be cleaned and loaded into SQLite before the dashboard can run.

```bash
python scripts/export_to_sqlite.py
```

**What this does**:
- Reads `data/accepted_2007_to_2018Q4.csv` (~2.2M loans)
- Applies 21 cleaning steps (documented in [data_cleaning.md](data_cleaning.md))
- Writes cleaned data to `data/loans.db` (SQLite)
- Final count: 2,255,494 loans

**Expected runtime**: 2-5 minutes depending on hardware.

**Verify**: After running, `data/loans.db` should exist and be approximately 200 MB (the column whitelist reduces the original ~2-3 GB dataset significantly).

---

## 3. Running Tests

```bash
pytest tests/ -v
```

The test suite contains 209 tests across 4 modules:

| Test File | Tests | Coverage |
|-----------|------:|----------|
| `test_amortization.py` | 19 | Monthly payment, balance, payment count, full calc_amort |
| `test_portfolio_analytics.py` | 72 | Credit metrics, performance metrics, transition matrix, 7-state transitions, late sub-states |
| `test_cashflow_engine.py` | 77 | Pool assumptions, characteristics, cash flows, IRR, price solver, state-transition engine, implied CPR, curtailment rates, curtailments in projections |
| `test_scenario_analysis.py` | 54 | Base assumptions, scenario builder, scenario comparison, transition scenarios, vintage percentiles, percentile-based scenarios, scenario curtailments |

All tests use synthetic data fixtures and run independently of the database.

---

## 4. Launching the Dashboard

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

**First load**: The initial load takes several minutes because `calc_amort()` runs on the entire dataset (~2.2M loans). This computation is cached by Streamlit — subsequent page interactions are fast.

**Performance tips**:
- The first load is the slowest. Once cached, filter changes are near-instant.
- If you restart Streamlit, the cache persists for the same session directory.
- For faster iteration during development, filter to a specific strata (e.g., Grade B) to reduce the dataset size.

---

## 5. Dashboard Walkthrough

The dashboard is a single-page Streamlit app with a sidebar for filters and three content tabs.

### Sidebar Controls

The sidebar contains two controls that affect all three tabs:

#### Strata Type Dropdown

Selects the dimension for filtering the loan pool:

| Option | Column | Description |
|--------|--------|-------------|
| ALL | (none) | Use the entire dataset |
| Grade | `grade` | LC credit grade (A through G) |
| Term | `term_months` | Loan term (36 or 60 months) |
| Purpose | `purpose` | Loan purpose (debt consolidation, credit card, etc.) |
| State | `addr_state` | Borrower state (50 states + DC) |
| Vintage | `issue_quarter` | Origination quarter (2007Q2 through 2018Q4) |

#### Strata Value Dropdown

After selecting a Strata Type, this dropdown populates with all available values for that dimension. Select a specific value (e.g., "B" for Grade, "2018Q1" for Vintage) or "ALL" for the entire dimension.

When Strata Type is set to "ALL", this dropdown is disabled.

Purchase Price controls live on their respective tabs (Tab 2 and Tab 3). Tab 3 also has editable CDR/CPR scenario inputs.

#### Sidebar Summary

Below the controls, the sidebar displays:
- **Filtered Loans**: Count of loans matching the current filter
- **Total UPB**: Sum of outstanding principal for filtered loans

---

### Tab 1: Portfolio Analytics

This tab provides a comprehensive view of the filtered loan pool's composition and historical performance.

#### Pool Summary (Metric Cards)

Four metric cards at the top:

| Metric | Definition |
|--------|-----------|
| Total Loans | Count of loans in filtered pool |
| Total UPB | Sum of outstanding principal ($) |
| Avg FICO | Mean original FICO score |
| Avg DTI | Mean debt-to-income ratio (%) |

#### Credit Metrics Table

A detailed table stratified by the selected dimension (or by Grade if "ALL" is selected). Each row represents a strata value plus an "ALL" aggregate row.

Key columns:

| Column | Meaning |
|--------|---------|
| `orig_total_upb_mm` | Original total UPB in millions |
| `orig_loan_count` | Number of loans originated |
| `orig_wac` | Original weighted average coupon (interest rate) |
| `orig_wam` | Original weighted average maturity (months) |
| `orig_avg_fico` | Original average FICO score |
| `orig_avg_dti` | Original average DTI |
| `active_total_upb_mm` | Currently active UPB in millions |
| `active_upb_current_perc` | % of active UPB that is Current |
| `active_upb_grace_perc` | % of active UPB In Grace Period |
| `active_upb_late_16_30_perc` | % of active UPB 16-30 days late |
| `active_upb_late_31_120_perc` | % of active UPB 31-120 days late |
| `curr_wac` | Current weighted average coupon |
| `curr_wam` | Current weighted average remaining maturity |
| `curr_wala` | Current weighted average loan age |
| `curr_avg_fico` | Current average FICO (latest pull) |
| `curr_avg_dti` | Current average DTI |
| `upb_fully_paid_perc` | % of UPB that is fully paid |
| `upb_lost_perc` | % of UPB that is charged off |

**Example interpretation**: If filtering to Grade B, you might see:
- 662,776 loans originated with $9.5B total UPB
- 38% of active UPB is Current, with a current WAC of 11.2%
- 13% of original UPB was lost to charge-offs

#### Pool Stratification Charts

Two side-by-side bar charts:

1. **UPB by Grade**: Outstanding principal broken down by LC credit grade (A through G). Shows concentration risk — typically Grade B and C dominate.

2. **UPB by Vintage**: Outstanding principal by origination quarter. Shows the portfolio's vintage mix — recent vintages have more outstanding principal because less has amortized.

#### Performance Metrics Table

A table with one row per vintage quarter, showing historical performance:

| Column | Meaning |
|--------|---------|
| `vintage` | Origination quarter |
| `orig_loan_count` | Loans originated that quarter |
| `orig_upb_mm` | Original UPB in millions |
| `pct_active` | % of loans still active |
| `pct_current` | % of loans that are Current |
| `pct_fully_paid` | % of loans fully paid off |
| `pct_charged_off` | % of loans charged off |
| `pct_defaulted_count` | Default rate by count |
| `pct_defaulted_upb` | Default rate by UPB |
| `pool_cpr` | CPR for Current + Fully Paid (March 2019) loans (delinquent loans excluded) |
| `loss_severity` | Loss severity for charged-off loans |
| `recovery_rate` | Recovery rate (= 1 - loss severity) |

**Key insights to look for**:
- Older vintages have higher fully-paid and charged-off rates (more time to mature)
- Recent vintages (2017-2018) have higher active rates
- CPR and CDR vary by vintage, reflecting economic conditions at origination

#### Delinquency Transition Flow Table

Shows the probability of loans moving between delinquency states:

| Column | Meaning |
|--------|---------|
| `from_current_to_fully_paid_clean` | % of Current loans that paid off cleanly |
| `from_current_to_current_clean` | % of Current loans still current next period |
| `from_current_to_delinquent` | % of Current loans that became delinquent |
| `from_grace_still_in_grace` | % of Grace Period loans still in grace |
| `from_grace_progressed` | % of Grace Period loans that worsened |
| `from_late16_cured` | % of Late (16-30) loans that cured |
| `from_late16_still_in_late16` | % of Late (16-30) loans still in same bucket |
| `from_late16_progressed` | % of Late (16-30) loans that worsened |
| `from_late31_still_in_late31` | % of Late (31-120) loans still in same bucket |
| `from_late31_charged_off` | % of Late (31-120) loans charged off |

#### Age-Weighted Transition Matrix

Displays 7-state transition probabilities at individual monthly loan ages. The 7 states are: Current, Delinquent (0-30), Late_1, Late_2, Late_3, Charged Off, Fully Paid. The Late (31-120) bucket is split into 3 sub-states representing the delinquency pipeline.

Shown as an observation-weighted average heatmap with an expandable raw monthly probability table. These probabilities are the same ones used by the state-transition cash flow model in Tabs 2 and 3.

---

### Tab 2: Cash Flow Projection

This tab projects monthly cash flows using the state-transition model and computes investment returns.

**Population**: All active loans are used for projections: Current loans with March 2019 last payment date plus all delinquent loans (In Grace Period + Late 16-30 + Late 31-120) regardless of last payment date. CDR and loss severity use all loans in the filter for historical rates.

#### Base Assumptions (Metric Cards)

Cards showing the computed base-case assumptions:

| Metric | Source |
|--------|--------|
| CDR (Conditional) | Annualized from trailing 12-month average MDR (all loans in filter) |
| Avg MDR | Un-annualized monthly default rate |
| Full Payoff CPR | CPR from Fully Paid March 2019 loans only (full loan payoffs) |
| Curtailment CPR | CPR from Current loans' partial prepayments above scheduled installment |
| Loss Severity | From Charged Off loans with capped recoveries |
| Recovery Rate | 1 - Loss Severity |
| Cumulative Default Rate | Raw lifetime rate (reference only, NOT used in projections) |

Total CPR (full payoffs + curtailments combined) shown as caption.

#### Pool Characteristics (Metric Cards)

Four cards showing the pool being modeled:

| Metric | Definition |
|--------|-----------|
| Total UPB | Sum of outstanding principal for active March 2019 loans |
| WAC | Weighted average coupon (interest rate, weighted by UPB) |
| WAM | Weighted average remaining maturity (months) |
| Monthly Payment | Sum of all loan installments |

#### State-Transition Projection

**Projection Input**: Purchase Price (%) only. No CDR/CPR inputs — defaults and prepayments are driven entirely by age-specific empirical transition probabilities. Prepayment rates use the observed age-specific Current→Fully Paid rates directly, preserving the natural variation across loan ages. **Curtailments** (partial prepayments from loans that stay Current) are also applied at age-specific rates computed from Current loans' observed extra payments.

**How it works**: The pool is tracked across 7 states (Current, Delinquent (0-30), Late_1, Late_2, Late_3, Charged Off, Fully Paid). Each month, loan balances transition between states based on age-specific probabilities derived from the dataset. Defaults flow through a 5-month pipeline:

```
Current → Delinquent (0-30) → Late_1 → Late_2 → Late_3 → Charged Off
```

**Key consequence**: For an all-Current pool, the first defaults appear at month 5 (not month 1).

**UPB by State Chart**: A stacked area chart showing the pool balance broken down by state over time. Visualizes the delinquency pipeline progression.

**Price solver**: Uses `solve_price_transition()` — projects cash flows once (price-independent), then solves for price.

#### IRR Display and Price Solver

A metric card shows the **Projected IRR**. Next to it, enter a **Target IRR** (default 12%) and the tool computes the purchase price that would achieve that return. The solver searches between 50 and 150 cents on the dollar. If no price can achieve the target, it displays "No solution".

#### Monthly Cash Flow Components Chart

A stacked bar chart showing the composition of monthly cash flows, with a toggle to show as % of total:

| Component | Color | Meaning |
|-----------|-------|---------|
| Interest | Blue (#4A90D9) | Interest earned on performing balance |
| Scheduled Principal | Orange (#E67E22) | Regular amortization principal |
| Full Payoffs | Green (#2ECC71) | Full prepayment principal (Current→Fully Paid) |
| Curtailments | Teal (#1ABC9C) | Partial prepayments above scheduled installment |
| Recoveries | Purple (#9B59B6) | Post-default recoveries |

An overlay line shows the ending balance on a secondary y-axis (hidden in % mode).

#### Full Cash Flow Table (Expandable)

Click "View Full Cash Flow Table" to see the complete month-by-month projection with all columns: month, date, beginning balance, defaults, loss, recovery, interest, scheduled principal, prepayments, curtailments, total principal, ending balance, total cashflow.

---

### Tab 3: Scenario Analysis

This tab compares investment returns under three scenarios: Base, Stress, and Upside. Scenarios are grounded in empirical vintage-level CDR/CPR variation, not arbitrary sliders.

#### Tab 3 Controls

- **Purchase Price (%)**: Sets the purchase price for scenario IRR calculations (default 95%)
- **Editable CDR/CPR Table**: 3 rows (Base, Stress, Upside) × 2 columns (CDR %, CPR %). Base pre-populated from pool-level empirical values; Stress/Upside from P25/P75 vintage percentiles. Users can override any value.

#### Vintage Percentile Methodology

For each quarterly vintage in the current filter with >= 1,000 loans, the tool computes vintage-specific CDR and CPR. CDR and CPR have separate qualifying vintage counts (a vintage with many loans but no March 2019 Current/Fully Paid loans qualifies for CDR but not CPR). Unweighted P25/P75 percentiles across qualifying vintages determine the stress/upside scenario values:

| Scenario | CDR | CPR |
|----------|-----|-----|
| Base | Pool-level empirical | Pool-level empirical |
| Stress | P75 (high defaults) | P25 (low prepayments) |
| Upside | P25 (low defaults) | P75 (high prepayments) |

Base uses the pool-level CDR/CPR from `compute_pool_assumptions()`, which reflects the actual cohort behavior with proper weighting. Percentiles characterize the cross-vintage distribution for stress/upside bounds.

The caption below the inputs shows how many CDR and CPR vintages qualified. An expander reveals the per-vintage CDR/CPR distribution table.

**Fallback**: If fewer than 3 qualifying CDR vintages exist (e.g., very narrow filter), the tool warns and pre-populates with pool-level CDR/CPR ± 15%.

#### How CDR/CPR Drive Transition Probabilities

The scenario CDR scales the delinquency entry rate and cure rates in the transition matrix:
- `cdr_ratio = scenario_cdr / pool_cdr`
- Current→Delinquent rate scaled up/down by the CDR ratio
- Cure rates inversely scaled
- Late_3→Charged Off increases mechanically via re-normalization (not directly scaled)

Each scenario's CPR scales the age-specific Current→Fully Paid rates via a ratio (`cpr_ratio = scenario_cpr / base_cpr`), preserving the natural age-dependent prepayment shape while shifting the overall level. Curtailment rates are also scaled by the same ratio.

**Loss severity**: FIXED across all scenarios.

**Scenario Comparison Table**: One row per scenario with columns for CDR, CPR, loss severity, IRR, total interest/principal/losses/recoveries/defaults, and WAL.

**Projected Balance by Scenario**: Multi-line chart showing the pool balance over time for each scenario.

#### Expected Ordering

**Stress IRR < Base IRR < Upside IRR** — higher defaults always hurt returns.

The height difference between Stress and Base shows downside risk. The difference between Upside and Base shows potential upside.

---

## 6. Codebase Overview

### Source Modules (`src/`)

| Module | Functions | Purpose |
|--------|-----------|---------|
| `amortization.py` | `calc_amort`, `calc_monthly_payment`, `calc_balance`, `calc_payment_num` | Loan-level amortization calculations. Adds ~20 runtime columns to the DataFrame. |
| `portfolio_analytics.py` | `calculate_credit_metrics`, `calculate_performance_metrics`, `calculate_transition_matrix`, `reconstruct_loan_timeline`, `get_loan_status_at_age`, `compute_age_transition_probabilities`, `compute_pool_transition_matrix`, `compute_default_timing`, `compute_loan_age_status_matrix` | Pool stratification, vintage performance, delinquency transition flows, age-specific transition probabilities (5-state and 7-state). |
| `cashflow_engine.py` | `compute_pool_assumptions`, `compute_pool_characteristics`, `compute_implied_cpr`, `compute_curtailment_rates`, `calculate_irr`, `build_pool_state`, `project_cashflows_transition`, `solve_price_transition` | Pool-level cash flow projection engine (state-transition model), CPR split (full payoff + curtailment), age-specific curtailment rates, IRR, and price solver. Also contains `project_cashflows`, `solve_price`, and `adjust_prepayment_rates` (flat CDR/CPR model, available but not used in dashboard). |
| `scenario_analysis.py` | `compute_base_assumptions`, `compute_vintage_percentiles`, `build_scenarios_transition`, `build_scenarios_from_percentiles`, `compare_scenarios_transition` | Vintage percentile analysis, base/stress/upside scenario builder using percentile-derived CDR/CPR with ratio scaling. Also contains `build_scenarios` and `compare_scenarios` (flat CDR/CPR model, available but not used in dashboard). |

### Data Pipeline

```
Raw CSV → export_to_sqlite.py → loans.db → app.py (load_data) → calc_amort → DataFrame
                                                                      ↓
                                              Sidebar filters → df_filtered
                                                                      ↓
                                              ┌──────────────────────────────────────────┐
                                              │ Tab 1: Portfolio Analytics               │
                                              │   calculate_credit_metrics()             │
                                              │   calculate_performance_metrics()        │
                                              │   calculate_transition_matrix()          │
                                              │   compute_age_transition_probabilities() │
                                              │     (7-state, monthly ages)              │
                                              ├──────────────────────────────────────────┤
                                              │ Tab 2: Cash Flow Projection              │
                                              │   compute_pool_assumptions()             │
                                              │   compute_pool_characteristics()         │
                                              │   build_pool_state()                     │
                                              │   project_cashflows_transition()         │
                                              │   solve_price_transition()               │
                                              │   calculate_irr()                        │
                                              ├──────────────────────────────────────────┤
                                              │ Tab 3: Scenario Analysis                 │
                                              │   compute_vintage_percentiles()          │
                                              │   build_scenarios_from_percentiles()     │
                                              │   compare_scenarios_transition()         │
                                              └──────────────────────────────────────────┘
```

### Population Splits

Understanding which loans feed into which calculations is critical:

- **Tab 1** uses ALL loans matching the sidebar filter
- **Tabs 2 & 3** use two populations:
  - **CDR and Loss Severity**: ALL loans in the filter (need Charged Off loans)
  - **Pool Characteristics, Cash Flows**: Active loans — Current with `last_pymnt_d = 2019-03-01` plus all delinquent (In Grace + Late) regardless of last payment date
  - **CPR**: Current + Fully Paid loans with `last_pymnt_d = 2019-03-01`. Fully Paid March 2019 loans represent payoffs that month. Delinquent loans are excluded — they are behind on payments, not prepaying.

---

## 7. Common Workflows

### Evaluating a Specific Vintage

1. Set Strata Type to **Vintage**
2. Set Strata Value to **2018Q1**
3. Tab 1: Review credit metrics and performance for that vintage
4. Tab 2: Check the IRR at your desired purchase price
5. Tab 3: Compare base/stress/upside scenarios

### Finding the Right Purchase Price

1. Select your target pool (strata filters)
2. Go to Tab 2
3. Enter your target IRR in the price solver (e.g., 12%)
4. The solver returns the maximum price you can pay to achieve that return
5. Verify by setting the purchase price slider to the solved price — the displayed IRR should match

### Stress Testing a Portfolio

1. Select your pool
2. Go to Tab 3
3. Review the pre-populated CDR/CPR values derived from vintage percentiles
4. Observe the IRR spread between Stress and Base scenarios
5. Manually increase the Stress CDR or decrease the Stress CPR to test more extreme scenarios
6. A wider spread indicates more sensitivity to credit deterioration

### Comparing Grades

1. Set Strata Type to **Grade**
2. Select Grade **A**, review Tab 2 IRR
3. Select Grade **D**, review Tab 2 IRR
4. Higher-grade pools have lower CDR but also lower WAC, so the IRR tradeoff is interesting
5. Tab 3 shows whether the IRR advantage of lower-grade pools holds up under stress

---

## 8. Troubleshooting

### "No active loans with March 2019 payment date"

This warning appears on Tabs 2 and 3 when the filtered pool has no active loans (Current + In Grace + Late) with a March 2019 last payment. This happens for:
- Very old vintages (e.g., 2007-2010) where most loans have already paid off or charged off
- Filters that exclude all active loans

**Fix**: Select a more recent vintage or broader filter.

### Slow Initial Load

The first dashboard load runs `calc_amort()` on ~2.2M loans, which takes several minutes. This is cached — subsequent interactions are fast.

**Tip**: If you need faster iteration, filter to a specific strata before doing detailed analysis.

### IRR Shows "N/A"

The IRR calculation uses `numpy_financial.irr()`, which returns NaN when the cash flow series doesn't have a unique real root. This can happen with extreme parameters (very high CDR, very low purchase price).

### Price Solver Shows "No solution"

The solver searches for a price between 0.50 and 1.50 that achieves the target IRR. If the target is unrealistically high (e.g., 500%) or the cash flows are too small relative to any price in that range, no solution exists.

### Import Errors

Ensure you're running from the project root directory and the virtual environment is activated:

```bash
cd /path/to/aravalli
source .env/bin/activate    # or .env\Scripts\activate on Windows
streamlit run app.py
```

### Database Not Found

If you see a SQLite error, ensure `data/loans.db` exists:

```bash
python scripts/export_to_sqlite.py
```
