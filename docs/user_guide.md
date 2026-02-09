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
- Applies 20 cleaning steps (documented in [data_cleaning.md](data_cleaning.md))
- Writes cleaned data to `data/loans.db` (SQLite)
- Final count: 2,255,494 loans

**Expected runtime**: 2-5 minutes depending on hardware.

**Verify**: After running, `data/loans.db` should exist and be approximately 2-3 GB.

---

## 3. Running Tests

```bash
pytest tests/ -v
```

The test suite contains 101 tests across 4 modules:

| Test File | Tests | Coverage |
|-----------|------:|----------|
| `test_amortization.py` | 19 | Monthly payment, balance, payment count, full calc_amort |
| `test_portfolio_analytics.py` | 29 | Credit metrics, performance metrics, transition matrix |
| `test_cashflow_engine.py` | 31 | Pool assumptions, characteristics, cash flows, IRR, price solver |
| `test_scenario_analysis.py` | 22 | Base assumptions, scenario builder, scenario comparison |

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

The sidebar contains four controls that affect all three tabs:

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

#### Purchase Price Slider

Sets the purchase price as a fraction of UPB (unpaid principal balance):

- **Range**: 0.50 to 1.20
- **Default**: 0.95 (95 cents on the dollar)
- **Step**: 0.01

This controls the initial outlay in IRR calculations (Tab 2) and scenario analysis (Tab 3). A value of 0.95 means the investor pays $95 for every $100 of outstanding principal.

#### Stress / Upside % Slider

Controls the multiplicative shift applied to CDR and CPR for stress and upside scenarios:

- **Range**: 0.05 to 0.50
- **Default**: 0.15 (15%)
- **Step**: 0.01

A value of 0.15 means:
- Stress CDR = Base CDR x 1.15 (15% higher defaults)
- Stress CPR = Base CPR x 0.85 (15% lower prepayments)
- Upside CDR = Base CDR x 0.85 (15% lower defaults)
- Upside CPR = Base CPR x 1.15 (15% higher prepayments)

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
| `pool_cpr_active` | CPR for all active loans |
| `pool_cpr_current` | CPR for Current loans only |
| `pct_prepaid_active` | % of active loans that prepaid |
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

---

### Tab 2: Cash Flow Projection

This tab projects monthly cash flows for the filtered pool and computes investment returns.

**Population**: Only Current loans with March 2019 last payment date are used for projections. CDR and loss severity use all loans in the filter for historical rates.

#### Base Assumptions (Metric Cards)

Four cards showing the computed base-case assumptions:

| Metric | Source |
|--------|--------|
| CDR | From all loans in filter (Charged Off / total) |
| CPR | From Current March 2019 loans (pool-level SMM annualized) |
| Loss Severity | From Charged Off loans with capped recoveries |
| Recovery Rate | 1 - Loss Severity |

#### Pool Characteristics (Metric Cards)

Four cards showing the pool being modeled:

| Metric | Definition |
|--------|-----------|
| Total UPB | Sum of outstanding principal for Current March 2019 loans |
| WAC | Weighted average coupon (interest rate, weighted by UPB) |
| WAM | Weighted average remaining maturity (months) |
| Monthly Payment | Sum of all loan installments |

#### IRR Display

A prominent metric card showing the **Projected IRR** — the annualized internal rate of return at the selected purchase price, using the base-case CDR, CPR, and loss severity.

**Example**: At 95 cents on the dollar with CDR=8%, CPR=12%, and loss severity=85%, the IRR might be ~10-15% depending on the pool's WAC.

#### Price Solver

Next to the IRR display, enter a **Target IRR** (default 12%) and the tool computes the purchase price that would achieve that return.

**Example**: If you want a 12% IRR, the solver might return 0.9234, meaning you'd need to pay 92.34 cents on the dollar.

The solver uses Brent's method to search between 50 cents and 150 cents on the dollar. If no price can achieve the target IRR (e.g., target is unrealistically high), it displays "No solution".

#### Projected Pool Balance Chart

A line chart showing the ending balance over time as the pool amortizes, defaults, and prepays. The curve should decline smoothly from total UPB to zero.

**What to look for**:
- Steeper decline = faster runoff (higher CDR and/or CPR)
- The x-axis shows dates from April 2019 through the WAM endpoint
- If the balance reaches zero before WAM, the pool paid off early

#### Monthly Cash Flow Components Chart

A stacked area chart showing the composition of monthly cash flows:

| Component | Color | Meaning |
|-----------|-------|---------|
| Interest | Blue (#636EFA) | Interest earned on performing balance |
| Scheduled Principal | Orange (#FFA15A) | Regular amortization principal |
| Prepayments | Green (#00CC96) | Voluntary prepayment principal |
| Recoveries | Purple (#AB63FA) | Post-default recoveries |

**What to look for**:
- Interest starts high and declines as the balance decreases
- Scheduled principal starts low and increases (standard amortization behavior)
- Prepayments are relatively constant as a fraction of remaining balance
- Recoveries are small relative to other components

#### Full Cash Flow Table (Expandable)

Click "View Full Cash Flow Table" to see the complete month-by-month projection with all columns: month, date, beginning balance, defaults, loss, recovery, interest, scheduled principal, prepayments, total principal, ending balance, total cashflow.

---

### Tab 3: Scenario Analysis

This tab compares investment returns under three scenarios: Base, Stress, and Upside.

#### Scenario Assumptions Table

Shows the CDR, CPR, and loss severity for each scenario:

| Scenario | CDR | CPR | Loss Severity |
|----------|-----|-----|---------------|
| Base | Historical CDR | Historical CPR | Historical LS |
| Stress | CDR x (1 + pct) | CPR x (1 - pct) | Same as Base |
| Upside | CDR x (1 - pct) | CPR x (1 + pct) | Same as Base |

Where `pct` is the Stress/Upside % slider value (default 15%).

#### Scenario Comparison Table

A summary table with one row per scenario:

| Column | Meaning |
|--------|---------|
| scenario | Base / Stress / Upside |
| cdr | Annual default rate used |
| cpr | Annual prepayment rate used |
| loss_severity | Loss severity (same for all) |
| irr | Annualized internal rate of return |
| total_interest | Sum of all interest cash flows |
| total_principal | Sum of all principal cash flows |
| total_losses | Sum of all losses |
| total_recoveries | Sum of all recoveries |
| weighted_avg_life | Average years until principal return |

**Expected ordering**: Stress IRR < Base IRR < Upside IRR.

#### IRR by Scenario Bar Chart

A color-coded bar chart showing IRR for each scenario:
- **Base**: Blue (#636EFA)
- **Stress**: Red (#EF553B)
- **Upside**: Green (#00CC96)

The height difference between Stress and Base shows downside risk. The difference between Upside and Base shows potential upside.

#### Projected Balance by Scenario Chart

Three overlaid line charts showing the pool balance over time under each scenario:
- **Stress**: Slowest decline (fewer prepayments keep balance higher despite more defaults)
- **Base**: Middle
- **Upside**: Fastest decline (more prepayments accelerate runoff)

---

## 6. Codebase Overview

### Source Modules (`src/`)

| Module | Functions | Purpose |
|--------|-----------|---------|
| `amortization.py` | `calc_amort`, `calc_monthly_payment`, `calc_balance`, `calc_payment_num` | Loan-level amortization calculations. Adds ~20 runtime columns to the DataFrame. |
| `portfolio_analytics.py` | `calculate_credit_metrics`, `calculate_performance_metrics`, `calculate_transition_matrix` | Pool stratification, vintage performance, delinquency transition flows. |
| `cashflow_engine.py` | `compute_pool_assumptions`, `compute_pool_characteristics`, `project_cashflows`, `calculate_irr`, `solve_price` | Pool-level cash flow projection engine with IRR and price solver. |
| `scenario_analysis.py` | `compute_base_assumptions`, `build_scenarios`, `compare_scenarios` | Base/stress/upside scenario builder and comparison. |

### Data Pipeline

```
Raw CSV → export_to_sqlite.py → loans.db → app.py (load_data) → calc_amort → DataFrame
                                                                      ↓
                                              Sidebar filters → df_filtered
                                                                      ↓
                                              ┌─────────────────────────────────┐
                                              │ Tab 1: Portfolio Analytics      │
                                              │   calculate_credit_metrics()    │
                                              │   calculate_performance_metrics()│
                                              │   calculate_transition_matrix() │
                                              ├─────────────────────────────────┤
                                              │ Tab 2: Cash Flow Projection     │
                                              │   compute_pool_assumptions()    │
                                              │   compute_pool_characteristics()│
                                              │   project_cashflows()           │
                                              │   calculate_irr()               │
                                              │   solve_price()                 │
                                              ├─────────────────────────────────┤
                                              │ Tab 3: Scenario Analysis        │
                                              │   build_scenarios()             │
                                              │   compare_scenarios()           │
                                              └─────────────────────────────────┘
```

### Population Splits

Understanding which loans feed into which calculations is critical:

- **Tab 1** uses ALL loans matching the sidebar filter
- **Tabs 2 & 3** use two populations:
  - **CDR and Loss Severity**: ALL loans in the filter (need Charged Off loans)
  - **CPR, Pool Characteristics, Cash Flows**: Current loans with `last_pymnt_d = 2019-03-01` only

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
3. Start with the default 15% stress/upside shift
4. Observe the IRR spread between Stress and Base scenarios
5. Increase the stress % to see how much the IRR degrades
6. A wider spread indicates more sensitivity to credit deterioration

### Comparing Grades

1. Set Strata Type to **Grade**
2. Select Grade **A**, review Tab 2 IRR
3. Select Grade **D**, review Tab 2 IRR
4. Higher-grade pools have lower CDR but also lower WAC, so the IRR tradeoff is interesting
5. Tab 3 shows whether the IRR advantage of lower-grade pools holds up under stress

---

## 8. Troubleshooting

### "No Current loans with March 2019 payment date"

This warning appears on Tabs 2 and 3 when the filtered pool has no Current loans with a March 2019 last payment. This happens for:
- Very old vintages (e.g., 2007-2010) where most loans have already paid off or charged off
- Filters that exclude all Current loans

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
