# Skill: Streamlit Dashboard Patterns

Reference this skill when building or modifying `app.py` — the Streamlit dashboard that surfaces all three project parts.

---

## Architecture

Single-file Streamlit app (`app.py`) with:
- **Sidebar**: Strata filters only (shared across tabs). Purchase price, CDR/CPR, and stress controls live on their respective tabs.
- **Tab 1**: Portfolio Analytics (Part 1)
- **Tab 2**: Cash Flow Projection & IRR (Part 2)
- **Tab 3**: Scenario Analysis (Part 3)

The app imports from `src/` modules — no financial logic lives in `app.py`. The app's job is: load data → filter → call functions → display results.

---

## App Structure Template

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from src.amortization import calc_amort
from src.portfolio_analytics import (
    calculate_credit_metrics,
    calculate_performance_metrics,
    calculate_transition_matrix
)
from src.cashflow_engine import (
    compute_pool_assumptions,
    compute_pool_characteristics,
    project_cashflows,
    calculate_irr,
    solve_price
)
from src.scenario_analysis import build_scenarios, compare_scenarios

st.set_page_config(page_title="Lending Club Portfolio Analyzer", layout="wide")
st.title("Lending Club Loan Portfolio Investment Analysis")

# --- Data Loading ---
@st.cache_data
def load_data():
    conn = sqlite3.connect('data/loans.db')
    df = pd.read_sql('SELECT * FROM loans', conn)
    conn.close()
    # Run calc_amort to add engineered columns
    df = calc_amort(df, verbose=False)
    return df

df = load_data()

# --- Sidebar Filters ---
# [filter widgets here]

# --- Apply Filters ---
df_filtered = df.copy()
# [apply each filter]

# --- Tabs ---
tab1, tab2, tab3 = st.tabs([
    "📊 Portfolio Analytics",
    "💰 Cash Flow Projection",
    "📈 Scenario Analysis"
])

with tab1:
    # Part 1 content
    pass

with tab2:
    # Part 2 content
    pass

with tab3:
    # Part 3 content
    pass
```

---

## Sidebar Filter Pattern

Each filter follows the same pattern: multiselect with "Select All" as default.

```python
with st.sidebar:
    st.header("Portfolio Filters")

    # Grade filter
    all_grades = sorted(df['grade'].dropna().unique())
    selected_grades = st.multiselect(
        "Grade",
        options=all_grades,
        default=all_grades,
        key="grade_filter"
    )

    # Term filter
    all_terms = sorted(df['term_months'].dropna().unique())
    selected_terms = st.multiselect(
        "Term (months)",
        options=all_terms,
        default=all_terms,
        key="term_filter"
    )

    # Vintage filter
    all_vintages = sorted(df['issue_quarter'].dropna().unique())
    selected_vintages = st.multiselect(
        "Vintage",
        options=all_vintages,
        default=all_vintages,
        key="vintage_filter"
    )

    # Purpose filter
    all_purposes = sorted(df['purpose'].dropna().unique())
    selected_purposes = st.multiselect(
        "Purpose",
        options=all_purposes,
        default=all_purposes,
        key="purpose_filter"
    )

    # State filter
    all_states = sorted(df['addr_state'].dropna().unique())
    selected_states = st.multiselect(
        "State",
        options=all_states,
        default=all_states,
        key="state_filter"
    )
```

### Applying Filters

```python
# Apply all filters
df_filtered = df[
    (df['grade'].isin(selected_grades)) &
    (df['term_months'].isin(selected_terms)) &
    (df['issue_quarter'].isin(selected_vintages)) &
    (df['purpose'].isin(selected_purposes)) &
    (df['addr_state'].isin(selected_states))
]

# Show filter summary in sidebar
st.sidebar.markdown("---")
st.sidebar.metric("Filtered Loans", f"{len(df_filtered):,}")
st.sidebar.metric("Total UPB", f"${df_filtered['out_prncp'].sum():,.0f}")
```

---

## Tab 1: Portfolio Analytics

### Layout

```
┌─────────────────────────────────────────────────┐
│ Summary Metrics Row (4 columns)                 │
│ [Total Loans] [Total UPB] [Avg FICO] [Avg DTI] │
├─────────────────────────────────────────────────┤
│ Credit Metrics Table (from calculate_credit_metrics) │
├────────────────────┬────────────────────────────┤
│ Stratification     │ Performance Metrics         │
│ Charts (grade,     │ Table (CDR, CPR, loss       │
│ term, vintage)     │ severity by vintage)        │
├────────────────────┴────────────────────────────┤
│ Transition Matrix / Roll Rates                   │
└─────────────────────────────────────────────────┘
```

### Key Patterns

```python
with tab1:
    st.subheader("Pool Summary")

    # Summary metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Loans", f"{len(df_filtered):,}")
    col2.metric("Total UPB", f"${df_filtered['out_prncp'].sum():,.0f}")
    col3.metric("Avg FICO", f"{df_filtered['original_fico'].mean():.0f}")
    col4.metric("Avg DTI", f"{df_filtered['dti_clean'].mean():.1f}%")

    # Credit metrics
    st.subheader("Credit Metrics")
    credit_metrics = calculate_credit_metrics(df_filtered)
    st.dataframe(credit_metrics, use_container_width=True)

    # Stratification charts
    st.subheader("Pool Stratification")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        grade_dist = df_filtered.groupby('grade')['out_prncp'].sum().reset_index()
        fig = px.bar(grade_dist, x='grade', y='out_prncp',
                     title="UPB by Grade",
                     labels={'out_prncp': 'Outstanding Principal ($)'})
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        vintage_dist = df_filtered.groupby('issue_quarter')['out_prncp'].sum().reset_index()
        fig = px.bar(vintage_dist, x='issue_quarter', y='out_prncp',
                     title="UPB by Vintage")
        st.plotly_chart(fig, use_container_width=True)
```

---

## Tab 2: Cash Flow Projection

### Layout

```
┌─────────────────────────────────────────────────┐
│ Historical Base Assumptions (read-only metrics)   │
├─────────────────────────────────────────────────┤
│ Input Controls Row (all in % format)             │
│ [Purchase Price %] [CDR %] [CPR %]              │
├─────────────────────────────────────────────────┤
│ Pool Characteristics (WAC, WAM, UPB, etc.)      │
├─────────────────────────────────────────────────┤
│ IRR Result (big number) + Solved Price           │
├─────────────────────────────────────────────────┤
│ Cash Flow Chart (balance over time + components) │
├─────────────────────────────────────────────────┤
│ Cash Flow Table (expandable)                     │
└─────────────────────────────────────────────────┘
```

### Key Patterns

```python
with tab2:
    st.subheader("Cash Flow Projection & IRR")

    # Separate populations
    df_all_filtered = df_filtered  # all loans for CDR
    non_current_active = ['In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
    df_active_march = df_filtered[
        ((df_filtered['loan_status'] == 'Current') & (df_filtered['last_pymnt_d'] == '2019-03-01'))
        | (df_filtered['loan_status'].isin(non_current_active))
    ]

    if len(df_active_march) == 0:
        st.warning("No active loans with March 2019 payment date in this selection.")
        st.stop()

    # Compute historical assumptions
    pool_assumptions = compute_pool_assumptions(df_all_filtered, df_active_march)
    pool_chars = compute_pool_characteristics(df_active_march)

    # User-adjustable inputs (all in percentage format, converted to decimal internally)
    i1, i2, i3 = st.columns(3)
    with i1:
        t2_purchase_price_pct = st.number_input("Purchase Price (%)", 50.0, 120.0, 95.0, 0.01, format="%.2f")
    with i2:
        t2_cdr_pct = st.number_input("CDR (%)", 0.0, 100.0, round(pool_assumptions['cdr']*100, 2), 0.01, format="%.2f")
    with i3:
        t2_cpr_pct = st.number_input("CPR (%)", 0.0, 100.0, round(pool_assumptions['cpr']*100, 2), 0.01, format="%.2f")

    purchase_price = t2_purchase_price_pct / 100
    cdr = t2_cdr_pct / 100
    cpr = t2_cpr_pct / 100

    # Project cash flows
    cf_df = project_cashflows(pool_chars, cdr, cpr, pool_assumptions['loss_severity'], purchase_price)
    irr = calculate_irr(cf_df, pool_chars, purchase_price)

    # Display IRR and price solver (all in percentage format)
    irr_col, price_col = st.columns(2)
    irr_col.metric("Projected IRR", f"{irr * 100:.2f}%")

    target_irr_pct = st.number_input("Target IRR (%) for price solver", 0.0, 100.0, 12.0, 0.01, format="%.2f")
    solved = solve_price(pool_chars, target_irr_pct / 100, cdr, cpr, pool_assumptions['loss_severity'])
    price_col.metric(f"Price for {target_irr_pct:.2f}% IRR", f"{solved * 100:.2f}%")

    # Cash flow chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cf_df['month'], y=cf_df['interest'],
                             name='Interest', stackgroup='one'))
    fig.add_trace(go.Scatter(x=cf_df['month'], y=cf_df['scheduled_principal'],
                             name='Scheduled Principal', stackgroup='one'))
    fig.add_trace(go.Scatter(x=cf_df['month'], y=cf_df['prepayments'],
                             name='Prepayments', stackgroup='one'))
    fig.add_trace(go.Scatter(x=cf_df['month'], y=cf_df['recovery'],
                             name='Recoveries', stackgroup='one'))
    fig.update_layout(title="Monthly Cash Flow Components",
                      yaxis_title="$", xaxis_title="Month")
    st.plotly_chart(fig, use_container_width=True)

    # Balance over time
    fig2 = px.line(cf_df, x='month', y='ending_balance',
                   title="Projected Pool Balance")
    fig2.update_layout(yaxis_title="$", xaxis_title="Month")
    st.plotly_chart(fig2, use_container_width=True)

    # Expandable table
    with st.expander("View Full Cash Flow Table"):
        st.dataframe(cf_df, use_container_width=True)
```

---

## Tab 3: Scenario Analysis

### Layout

```
┌─────────────────────────────────────────────────┐
│ Scenario Controls                                │
│ [Purchase Price %] [Stress / Upside % slider]    │
├─────────────────────────────────────────────────┤
│ Scenario Comparison Table                        │
│ Scenario | CDR | CPR | Loss Sev | IRR | Price   │
├─────────────────────────────────────────────────┤
│ IRR Comparison Bar Chart                         │
├─────────────────────────────────────────────────┤
│ Balance Over Time (all 3 scenarios overlaid)     │
├─────────────────────────────────────────────────┤
│ Cash Flow Comparison Charts                      │
└─────────────────────────────────────────────────┘
```

### Key Patterns

```python
with tab3:
    st.subheader("Scenario Analysis")

    # Tab 3 controls (purchase price % + stress/upside slider)
    s1, s2 = st.columns(2)
    with s1:
        t3_price_pct = st.number_input("Purchase Price (%)", 50.0, 120.0, 95.0, 0.01, format="%.2f", key="t3_pp")
    with s2:
        stress_upside_pct = st.slider("Stress / Upside %", 0.05, 0.50, 0.15, 0.01, key="t3_su")

    t3_price = t3_price_pct / 100
    scenarios = build_scenarios(base_assumptions, stress_pct=stress_upside_pct, upside_pct=stress_upside_pct)
    comparison = compare_scenarios(pool_chars_t3, scenarios, t3_price)

    # Comparison table
    st.dataframe(comparison, use_container_width=True)

    # IRR bar chart
    fig = px.bar(comparison, x='scenario', y='irr',
                 color='scenario',
                 title="IRR by Scenario",
                 color_discrete_map={
                     'Base': '#636EFA',
                     'Stress': '#EF553B',
                     'Upside': '#00CC96'
                 })
    fig.update_layout(yaxis_tickformat='.1%')
    st.plotly_chart(fig, use_container_width=True)
```

---

## Plotly Styling Conventions

### Color Palette

| Element | Color | Hex |
|---------|-------|-----|
| Base scenario | Blue | #636EFA |
| Stress scenario | Red | #EF553B |
| Upside scenario | Green | #00CC96 |
| Interest component | Light blue | #636EFA |
| Principal component | Orange | #FFA15A |
| Prepayment component | Green | #00CC96 |
| Recovery component | Purple | #AB63FA |
| Loss component | Red | #EF553B |

### Number Formatting

```python
# Currencies: comma-separated, no decimals for large numbers
f"${value:,.0f}"           # $1,234,567

# Percentages: 2 decimal places
f"{rate:.2%}"              # 12.34%

# IRR: 2 decimal places as percentage
f"{irr * 100:.2f}%"        # 10.25%

# Purchase price: 2 decimal places as percentage
f"{price * 100:.2f}%"      # 95.23%

# CDR/CPR: 2 decimal places as percentage
f"{cdr * 100:.2f}%"        # 8.73%

# Plotly axis formats
fig.update_layout(yaxis_tickformat='$,.0f')   # dollar axis
fig.update_layout(yaxis_tickformat='.1%')      # percentage axis
```

### Chart Sizing

- Always use `use_container_width=True` for `st.plotly_chart()`
- Charts default to the full width of their container column
- Set explicit height only when needed: `fig.update_layout(height=400)`

---

## Caching Strategy

```python
@st.cache_data
def load_data():
    """Cache the full dataset load + calc_amort. Only re-runs if loans.db changes."""
    conn = sqlite3.connect('data/loans.db')
    df = pd.read_sql('SELECT * FROM loans', conn)
    conn.close()
    df = calc_amort(df, verbose=False)
    return df
```

- **Cache the data load**: `load_data()` with `@st.cache_data` — this is the expensive operation (~2.2M rows + calc_amort)
- **Don't cache filter application**: Filtering is fast on a DataFrame already in memory
- **Don't cache financial calculations**: They depend on user inputs (purchase price, CDR/CPR) which change constantly
- **Don't cache charts**: They're rebuilt each time inputs change

---

## Error Handling

Always handle the empty-cohort case gracefully:

```python
# After filtering
if len(df_filtered) == 0:
    st.warning("No loans match the current filters. Adjust your selections.")
    st.stop()

# Before cash flow projection
non_current_active = ['In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
df_active_march = df_filtered[
    ((df_filtered['loan_status'] == 'Current') & (df_filtered['last_pymnt_d'] == '2019-03-01'))
    | (df_filtered['loan_status'].isin(non_current_active))
]
if len(df_active_march) == 0:
    st.warning("No active loans with March 2019 payment date in this selection. "
               "Cash flow projections require active loans.")
    st.stop()

# After IRR calculation
if np.isnan(irr) or np.isinf(irr):
    st.error("IRR could not be computed. Check your assumptions.")
```

---

## Performance Tips

1. **Load data once**: Use `@st.cache_data` for the database load. Don't reconnect to SQLite on every interaction.
2. **Avoid rerunning calc_amort on filtered data**: Run it once on the full dataset in `load_data()`, then filter the result.
3. **Use `st.stop()`**: When a filter produces an empty set, stop early instead of letting downstream code fail with cryptic errors.
4. **Column selection**: If only certain columns are needed for a specific tab, select them early to reduce memory usage in intermediate operations.
