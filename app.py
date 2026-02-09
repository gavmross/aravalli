"""
Lending Club Loan Portfolio Investment Analysis Dashboard.

Single-page Streamlit app with sidebar filters and three tabs:
  Tab 1: Portfolio Analytics (credit metrics, performance, transitions)
  Tab 2: Cash Flow Projection & IRR
  Tab 3: Scenario Analysis (base/stress/upside)
"""

import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.amortization import calc_amort
from src.cashflow_engine import (
    compute_pool_assumptions,
    compute_pool_characteristics,
    project_cashflows,
    calculate_irr,
    solve_price,
)
from src.portfolio_analytics import (
    calculate_credit_metrics,
    calculate_performance_metrics,
    calculate_transition_matrix,
    reconstruct_loan_timeline,
    compute_age_transition_probabilities,
    compute_pool_transition_matrix,
    compute_default_timing,
    compute_loan_age_status_matrix,
)
from src.scenario_analysis import build_scenarios, compare_scenarios
from src.transition_viz import render_sankey_diagram

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Lending Club Portfolio Analyzer", layout="wide")
st.title("Lending Club Loan Portfolio Investment Analysis")

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load loans.db and run calc_amort to add engineered columns."""
    conn = sqlite3.connect("data/loans.db")
    df = pd.read_sql("SELECT * FROM loans", conn)
    conn.close()
    df = calc_amort(df, verbose=False)
    return df


@st.cache_data
def compute_irr_grid(
    _pool_chars: dict,
    cdr_values: tuple,
    cpr_values: tuple,
    loss_severity: float,
    purchase_price: float,
) -> list:
    """Compute IRR for each (CDR, CPR) combination. Returns 2D list."""
    irr_matrix = []
    for cdr in cdr_values:
        row = []
        for cpr in cpr_values:
            cf = project_cashflows(_pool_chars, cdr, cpr, loss_severity, purchase_price)
            irr = calculate_irr(cf, _pool_chars, purchase_price)
            row.append(irr if not (np.isnan(irr) or np.isinf(irr)) else None)
        irr_matrix.append(row)
    return irr_matrix


@st.cache_data
def cached_reconstruct_timeline(_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct loan timelines for backsolve analysis."""
    return reconstruct_loan_timeline(_df)


@st.cache_data
def cached_age_transition_probs(_df: pd.DataFrame, bucket_size: int) -> pd.DataFrame:
    """Compute age-bucketed transition probabilities."""
    return compute_age_transition_probabilities(_df, bucket_size=bucket_size)


df = load_data()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Portfolio Filters")

    # Strata type dropdown
    strata_options = {
        "ALL": "ALL",
        "Grade": "grade",
        "Term": "term_months",
        "Purpose": "purpose",
        "State": "addr_state",
        "Vintage": "issue_quarter",
    }
    strata_label = st.selectbox("Strata Type", options=list(strata_options.keys()))
    strata_col = strata_options[strata_label]

    # Strata value dropdown (disabled if ALL)
    if strata_col == "ALL":
        st.selectbox("Strata Value", options=["ALL"], disabled=True)
        strata_value = "ALL"
    else:
        unique_values = sorted(df[strata_col].dropna().unique())
        strata_value = st.selectbox(
            "Strata Value",
            options=["ALL"] + [str(v) for v in unique_values],
        )

    # Purchase price and stress/upside controls are on their respective tabs

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
if strata_col != "ALL" and strata_value != "ALL":
    # Cast strata_value to match column dtype
    col_dtype = df[strata_col].dtype
    if pd.api.types.is_integer_dtype(col_dtype):
        filter_val = int(strata_value)
    elif pd.api.types.is_float_dtype(col_dtype):
        filter_val = float(strata_value)
    else:
        filter_val = strata_value
    df_filtered = df[df[strata_col] == filter_val].copy()
else:
    df_filtered = df.copy()

if len(df_filtered) == 0:
    st.warning("No loans match the current filters. Adjust your selections.")
    st.stop()

# Sidebar summary
with st.sidebar:
    st.markdown("---")
    st.metric("Filtered Loans", f"{len(df_filtered):,}")
    st.metric("Total UPB", f"${df_filtered['out_prncp'].sum():,.0f}")

# ---------------------------------------------------------------------------
# Population splits
# ---------------------------------------------------------------------------
df_current_march = df_filtered[
    (df_filtered["loan_status"] == "Current")
    & (df_filtered["last_pymnt_d"] == "2019-03-01")
].copy()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["Portfolio Analytics", "Cash Flow Projection", "Scenario Analysis"]
)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Portfolio Analytics
# ═══════════════════════════════════════════════════════════════════════════
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
    credit_strata = strata_col if strata_col != "ALL" else "grade"
    credit_metrics = calculate_credit_metrics(
        df_filtered, strata_col=credit_strata, verbose=False
    )

    # Format for display
    cm_display = credit_metrics.copy()
    pct_cols = [c for c in cm_display.columns if c.endswith("_perc")]
    for c in pct_cols:
        cm_display[c] = cm_display[c].apply(lambda x: f"{x * 100:.2f}%")
    for c in ["orig_wac", "curr_wac"]:
        if c in cm_display.columns:
            cm_display[c] = cm_display[c].apply(lambda x: f"{x * 100:.2f}%")
    for c in ["orig_total_upb_mm", "active_total_upb_mm"]:
        if c in cm_display.columns:
            cm_display[c] = cm_display[c].apply(lambda x: f"${x:,.1f}M")
    cm_display = cm_display.rename(columns={
        "strata_type": "Strata Type",
        "strata_value": "Strata",
        "orig_total_upb_mm": "Orig. UPB ($M)",
        "orig_loan_count": "Orig. Loans",
        "orig_wac": "Orig. WAC",
        "orig_wam": "Orig. WAM (mo)",
        "orig_avg_fico": "Orig. Avg FICO",
        "orig_avg_dti": "Orig. Avg DTI",
        "active_total_upb_mm": "Active UPB ($M)",
        "active_upb_current_perc": "% Current",
        "active_upb_grace_perc": "% Grace",
        "active_upb_late_16_30_perc": "% Late 16-30",
        "active_upb_late_31_120_perc": "% Late 31-120",
        "curr_wac": "WAC (Current Only)",
        "curr_wam": "WAM (Current Only, mo)",
        "curr_wala": "WALA (Current Only, mo)",
        "curr_avg_fico": "Avg FICO (Current Only)",
        "curr_avg_dti": "Avg DTI (Current Only)",
        "upb_fully_paid_perc": "% Fully Paid of OUPB",
        "upb_lost_perc": "% Lost of OUPB",
    })
    st.dataframe(cm_display, use_container_width=True, hide_index=True)

    # Stratification charts
    st.subheader("Pool Stratification")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        grade_dist = (
            df_filtered.groupby("grade")["out_prncp"]
            .sum()
            .reset_index()
            .sort_values("grade")
        )
        fig = px.bar(
            grade_dist,
            x="grade",
            y="out_prncp",
            title="UPB by Grade",
            labels={"out_prncp": "Outstanding Principal ($)", "grade": "Grade"},
            color_discrete_sequence=["#636EFA"],
        )
        fig.update_layout(yaxis_tickformat="$,.0f")
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        vintage_dist = (
            df_filtered.groupby("issue_quarter")["out_prncp"]
            .sum()
            .reset_index()
            .sort_values("issue_quarter")
        )
        fig = px.bar(
            vintage_dist,
            x="issue_quarter",
            y="out_prncp",
            title="UPB by Vintage",
            labels={"out_prncp": "Outstanding Principal ($)", "issue_quarter": "Vintage"},
            color_discrete_sequence=["#FFA15A"],
        )
        fig.update_layout(yaxis_tickformat="$,.0f")
        st.plotly_chart(fig, use_container_width=True)

    # Performance metrics by vintage
    st.subheader("Performance Metrics by Vintage")
    perf_metrics = calculate_performance_metrics(
        df_filtered, vintage_col="issue_quarter", verbose=False
    )

    pm_display = perf_metrics.copy()
    # Drop pool_cpr_active — only show CPR for Current loans
    if "pool_cpr_active" in pm_display.columns:
        pm_display = pm_display.drop(columns=["pool_cpr_active"])
    pm_pct_cols = [
        "pct_active", "pct_current", "pct_fully_paid", "pct_charged_off",
        "pct_defaulted_count", "pct_defaulted_upb",
        "pool_cpr_current", "pct_prepaid_current", "loss_severity", "recovery_rate",
    ]
    for c in pm_pct_cols:
        if c in pm_display.columns:
            pm_display[c] = pm_display[c].apply(lambda x: f"{x * 100:.2f}%")
    if "orig_upb_mm" in pm_display.columns:
        pm_display["orig_upb_mm"] = pm_display["orig_upb_mm"].apply(
            lambda x: f"${x:,.1f}M"
        )
    pm_display = pm_display.rename(columns={
        "vintage": "Vintage",
        "orig_loan_count": "Orig. Loans",
        "orig_upb_mm": "Orig. UPB ($M)",
        "pct_active": "% Active",
        "pct_current": "% Current",
        "pct_fully_paid": "% Fully Paid",
        "pct_charged_off": "% Charged Off",
        "pct_defaulted_count": "Loans Defaulted of Originated",
        "pct_defaulted_upb": "UPB Defaulted of OUPB",
        "pool_cpr_current": "CPR (Current Only)",
        "pct_prepaid_current": "UPB Prepaid of OUPB",
        "loss_severity": "Total Loss Severity",
        "recovery_rate": "Total Recovery Rate",
    })
    st.dataframe(pm_display, use_container_width=True, hide_index=True)

    # Transition matrix
    st.subheader("Delinquency Transition Flow")
    trans_strata = strata_col if strata_col != "ALL" else None
    transition = calculate_transition_matrix(
        df_filtered, strata_col=trans_strata, verbose=False
    )

    # Pick the row for the Sankey — use "ALL" row (aggregate for current filter)
    all_row = transition[transition["strata_value"] == "ALL"].iloc[0]
    sankey_label = (
        f"{strata_label}: {strata_value}" if strata_value != "ALL" else "ALL"
    )
    fig_sankey = render_sankey_diagram(
        all_row, total_loans=int(all_row["total_loans"]), strata_label=sankey_label
    )
    st.plotly_chart(fig_sankey, use_container_width=True, theme=None)

    # Raw table in expander
    with st.expander("View raw transition data"):
        tm_display = transition.copy()
        flow_cols = [c for c in tm_display.columns if c.startswith("from_")]
        for c in flow_cols:
            tm_display[c] = tm_display[c].apply(lambda x: f"{x * 100:.2f}%")
        st.dataframe(tm_display, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------
    # NEW: Backsolve-based analytics
    # -------------------------------------------------------------------
    st.markdown("---")

    # Reconstruct loan timelines once for all new visualizations
    with st.spinner("Reconstructing loan timelines..."):
        df_enriched = cached_reconstruct_timeline(df_filtered)

    # Current loans with March 2019 payment (with timeline columns)
    # Note: no .copy() — the enriched df has mixed dtypes that cause
    # pandas consolidation errors on copy; this slice is read-only
    df_current_march_enriched = df_enriched[
        (df_enriched["loan_status"] == "Current")
        & (df_enriched["last_pymnt_d"] == "2019-03-01")
    ]

    # ── Age-Weighted Transition Matrix ────────────────────────────────
    st.subheader("Age-Weighted Transition Matrix")

    with st.spinner("Computing age-bucketed transition probabilities..."):
        age_probs = cached_age_transition_probs(df_enriched, bucket_size=6)

    if len(age_probs) > 0 and len(df_current_march_enriched) > 0:
        pool_matrix = compute_pool_transition_matrix(
            df_current_march_enriched, age_probs,
        )

        # Aggregate matrix as heatmap
        agg = pool_matrix["aggregate_matrix"]
        agg_pct = pool_matrix["aggregate_matrix_pct"]

        # Build matrix data for the Current row (the meaningful one)
        from src.portfolio_analytics import TRANSITION_STATES
        current_flows = agg.get("Current", {})
        current_pcts = agg_pct.get("Current", {})

        # Display as metric cards
        st.markdown("**Expected Next-Month Dollar Flows (from Current Pool)**")
        flow_cols = st.columns(5)
        for i, state in enumerate(TRANSITION_STATES):
            dollar_val = current_flows.get(state, 0)
            pct_val = current_pcts.get(state, 0)
            flow_cols[i].metric(
                f"→ {state}",
                f"${dollar_val:,.0f}",
                f"{pct_val * 100:.2f}%",
                delta_color="off",
            )

        # Full transition matrix heatmap (all from-states)
        matrix_rows = []
        for from_state in TRANSITION_STATES:
            if from_state in agg_pct:
                row = [agg_pct[from_state].get(to_state, 0)
                       for to_state in TRANSITION_STATES]
            else:
                row = [0.0] * len(TRANSITION_STATES)
            matrix_rows.append(row)

        # Text labels for heatmap cells
        text_matrix = [
            [f"{v * 100:.1f}%" for v in row] for row in matrix_rows
        ]

        fig_tm = go.Figure(go.Heatmap(
            z=matrix_rows,
            x=[s.replace("(0-30)", "(0-30d)").replace("(31-120)", "(31-120d)")
               for s in TRANSITION_STATES],
            y=[s.replace("(0-30)", "(0-30d)").replace("(31-120)", "(31-120d)")
               for s in TRANSITION_STATES],
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorscale=[
                [0, "#1a1a2e"],
                [0.5, "#e67e22"],
                [1, "#e74c3c"],
            ],
            colorbar=dict(
                title=dict(text="Prob", font=dict(color="#e2e8f0")),
                tickformat=".0%",
                tickfont=dict(color="#e2e8f0"),
            ),
            hovertemplate=(
                "From: %{y}<br>To: %{x}<br>Prob: %{z:.2%}<extra></extra>"
            ),
        ))
        fig_tm.update_layout(
            title="Transition Probability Matrix (Age-Weighted Average)",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            xaxis=dict(title="To Status", side="bottom"),
            yaxis=dict(title="From Status", autorange="reversed"),
            height=400,
        )
        st.plotly_chart(fig_tm, use_container_width=True, theme=None)

        # Age bucket breakdown table
        with st.expander("View age bucket breakdown"):
            bd = pool_matrix["breakdown_by_age"].copy()
            bd["upb"] = bd["upb"].apply(lambda x: f"${x:,.0f}")
            for c in [col for col in bd.columns if col.endswith("_$")]:
                bd[c] = bd[c].apply(lambda x: f"${x:,.0f}")
            for c in [col for col in bd.columns if col.endswith("_rate")]:
                bd[c] = bd[c].apply(lambda x: f"{x * 100:.2f}%")
            bd = bd.rename(columns={
                "age_bucket_label": "Age Bucket",
                "upb": "UPB",
                "to_current_$": "→ Current ($)",
                "to_delinquent_$": "→ Delinquent ($)",
                "to_late_31_120_$": "→ Late 31-120 ($)",
                "to_charged_off_$": "→ Charged Off ($)",
                "to_fully_paid_$": "→ Fully Paid ($)",
                "to_current_rate": "→ Current Rate",
                "to_delinquent_rate": "→ Delinquent Rate",
                "to_late_31_120_rate": "→ Late 31-120 Rate",
                "to_charged_off_rate": "→ Charged Off Rate",
                "to_fully_paid_rate": "→ Fully Paid Rate",
            })
            st.dataframe(bd, use_container_width=True, hide_index=True)

    else:
        st.info("Not enough data to compute age-weighted transition matrix.")

    # ── Default Timing Curve ──────────────────────────────────────────
    st.subheader("Default Timing Curve")

    default_timing = compute_default_timing(df_enriched)

    if len(default_timing) > 0:
        fig_dt = go.Figure()

        # Histogram bars (% of defaults)
        fig_dt.add_trace(go.Bar(
            x=default_timing["default_age_months"],
            y=default_timing["pct_of_defaults"] * 100,
            name="% of Defaults",
            marker_color="#E74C3C",
            hovertemplate=(
                "Age: %{x} months<br>"
                "% of Defaults: %{y:.1f}%<br>"
                "Loans: %{customdata[0]:,}"
                "<extra></extra>"
            ),
            customdata=default_timing[["loan_count"]].values,
        ))

        # Cumulative line on secondary y-axis
        fig_dt.add_trace(go.Scatter(
            x=default_timing["default_age_months"],
            y=default_timing["cumulative_pct"] * 100,
            name="Cumulative %",
            yaxis="y2",
            mode="lines",
            line=dict(color="#F39C12", width=2.5),
            hovertemplate=(
                "Age: %{x} months<br>"
                "Cumulative: %{y:.1f}%"
                "<extra></extra>"
            ),
        ))

        fig_dt.update_layout(
            title="Default Timing: Loan Age at Charge-Off",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            xaxis=dict(
                title="Loan Age (Months)",
                gridcolor="rgba(255,255,255,0.05)",
            ),
            yaxis=dict(
                title="% of Total Defaults",
                gridcolor="rgba(255,255,255,0.05)",
            ),
            yaxis2=dict(
                title=dict(text="Cumulative %", font=dict(color="#F39C12")),
                overlaying="y",
                side="right",
                showgrid=False,
                tickfont=dict(color="#F39C12"),
                range=[0, 105],
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="center", x=0.5,
            ),
            bargap=0.1,
        )
        st.plotly_chart(fig_dt, use_container_width=True, theme=None)

        # Optional: by grade if strata is grade
        if strata_col == "grade" and strata_value == "ALL":
            with st.expander("Default timing by grade"):
                dt_grade = compute_default_timing(df_enriched, group_col="grade")
                if len(dt_grade) > 0:
                    fig_dtg = go.Figure()
                    grade_colors = {
                        "A": "#2ECC71", "B": "#3498DB", "C": "#F39C12",
                        "D": "#E67E22", "E": "#E74C3C", "F": "#9B59B6",
                        "G": "#1ABC9C",
                    }
                    for grade_val in sorted(dt_grade["grade"].unique()):
                        gdf = dt_grade[dt_grade["grade"] == grade_val]
                        fig_dtg.add_trace(go.Scatter(
                            x=gdf["default_age_months"],
                            y=gdf["cumulative_pct"] * 100,
                            name=f"Grade {grade_val}",
                            mode="lines",
                            line=dict(
                                color=grade_colors.get(grade_val, "#95a5a6"),
                                width=2,
                            ),
                        ))
                    fig_dtg.update_layout(
                        title="Cumulative Default Timing by Grade",
                        paper_bgcolor="#0f172a",
                        plot_bgcolor="#0f172a",
                        font=dict(color="#e2e8f0"),
                        xaxis=dict(
                            title="Loan Age (Months)",
                            gridcolor="rgba(255,255,255,0.05)",
                        ),
                        yaxis=dict(
                            title="Cumulative % of Defaults",
                            gridcolor="rgba(255,255,255,0.05)",
                            range=[0, 105],
                        ),
                        legend=dict(
                            orientation="h", yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5,
                        ),
                    )
                    st.plotly_chart(fig_dtg, use_container_width=True, theme=None)
    else:
        st.info("No charged-off loans in this selection for default timing analysis.")

    # ── Loan Age Status Distribution ──────────────────────────────────
    st.subheader("Loan Age Status Distribution")

    bucket_toggle = st.toggle(
        "Monthly granularity", value=False, key="age_status_monthly"
    )
    age_bucket_size = 1 if bucket_toggle else 6

    age_status = compute_loan_age_status_matrix(df_enriched, bucket_size=age_bucket_size)

    if len(age_status) > 0:
        # Stacked bar chart
        fig_as = go.Figure()

        status_config = [
            ("pct_current", "Current", "#4A90D9"),
            ("pct_fully_paid", "Fully Paid", "#2ECC71"),
            ("pct_charged_off", "Charged Off", "#E74C3C"),
            ("pct_late_grace", "Late / Grace", "#F39C12"),
        ]

        for col, label, color in status_config:
            fig_as.add_trace(go.Bar(
                x=age_status["age_bucket"].astype(str),
                y=age_status[col] * 100,
                name=label,
                marker_color=color,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Age Bucket: %{x}<br>"
                    "%{y:.1f}%"
                    "<extra></extra>"
                ),
            ))

        fig_as.update_layout(
            barmode="stack",
            title="Status Distribution by Loan Age at Snapshot",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            xaxis=dict(
                title="Loan Age (Months)" if bucket_toggle else "Loan Age Bucket",
                gridcolor="rgba(255,255,255,0.05)",
                type="category",
            ),
            yaxis=dict(
                title="% of Loans",
                gridcolor="rgba(255,255,255,0.05)",
                range=[0, 105],
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="center", x=0.5,
            ),
        )
        st.plotly_chart(fig_as, use_container_width=True, theme=None)

        with st.expander("View age status data"):
            as_display = age_status.copy()
            as_display["total_upb"] = as_display["total_upb"].apply(
                lambda x: f"${x:,.0f}"
            )
            for c in ["pct_current", "pct_fully_paid", "pct_charged_off", "pct_late_grace"]:
                as_display[c] = as_display[c].apply(lambda x: f"{x * 100:.2f}%")
            as_display = as_display.rename(columns={
                "age_bucket": "Age Bucket",
                "total_loans": "Total Loans",
                "total_upb": "Total UPB",
                "pct_current": "% Current",
                "pct_fully_paid": "% Fully Paid",
                "pct_charged_off": "% Charged Off",
                "pct_late_grace": "% Late/Grace",
            })
            st.dataframe(as_display, use_container_width=True, hide_index=True)
    else:
        st.info("No data available for age status distribution.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Cash Flow Projection
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Cash Flow Projection & IRR")

    if len(df_current_march) == 0:
        st.warning(
            "No Current loans with March 2019 payment date in this selection. "
            "Cash flow projections require active loans."
        )
    else:
        # Compute historical assumptions & pool characteristics
        pool_assumptions = compute_pool_assumptions(df_filtered, df_current_march)
        pool_chars = compute_pool_characteristics(df_current_march)

        # Historical base assumptions (read-only display)
        st.markdown("#### Historical Base Assumptions")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("CDR", f"{pool_assumptions['cdr'] * 100:.2f}%")
        a2.metric("CPR", f"{pool_assumptions['cpr'] * 100:.2f}%")
        a3.metric("Loss Severity", f"{pool_assumptions['loss_severity'] * 100:.2f}%")
        a4.metric("Recovery Rate", f"{pool_assumptions['recovery_rate'] * 100:.2f}%")

        # Pool characteristics
        st.markdown("#### Pool Characteristics")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Total UPB", f"${pool_chars['total_upb']:,.0f}")
        p2.metric("WAC", f"{pool_chars['wac'] * 100:.2f}%")
        p3.metric("WAM", f"{pool_chars['wam']} months")
        p4.metric("Monthly Payment", f"${pool_chars['monthly_payment']:,.0f}")

        # User-adjustable projection inputs
        st.markdown("---")
        st.markdown("#### Projection Inputs")
        i1, i2, i3 = st.columns(3)
        with i1:
            t2_purchase_price_pct = st.number_input(
                "Purchase Price (%)",
                min_value=50.00,
                max_value=120.00,
                value=95.00,
                step=0.01,
                format="%.2f",
                key="t2_purchase_price",
            )
        with i2:
            t2_cdr_pct = st.number_input(
                "CDR (%)",
                min_value=0.00,
                max_value=100.00,
                value=round(pool_assumptions["cdr"] * 100, 2),
                step=0.01,
                format="%.2f",
                key="t2_cdr",
            )
        with i3:
            t2_cpr_pct = st.number_input(
                "CPR (%)",
                min_value=0.00,
                max_value=100.00,
                value=round(pool_assumptions["cpr"] * 100, 2),
                step=0.01,
                format="%.2f",
                key="t2_cpr",
            )

        # Convert percentage inputs to decimals
        t2_purchase_price = t2_purchase_price_pct / 100
        cdr = t2_cdr_pct / 100
        cpr = t2_cpr_pct / 100
        loss_severity = pool_assumptions["loss_severity"]

        # Project cash flows
        cf_df = project_cashflows(
            pool_chars, cdr, cpr, loss_severity, t2_purchase_price
        )
        irr = calculate_irr(cf_df, pool_chars, t2_purchase_price)

        # IRR display and price solver
        st.markdown("---")
        irr_col, price_col = st.columns(2)

        if np.isnan(irr) or np.isinf(irr):
            irr_col.metric("Projected IRR", "N/A")
        else:
            irr_col.metric("Projected IRR", f"{irr * 100:.2f}%")

        # Price solver
        with price_col:
            target_irr_pct = st.number_input(
                "Target IRR (%) for price solver",
                min_value=0.00,
                max_value=100.00,
                value=12.00,
                step=0.01,
                format="%.2f",
                key="t2_target_irr",
            )
            target_irr = target_irr_pct / 100
            solved = solve_price(
                pool_chars, target_irr, cdr, cpr, loss_severity
            )
            if solved is not None:
                st.metric(
                    f"Price for {target_irr_pct:.2f}% IRR",
                    f"{solved * 100:.2f}",
                )
            else:
                st.metric(f"Price for {target_irr_pct:.2f}% IRR", "No solution")

        # Monthly Cash Flows & Pool Balance (dual-axis chart)
        st.markdown("---")
        show_pct = st.checkbox("Show as % of total", key="cf_pct_toggle")

        # Monthly totals for hover percentages and normalization
        monthly_total = (
            cf_df["interest"]
            + cf_df["scheduled_principal"]
            + cf_df["prepayments"]
            + cf_df["recovery"]
        )

        # Component definitions: (column, label, color)
        components = [
            ("interest", "Interest", "#4A90D9"),
            ("scheduled_principal", "Scheduled Principal", "#E67E22"),
            ("prepayments", "Prepayments", "#2ECC71"),
            ("recovery", "Recoveries", "#9B59B6"),
        ]

        fig_cf = go.Figure()
        for col, label, color in components:
            raw_vals = cf_df[col]
            pct_vals = (raw_vals / monthly_total.replace(0, 1)) * 100

            if show_pct:
                y_vals = pct_vals
                hover = (
                    f"<b>{label}</b><br>"
                    "%{customdata[0]}<br>"
                    "%{y:.1f}% of total"
                    "<extra></extra>"
                )
                customdata = [[f"${v:,.0f}"] for v in raw_vals]
            else:
                y_vals = raw_vals
                hover = (
                    f"<b>{label}</b><br>"
                    "$%{y:,.0f}<br>"
                    "%{customdata[0]:.1f}% of total"
                    "<extra></extra>"
                )
                customdata = [[p] for p in pct_vals]

            fig_cf.add_trace(go.Bar(
                x=cf_df["date"],
                y=y_vals,
                name=label,
                marker_color=color,
                customdata=customdata,
                hovertemplate=hover,
            ))

        # Overlay ending balance line on secondary y-axis (skip in % mode)
        if not show_pct:
            fig_cf.add_trace(go.Scatter(
                x=cf_df["date"],
                y=cf_df["ending_balance"],
                name="Ending Balance",
                yaxis="y2",
                mode="lines",
                line=dict(color="#cbd5e1", width=2.5),
                hovertemplate="<b>Ending Balance</b><br>$%{y:,.0f}<extra></extra>",
            ))

        layout_kwargs = dict(
            barmode="stack",
            title="Monthly Cash Flows & Pool Balance",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            xaxis=dict(
                title="Date",
                gridcolor="rgba(255,255,255,0.05)",
            ),
            yaxis=dict(
                title="% of Total" if show_pct else "$",
                tickformat=".0f" if show_pct else "$,.0s",
                gridcolor="rgba(255,255,255,0.05)",
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="center", x=0.5,
            ),
        )

        if not show_pct:
            layout_kwargs["yaxis2"] = dict(
                title=dict(text="Ending Balance ($)", font=dict(color="#cbd5e1")),
                overlaying="y",
                side="right",
                showgrid=False,
                tickformat="$,.0s",
                tickfont=dict(color="#cbd5e1"),
            )

        fig_cf.update_layout(**layout_kwargs)
        st.plotly_chart(fig_cf, use_container_width=True, theme=None)

        # Expandable cash flow table
        with st.expander("View Full Cash Flow Table"):
            display_cf = cf_df.copy()
            display_cf["date"] = display_cf["date"].dt.strftime("%Y-%m-%d")
            dollar_cols = [
                "beginning_balance", "defaults", "loss", "recovery", "interest",
                "scheduled_principal", "prepayments", "total_principal",
                "ending_balance", "total_cashflow",
            ]
            for c in dollar_cols:
                display_cf[c] = display_cf[c].apply(lambda x: f"${x:,.2f}")
            display_cf = display_cf.rename(columns={
                "month": "Month",
                "date": "Date",
                "beginning_balance": "Beginning Balance",
                "defaults": "Defaults",
                "loss": "Losses",
                "recovery": "Recoveries",
                "interest": "Interest",
                "scheduled_principal": "Sched. Principal",
                "prepayments": "Prepayments",
                "total_principal": "Total Principal",
                "ending_balance": "Ending Balance",
                "total_cashflow": "Total Cash Flow",
            })
            st.dataframe(display_cf, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: Scenario Analysis
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Scenario Analysis")

    if len(df_current_march) == 0:
        st.warning(
            "No Current loans with March 2019 payment date in this selection. "
            "Scenario analysis requires active loans."
        )
    else:
        # Reuse assumptions from Tab 2 computations
        pool_assumptions_t3 = compute_pool_assumptions(df_filtered, df_current_march)
        pool_chars_t3 = compute_pool_characteristics(df_current_march)

        base_assumptions = {
            "cdr": pool_assumptions_t3["cdr"],
            "cpr": pool_assumptions_t3["cpr"],
            "loss_severity": pool_assumptions_t3["loss_severity"],
        }

        # Tab 3 controls: purchase price and stress/upside %
        s1, s2 = st.columns(2)
        with s1:
            t3_purchase_price_pct = st.number_input(
                "Purchase Price (%)",
                min_value=50.00,
                max_value=120.00,
                value=95.00,
                step=0.01,
                format="%.2f",
                key="t3_purchase_price",
            )
        with s2:
            stress_upside_pct_int = st.slider(
                "Stress / Upside %",
                min_value=5,
                max_value=50,
                value=15,
                step=1,
                key="t3_stress_upside",
            )

        t3_purchase_price = t3_purchase_price_pct / 100
        stress_upside_pct = stress_upside_pct_int / 100

        # Build scenarios
        scenarios = build_scenarios(
            base_assumptions,
            stress_pct=stress_upside_pct,
            upside_pct=stress_upside_pct,
        )

        # Show scenario assumption table
        st.markdown("#### Scenario Assumptions")
        assumption_rows = []
        for name, a in scenarios.items():
            assumption_rows.append({
                "Scenario": name,
                "CDR": f"{a['cdr'] * 100:.2f}%",
                "CPR": f"{a['cpr'] * 100:.2f}%",
                "Loss Severity": f"{a['loss_severity'] * 100:.2f}%",
            })
        st.dataframe(
            pd.DataFrame(assumption_rows),
            use_container_width=True,
            hide_index=True,
        )

        # Run comparison
        comparison = compare_scenarios(pool_chars_t3, scenarios, t3_purchase_price)

        # Format comparison table
        st.markdown("#### Scenario Comparison")
        comp_display = comparison.copy()
        comp_display["cdr"] = comp_display["cdr"].apply(lambda x: f"{x * 100:.2f}%")
        comp_display["cpr"] = comp_display["cpr"].apply(lambda x: f"{x * 100:.2f}%")
        comp_display["loss_severity"] = comp_display["loss_severity"].apply(
            lambda x: f"{x * 100:.2f}%"
        )
        comp_display["irr"] = comp_display["irr"].apply(
            lambda x: f"{x * 100:.2f}%" if not np.isnan(x) else "N/A"
        )
        for c in ["total_interest", "total_principal", "total_losses", "total_recoveries"]:
            comp_display[c] = comp_display[c].apply(lambda x: f"${x:,.0f}")
        comp_display["weighted_avg_life"] = comp_display["weighted_avg_life"].apply(
            lambda x: f"{x:.1f} yrs"
        )
        comp_display = comp_display.rename(columns={
            "scenario": "Scenario",
            "cdr": "CDR",
            "cpr": "CPR",
            "loss_severity": "Loss Severity",
            "irr": "IRR",
            "total_interest": "Total Interest",
            "total_principal": "Total Principal",
            "total_losses": "Total Losses",
            "total_recoveries": "Total Recoveries",
            "weighted_avg_life": "WAL",
        })
        st.dataframe(comp_display, use_container_width=True, hide_index=True)

        # IRR Sensitivity Heatmap: CDR vs CPR
        st.markdown("#### IRR Sensitivity: CDR vs CPR")

        base_cdr = base_assumptions["cdr"]
        base_cpr = base_assumptions["cpr"]

        # Build grid: 0% to ~2× base, 7 values each
        cdr_grid = tuple(sorted(set([
            0.0,
            round(base_cdr * 0.5, 4),
            round(base_cdr, 4),
            round(base_cdr * 1.5, 4),
            round(base_cdr * 2.0, 4),
            round(base_cdr * 2.5, 4),
            round(base_cdr * 3.0, 4),
        ])))
        cpr_grid = tuple(sorted(set([
            0.0,
            round(base_cpr * 0.5, 4),
            round(base_cpr, 4),
            round(base_cpr * 1.5, 4),
            round(base_cpr * 2.0, 4),
            round(base_cpr * 3.0, 4),
            round(base_cpr * 4.0, 4),
        ])))

        with st.spinner("Computing IRR sensitivity grid..."):
            irr_matrix = compute_irr_grid(
                pool_chars_t3, cdr_grid, cpr_grid,
                base_assumptions["loss_severity"], t3_purchase_price,
            )

        # Format labels and cell text
        cdr_labels = [f"{v * 100:.2f}%" for v in cdr_grid]
        cpr_labels = [f"{v * 100:.2f}%" for v in cpr_grid]
        text_matrix = [
            [f"{v * 100:.1f}%" if v is not None else "N/A" for v in row]
            for row in irr_matrix
        ]

        fig_hm = go.Figure()

        fig_hm.add_trace(go.Heatmap(
            z=irr_matrix,
            x=cpr_labels,
            y=cdr_labels,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=12),
            colorscale=[
                [0, "#C0392B"],
                [0.5, "#F39C12"],
                [1, "#2ECC71"],
            ],
            colorbar=dict(
                title=dict(text="IRR", font=dict(color="#e2e8f0")),
                tickformat=".1%",
                tickfont=dict(color="#e2e8f0"),
            ),
            hovertemplate=(
                "CDR: %{y}<br>"
                "CPR: %{x}<br>"
                "IRR: %{z:.2%}"
                "<extra></extra>"
            ),
        ))

        # Highlight base case cell
        base_cdr_label = f"{base_cdr * 100:.2f}%"
        base_cpr_label = f"{base_cpr * 100:.2f}%"
        fig_hm.add_trace(go.Scatter(
            x=[base_cpr_label],
            y=[base_cdr_label],
            mode="markers",
            name="Base Case",
            marker=dict(
                symbol="square",
                size=14,
                color="rgba(0,0,0,0)",
                line=dict(color="white", width=2),
            ),
        ))

        fig_hm.update_layout(
            title="IRR Sensitivity: CDR vs CPR",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            xaxis=dict(title="CPR Assumption", type="category"),
            yaxis=dict(title="CDR Assumption", type="category"),
        )
        st.plotly_chart(fig_hm, use_container_width=True, theme=None)

        # Balance over time — grouped bar chart by scenario
        st.markdown("#### Projected Balance by Scenario")
        scenario_colors = {
            "Base": "#4A90D9",
            "Stress": "#E74C3C",
            "Upside": "#2ECC71",
        }
        fig_bal = go.Figure()
        for name, assumptions in scenarios.items():
            cf = project_cashflows(
                pool_chars_t3,
                assumptions["cdr"],
                assumptions["cpr"],
                assumptions["loss_severity"],
                t3_purchase_price,
            )
            fig_bal.add_trace(go.Bar(
                x=cf["date"],
                y=cf["ending_balance"],
                name=name,
                marker_color=scenario_colors[name],
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "%{x|%b %Y}<br>"
                    "$%{y:,.2s}"
                    "<extra></extra>"
                ),
            ))

        fig_bal.update_layout(
            barmode="group",
            bargap=0.15,
            bargroupgap=0.05,
            title="Pool Balance Over Time by Scenario",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            xaxis=dict(
                title="Date",
                gridcolor="rgba(255,255,255,0.05)",
            ),
            yaxis=dict(
                title="Ending Balance ($)",
                tickformat="$,.0s",
                gridcolor="rgba(255,255,255,0.05)",
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="center", x=0.5,
            ),
        )
        st.plotly_chart(fig_bal, use_container_width=True, theme=None)
