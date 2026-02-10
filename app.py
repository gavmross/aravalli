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
    calculate_irr,
    adjust_prepayment_rates,
    build_pool_state,
    project_cashflows_transition,
    solve_price_transition,
)
from src.portfolio_analytics import (
    calculate_credit_metrics,
    calculate_performance_metrics,
    calculate_transition_matrix,
    reconstruct_loan_timeline,
    compute_age_transition_probabilities,
    compute_default_timing,
    compute_loan_age_status_matrix,
    TRANSITION_STATES_7,
)
from src.scenario_analysis import (
    build_scenarios_transition,
    compare_scenarios_transition,
)
from src.transition_viz import render_sankey_diagram

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Lending Club Portfolio Analyzer", layout="wide")
st.title("Lending Club Loan Portfolio Investment Analysis")

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_data() -> pd.DataFrame:
    """Load loans.db and run calc_amort to add engineered columns."""
    conn = sqlite3.connect("data/loans.db")
    df = pd.read_sql("SELECT * FROM loans", conn)
    conn.close()
    df = calc_amort(df, verbose=False)
    # Downcast float64 → float32 to halve memory for numeric columns
    float64_cols = df.select_dtypes(include=['float64']).columns
    df[float64_cols] = df[float64_cols].astype('float32')
    # Downcast int64 → int32
    int64_cols = df.select_dtypes(include=['int64']).columns
    df[int64_cols] = df[int64_cols].astype('int32')
    # Convert low-cardinality object columns to category dtype
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < 100:
            df[col] = df[col].astype('category')
    return df


@st.cache_resource
def cached_reconstruct_timeline(_df: pd.DataFrame,
                                cache_key: str = '') -> pd.DataFrame:
    """Reconstruct loan timelines for backsolve analysis."""
    return reconstruct_loan_timeline(_df)


@st.cache_resource
def cached_age_transition_probs(_df: pd.DataFrame, bucket_size: int,
                                states: str = '5state',
                                cache_key: str = '') -> pd.DataFrame:
    """Compute age-bucketed transition probabilities."""
    return compute_age_transition_probabilities(
        _df, bucket_size=bucket_size, states=states)


# ---------------------------------------------------------------------------
# Cache versioning — bump this when cached data structures change.
# Clears all stale caches automatically on the first run after a code update.
# ---------------------------------------------------------------------------
_CACHE_VERSION = 7
if st.session_state.get("_cache_version") != _CACHE_VERSION:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state["_cache_version"] = _CACHE_VERSION

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

    st.markdown("---")
    st.caption(
        "State-transition model: defaults flow through a 5-month "
        "delinquency pipeline with age-specific transition probabilities."
    )

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
    df_filtered = df[df[strata_col] == filter_val]
else:
    df_filtered = df

# Cache key for filter-dependent cached functions
_filter_key = f"{strata_col}_{strata_value}"

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
active_statuses = ['Current', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
non_current_active = ['In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
df_active_march = df_filtered[
    ((df_filtered["loan_status"] == "Current") & (df_filtered["last_pymnt_d"] == "2019-03-01"))
    | (df_filtered["loan_status"].isin(non_current_active))
]

# ---------------------------------------------------------------------------
# Shared enriched timeline (computed once, used across all tabs)
# ---------------------------------------------------------------------------
df_enriched = cached_reconstruct_timeline(df_filtered, cache_key=_filter_key)

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
    pm_pct_cols = [
        "pct_active", "pct_current", "pct_fully_paid", "pct_charged_off",
        "pct_defaulted_count", "pct_defaulted_upb",
        "pool_cpr", "pct_prepaid", "loss_severity", "recovery_rate",
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
        "pool_cpr": "CPR (Active Loans)",
        "pct_prepaid": "UPB Prepaid of OUPB",
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

    # Active loans: Current with March 2019 payment + all delinquent
    # Note: no .copy() — the enriched df has mixed dtypes that cause
    # pandas consolidation errors on copy; this slice is read-only
    df_active_march_enriched = df_enriched[
        ((df_enriched["loan_status"] == "Current") & (df_enriched["last_pymnt_d"] == "2019-03-01"))
        | (df_enriched["loan_status"].isin(non_current_active))
    ]

    # ── Age-Weighted Transition Matrix (7-State Monthly) ──────────────
    st.subheader("Age-Weighted Transition Matrix")

    with st.spinner("Computing monthly transition probabilities..."):
        age_probs = cached_age_transition_probs(
            df_enriched, bucket_size=1, states='7state',
            cache_key=_filter_key)

    if len(age_probs) > 0 and len(df_active_march_enriched) > 0:
        state_list = TRANSITION_STATES_7
        pct_cols_7 = [
            'to_current_pct', 'to_delinquent_0_30_pct',
            'to_late_1_pct', 'to_late_2_pct', 'to_late_3_pct',
            'to_charged_off_pct', 'to_fully_paid_pct',
        ]

        # Build weighted-average heatmap across all ages
        matrix_rows = []
        for from_state in state_list:
            state_rows = age_probs[age_probs['from_status'] == from_state]
            if len(state_rows) > 0:
                total_obs = state_rows['observation_count'].sum()
                row = []
                for pct_col in pct_cols_7:
                    if total_obs > 0:
                        wavg = (
                            (state_rows[pct_col] * state_rows['observation_count'])
                            .sum() / total_obs
                        )
                    else:
                        wavg = 0.0
                    row.append(wavg)
            else:
                row = [0.0] * len(state_list)
            matrix_rows.append(row)

        text_matrix = [
            [f"{v * 100:.1f}%" for v in row] for row in matrix_rows
        ]

        fig_tm = go.Figure(go.Heatmap(
            z=matrix_rows,
            x=state_list,
            y=state_list,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=10),
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
            title="7-State Transition Probability Matrix (Observation-Weighted Average)",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            xaxis=dict(title="To Status", side="bottom"),
            yaxis=dict(title="From Status", autorange="reversed"),
            height=450,
        )
        st.plotly_chart(fig_tm, use_container_width=True, theme=None)

        # Show raw probabilities table
        with st.expander("View monthly transition probabilities"):
            ap_display = age_probs.copy()
            for c in pct_cols_7:
                ap_display[c] = ap_display[c].apply(lambda x: f"{x * 100:.2f}%")
            st.dataframe(ap_display, use_container_width=True, hide_index=True)

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

    age_status = compute_loan_age_status_matrix(df_enriched, bucket_size=1)

    if len(age_status) > 0:
        # Compute UPB by status for each loan age
        age_status["upb_current"] = age_status["total_upb"] * age_status["pct_current"]
        age_status["upb_fully_paid"] = age_status["total_upb"] * age_status["pct_fully_paid"]
        age_status["upb_charged_off"] = age_status["total_upb"] * age_status["pct_charged_off"]
        age_status["upb_late_grace"] = age_status["total_upb"] * age_status["pct_late_grace"]

        fig_as = go.Figure()

        status_config = [
            ("upb_current", "Current", "#4A90D9"),
            ("upb_fully_paid", "Fully Paid", "#2ECC71"),
            ("upb_charged_off", "Charged Off", "#E74C3C"),
            ("upb_late_grace", "Late / Grace", "#F39C12"),
        ]

        for col, label, color in status_config:
            fig_as.add_trace(go.Bar(
                x=age_status["age_bucket"].astype(str),
                y=age_status[col],
                name=label,
                marker_color=color,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Loan Age: %{x} months<br>"
                    "$%{y:,.0f}"
                    "<extra></extra>"
                ),
            ))

        fig_as.update_layout(
            barmode="stack",
            title="UPB by Loan Status at Each Loan Age",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
            xaxis=dict(
                title="Loan Age (Months)",
                gridcolor="rgba(255,255,255,0.05)",
                type="category",
            ),
            yaxis=dict(
                title="UPB ($)",
                tickformat="$,.0s",
                gridcolor="rgba(255,255,255,0.05)",
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="center", x=0.5,
            ),
        )
        st.plotly_chart(fig_as, use_container_width=True, theme=None)

        with st.expander("View age status data"):
            as_display = age_status[["age_bucket", "total_loans", "total_upb",
                                     "upb_current", "upb_fully_paid",
                                     "upb_charged_off", "upb_late_grace"]].copy()
            for c in ["total_upb", "upb_current", "upb_fully_paid",
                       "upb_charged_off", "upb_late_grace"]:
                as_display[c] = as_display[c].apply(lambda x: f"${x:,.0f}")
            as_display = as_display.rename(columns={
                "age_bucket": "Loan Age (Months)",
                "total_loans": "Total Loans",
                "total_upb": "Total UPB",
                "upb_current": "Current",
                "upb_fully_paid": "Fully Paid",
                "upb_charged_off": "Charged Off",
                "upb_late_grace": "Late/Grace",
            })
            st.dataframe(as_display, use_container_width=True, hide_index=True)
    else:
        st.info("No data available for age status distribution.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Cash Flow Projection
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Cash Flow Projection & IRR")

    if len(df_active_march) == 0:
        st.warning(
            "No active loans with March 2019 payment date in this selection. "
            "Cash flow projections require active loans."
        )
    else:
        # Compute historical assumptions & pool characteristics
        pool_assumptions = compute_pool_assumptions(df_enriched, df_active_march)
        pool_chars = compute_pool_characteristics(df_active_march)

        # Historical base assumptions (read-only display)
        st.markdown("#### Historical Base Assumptions")
        a1, a2, a3 = st.columns(3)
        a1.metric("CDR (Conditional)", f"{pool_assumptions['cdr'] * 100:.2f}%")
        a2.metric("Avg MDR", f"{pool_assumptions['avg_mdr'] * 100:.4f}%")
        a3.metric("CPR", f"{pool_assumptions['cpr'] * 100:.2f}%")

        b1, b2, b3 = st.columns(3)
        b1.metric("Loss Severity", f"{pool_assumptions['loss_severity'] * 100:.2f}%")
        b2.metric("Recovery Rate", f"{pool_assumptions['recovery_rate'] * 100:.2f}%")
        b3.metric("Cumulative Default Rate", f"{pool_assumptions['cumulative_default_rate'] * 100:.2f}%")
        st.caption("Cumulative Default Rate is the raw lifetime rate (reference only). CDR above is the conditional annualized rate used in projections.")

        # Pool characteristics
        st.markdown("#### Pool Characteristics")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Total UPB", f"${pool_chars['total_upb']:,.0f}")
        p2.metric("WAC", f"{pool_chars['wac'] * 100:.2f}%")
        p3.metric("WAM", f"{pool_chars['wam']} months")
        p4.metric("Monthly Payment", f"${pool_chars['monthly_payment']:,.0f}")

        st.markdown("---")

        # ── State-Transition Projection ──
        st.markdown("#### State-Transition Cash Flow Projection")
        st.caption(
            "Defaults flow through a 5-month delinquency pipeline "
            "(Current → Delinquent → Late_1 → Late_2 → Late_3 → Default). "
            "Transition probabilities are age-specific and empirically derived."
        )

        # Purchase price input
        t2_purchase_price_pct = st.number_input(
            "Purchase Price (%)",
            min_value=50.00,
            max_value=120.00,
            value=95.00,
            step=0.01,
            format="%.2f",
            key="t2_purchase_price",
        )
        t2_purchase_price = t2_purchase_price_pct / 100
        loss_severity = pool_assumptions["loss_severity"]
        recovery_rate = pool_assumptions["recovery_rate"]

        # Build pool state and age probs
        with st.spinner("Computing transition probabilities..."):
            age_probs_raw = cached_age_transition_probs(
                df_enriched, bucket_size=1, states='7state',
                cache_key=_filter_key)
            # Replace empirical Current→Fully Paid with CPR-derived SMM
            age_probs_7 = adjust_prepayment_rates(
                age_probs_raw, pool_assumptions['cpr'])

        # Enrich active loans for pool state (needs timeline columns)
        df_active_march_enriched_t2 = df_enriched[
            ((df_enriched["loan_status"] == "Current") & (df_enriched["last_pymnt_d"] == "2019-03-01"))
            | (df_enriched["loan_status"].isin(non_current_active))
        ]

        pool_st = build_pool_state(df_active_march_enriched_t2)

        num_months = pool_chars['wam']

        with st.spinner("Projecting cash flows (state-transition)..."):
            cf_df = project_cashflows_transition(
                pool_st, age_probs_7, loss_severity, recovery_rate,
                pool_chars, num_months,
            )

        irr = calculate_irr(cf_df, pool_chars, t2_purchase_price)

        # IRR display and price solver
        irr_col, price_col = st.columns(2)

        if np.isnan(irr) or np.isinf(irr):
            irr_col.metric("Projected IRR", "N/A")
        else:
            irr_col.metric("Projected IRR", f"{irr * 100:.2f}%")

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
            solved = solve_price_transition(
                pool_st, age_probs_7, loss_severity, recovery_rate,
                pool_chars, num_months, target_irr,
            )
            if solved is not None:
                st.metric(
                    f"Price for {target_irr_pct:.2f}% IRR",
                    f"{solved * 100:.2f}",
                )
            else:
                st.metric(f"Price for {target_irr_pct:.2f}% IRR", "No solution")

        # UPB by state stacked area chart
        if len(cf_df) > 0:
            st.markdown("---")
            st.markdown("#### Pool Balance by State")

            state_cols_chart = [
                ("current_upb", "Current", "#4A90D9"),
                ("delinquent_upb", "Delinquent (0-30)", "#F39C12"),
                ("late_1_upb", "Late_1", "#E67E22"),
                ("late_2_upb", "Late_2", "#E74C3C"),
                ("late_3_upb", "Late_3", "#9B59B6"),
            ]

            fig_state = go.Figure()
            for col, label, color in state_cols_chart:
                fig_state.add_trace(go.Scatter(
                    x=cf_df["date"],
                    y=cf_df[col],
                    name=label,
                    mode="lines",
                    stackgroup="one",
                    line=dict(width=0.5, color=color),
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        "%{x|%b %Y}<br>"
                        "$%{y:,.0f}"
                        "<extra></extra>"
                    ),
                ))
            fig_state.update_layout(
                title="Pool Balance by State Over Time",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                font=dict(color="#e2e8f0"),
                xaxis=dict(
                    title="Date",
                    gridcolor="rgba(255,255,255,0.05)",
                ),
                yaxis=dict(
                    title="UPB ($)",
                    tickformat="$,.0s",
                    gridcolor="rgba(255,255,255,0.05)",
                ),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5,
                ),
            )
            st.plotly_chart(fig_state, use_container_width=True, theme=None)

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

    if len(df_active_march) == 0:
        st.warning(
            "No active loans with March 2019 payment date in this selection. "
            "Scenario analysis requires active loans."
        )
    else:
        # Reuse assumptions from Tab 2 computations
        pool_assumptions_t3 = compute_pool_assumptions(df_enriched, df_active_march)
        pool_chars_t3 = compute_pool_characteristics(df_active_march)

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

        # ── State-Transition Scenarios ──
        st.markdown("#### State-Transition Scenarios")
        st.caption(
            "Stress/upside shifts are applied to transition probabilities: "
            "stress increases delinquency rates and decreases cure rates; "
            "upside does the opposite. Loss severity is FIXED."
        )

        with st.spinner("Computing transition probabilities..."):
            age_probs_t3_raw = cached_age_transition_probs(
                df_enriched, bucket_size=1, states='7state',
                cache_key=_filter_key)
            # Replace empirical Current→Fully Paid with CPR-derived SMM
            age_probs_t3 = adjust_prepayment_rates(
                age_probs_t3_raw, pool_assumptions_t3['cpr'])

        scenario_probs = build_scenarios_transition(
            age_probs_t3,
            stress_pct=stress_upside_pct,
            upside_pct=stress_upside_pct,
        )

        # Build pool state
        df_active_march_enriched_t3 = df_enriched[
            ((df_enriched["loan_status"] == "Current") & (df_enriched["last_pymnt_d"] == "2019-03-01"))
            | (df_enriched["loan_status"].isin(non_current_active))
        ]
        pool_st_t3 = build_pool_state(df_active_march_enriched_t3)
        num_months_t3 = pool_chars_t3['wam']
        loss_sev_t3 = pool_assumptions_t3["loss_severity"]
        rec_rate_t3 = pool_assumptions_t3["recovery_rate"]

        with st.spinner("Running scenario projections..."):
            comparison = compare_scenarios_transition(
                pool_st_t3, scenario_probs,
                loss_sev_t3, rec_rate_t3,
                pool_chars_t3, num_months_t3, t3_purchase_price,
            )

        st.markdown("#### Scenario Comparison")
        comp_display = comparison.copy()
        comp_display["loss_severity"] = comp_display["loss_severity"].apply(
            lambda x: f"{x * 100:.2f}%"
        )
        comp_display["irr"] = comp_display["irr"].apply(
            lambda x: f"{x * 100:.2f}%" if not np.isnan(x) else "N/A"
        )
        for c in ["total_interest", "total_principal", "total_losses",
                   "total_recoveries", "total_defaults"]:
            if c in comp_display.columns:
                comp_display[c] = comp_display[c].apply(lambda x: f"${x:,.0f}")
        comp_display["weighted_avg_life"] = comp_display["weighted_avg_life"].apply(
            lambda x: f"{x:.1f} yrs"
        )
        comp_display = comp_display.rename(columns={
            "scenario": "Scenario",
            "loss_severity": "Loss Severity",
            "irr": "IRR",
            "total_interest": "Total Interest",
            "total_principal": "Total Principal",
            "total_losses": "Total Losses",
            "total_recoveries": "Total Recoveries",
            "total_defaults": "Total Defaults",
            "weighted_avg_life": "WAL",
        })
        st.dataframe(comp_display, use_container_width=True, hide_index=True)

        # Balance by scenario chart
        st.markdown("#### Projected Balance by Scenario")
        scenario_colors = {
            "Base": "#4A90D9",
            "Stress": "#E74C3C",
            "Upside": "#2ECC71",
        }
        fig_bal = go.Figure()
        for name, probs_df in scenario_probs.items():
            cf = project_cashflows_transition(
                pool_st_t3, probs_df,
                loss_sev_t3, rec_rate_t3,
                pool_chars_t3, num_months_t3,
            )
            fig_bal.add_trace(go.Scatter(
                x=cf["date"],
                y=cf["ending_balance"],
                name=name,
                mode="lines",
                line=dict(color=scenario_colors.get(name, "#95a5a6"), width=2),
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "%{x|%b %Y}<br>"
                    "$%{y:,.0f}"
                    "<extra></extra>"
                ),
            ))
        fig_bal.update_layout(
            title="Pool Balance Over Time by Scenario (State-Transition)",
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
