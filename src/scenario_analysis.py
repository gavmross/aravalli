"""
Scenario analysis for base/stress/upside comparison.

Functions:
    compute_base_assumptions — extract base-case CDR, CPR, loss severity
    build_scenarios — apply multiplicative shifts for stress/upside
    compare_scenarios — run projections and IRR for each scenario
    build_scenarios_transition — apply stress/upside shifts to transition probs
    compare_scenarios_transition — run transition projections for each scenario
"""

import logging

import numpy as np
import pandas as pd

from src.cashflow_engine import (
    compute_pool_assumptions,
    project_cashflows,
    calculate_irr,
    project_cashflows_transition,
)

logger = logging.getLogger(__name__)


def compute_vintage_percentiles(
    df_all: pd.DataFrame,
    df_active: pd.DataFrame,
    pool_assumptions: dict,
    min_loans_cdr: int = 1000,
    min_loans_cpr: int = 1000,
) -> dict:
    """
    Compute per-vintage CDR and CPR, then extract percentiles for scenarios.

    For each quarterly vintage with >= min_loans total loans, computes CDR
    and CPR via compute_pool_assumptions(). Takes unweighted P25/P50/P75
    percentiles across qualifying vintages and maps them to scenarios:
      - Base:   Pool-level empirical CDR and CPR (from pool_assumptions)
      - Stress: P75 CDR, P25 CPR
      - Upside: P25 CDR, P75 CPR

    Parameters
    ----------
    df_all : pd.DataFrame
        All loans in the filtered strata (enriched with reconstruct_loan_timeline
        and calc_amort columns). Used for CDR and loss severity.
    df_active : pd.DataFrame
        Active loans (Current March 2019 + delinquent) with calc_amort columns.
        Used for CPR computation.
    pool_assumptions : dict
        Output of compute_pool_assumptions(). Provides pool-level empirical
        CDR and CPR for the base case.
    min_loans_cdr : int
        Minimum number of total loans for a vintage to qualify for CDR (default 1000).
    min_loans_cpr : int
        Minimum number of qualifying loans for CPR (default 1000).

    Returns
    -------
    dict
        Keys: 'fallback' (bool), 'vintage_cdrs' (list), 'vintage_cprs' (list),
        'n_cdr_vintages' (int), 'n_cpr_vintages' (int),
        'percentiles' (dict | None), 'scenarios' (dict | None),
        'vintage_data' (pd.DataFrame).
    """
    vintage_col = 'issue_quarter'
    vintages = df_all[vintage_col].dropna().unique()

    vintage_records = []
    for v in vintages:
        v_all = df_all[df_all[vintage_col] == v]
        if len(v_all) < min(min_loans_cdr, min_loans_cpr):
            continue

        v_active = df_active[df_active[vintage_col] == v]

        try:
            assumptions = compute_pool_assumptions(v_all, v_active)
        except Exception:
            logger.warning("compute_pool_assumptions failed for vintage %s", v)
            continue

        vintage_records.append({
            'vintage': v,
            'cdr': assumptions['cdr'],
            'cpr': assumptions['cpr'],
            'loan_count': len(v_all),
            'active_loan_count': len(v_active),
        })

    vintage_data = pd.DataFrame(vintage_records)

    # Separate CDR and CPR qualifying vintages
    # Only vintages with sufficient active (outstanding) loans are relevant
    # for forward-looking scenario analysis
    if len(vintage_data) > 0:
        cdr_qualifying = vintage_data[
            vintage_data['active_loan_count'] >= min_loans_cdr
        ]
        cpr_qualifying = vintage_data[
            (vintage_data['active_loan_count'] >= min_loans_cpr)
            & vintage_data['cpr'].notna()
        ]
    else:
        cdr_qualifying = vintage_data
        cpr_qualifying = vintage_data

    n_cdr_vintages = len(cdr_qualifying)
    n_cpr_vintages = len(cpr_qualifying)
    vintage_cdrs = cdr_qualifying['cdr'].tolist() if n_cdr_vintages > 0 else []
    vintage_cprs = cpr_qualifying['cpr'].tolist() if n_cpr_vintages > 0 else []

    if n_cdr_vintages < 3 or n_cpr_vintages < 3:
        return {
            'fallback': True,
            'vintage_cdrs': vintage_cdrs,
            'vintage_cprs': vintage_cprs,
            'n_cdr_vintages': n_cdr_vintages,
            'n_cpr_vintages': n_cpr_vintages,
            'percentiles': None,
            'scenarios': None,
            'vintage_data': vintage_data,
        }

    # CDR percentiles
    cdr_values = np.array(vintage_cdrs)
    cdr_p25, cdr_p50, cdr_p75 = np.percentile(cdr_values, [25, 50, 75])

    # CPR percentiles
    cpr_values = np.array(vintage_cprs)
    cpr_p25, cpr_p50, cpr_p75 = np.percentile(cpr_values, [25, 50, 75])

    pool_cdr = pool_assumptions['cdr']
    pool_cpr = pool_assumptions['cpr']

    percentiles = {
        'cdr_p25': float(cdr_p25),
        'cdr_p50': float(cdr_p50),
        'cdr_p75': float(cdr_p75),
        'cpr_p25': float(cpr_p25),
        'cpr_p50': float(cpr_p50),
        'cpr_p75': float(cpr_p75),
    }

    scenarios = {
        'Base': {'cdr': float(pool_cdr), 'cpr': float(pool_cpr)},
        'Stress': {'cdr': float(cdr_p75), 'cpr': float(cpr_p25)},
        'Upside': {'cdr': float(cdr_p25), 'cpr': float(cpr_p75)},
    }

    return {
        'fallback': False,
        'vintage_cdrs': vintage_cdrs,
        'vintage_cprs': vintage_cprs,
        'n_cdr_vintages': n_cdr_vintages,
        'n_cpr_vintages': n_cpr_vintages,
        'percentiles': percentiles,
        'scenarios': scenarios,
        'vintage_data': vintage_data,
    }


def compute_base_assumptions(df_all: pd.DataFrame,
                             df_active: pd.DataFrame) -> dict:
    """
    Wrapper that calls compute_pool_assumptions() and returns the base case.

    Parameters
    ----------
    df_all : pd.DataFrame
        All loans in the filtered strata (all statuses).
    df_active : pd.DataFrame
        Active loans (Current + In Grace + Late) with last_pymnt_d == 2019-03-01,
        with calc_amort columns.

    Returns
    -------
    dict with keys: 'cdr', 'cpr', 'loss_severity'
    """
    assumptions = compute_pool_assumptions(df_all, df_active)
    return {
        'cdr': assumptions['cdr'],
        'cpr': assumptions['cpr'],
        'loss_severity': assumptions['loss_severity'],
    }


def build_scenarios(base_assumptions: dict,
                    stress_pct: float = 0.15,
                    upside_pct: float = 0.15) -> dict:
    """
    Build three scenarios from base assumptions using multiplicative adjustments.

    - Base: CDR and CPR as-is
    - Stress: CDR × (1 + stress_pct), CPR × (1 - stress_pct)
    - Upside: CDR × (1 - upside_pct), CPR × (1 + upside_pct)
    - Loss severity: FIXED across all three scenarios

    Parameters
    ----------
    base_assumptions : dict
        Must contain 'cdr', 'cpr', 'loss_severity'.
    stress_pct : float
        Multiplicative shift for stress scenario (default 0.15 = 15%).
    upside_pct : float
        Multiplicative shift for upside scenario (default 0.15 = 15%).

    Returns
    -------
    dict mapping scenario name to assumption dict.
    """
    cdr = base_assumptions['cdr']
    cpr = base_assumptions['cpr']
    loss_severity = base_assumptions['loss_severity']

    return {
        'Base': {
            'cdr': cdr,
            'cpr': cpr,
            'loss_severity': loss_severity,
        },
        'Stress': {
            'cdr': cdr * (1 + stress_pct),
            'cpr': cpr * (1 - stress_pct),
            'loss_severity': loss_severity,
        },
        'Upside': {
            'cdr': cdr * (1 - upside_pct),
            'cpr': cpr * (1 + upside_pct),
            'loss_severity': loss_severity,
        },
    }


def compare_scenarios(pool_chars: dict,
                      scenarios: dict,
                      purchase_price: float) -> pd.DataFrame:
    """
    Run project_cashflows and calculate_irr for each scenario.

    Parameters
    ----------
    pool_chars : dict
        Output of compute_pool_characteristics().
    scenarios : dict
        Output of build_scenarios(). Maps scenario name to assumption dict.
    purchase_price : float
        Purchase price as fraction of UPB (e.g., 0.95).

    Returns
    -------
    pd.DataFrame with columns: scenario, cdr, cpr, loss_severity, irr,
        total_interest, total_principal, total_losses, total_recoveries,
        weighted_avg_life
    """
    results = []

    for name, assumptions in scenarios.items():
        cdr = assumptions['cdr']
        cpr = assumptions['cpr']
        loss_severity = assumptions['loss_severity']

        cf_df = project_cashflows(pool_chars, cdr, cpr, loss_severity, purchase_price)
        irr = calculate_irr(cf_df, pool_chars, purchase_price)

        total_interest = cf_df['interest'].sum()
        total_principal = cf_df['total_principal'].sum()
        total_losses = cf_df['loss'].sum()
        total_recoveries = cf_df['recovery'].sum()

        # Weighted average life = sum(month × total_principal) / sum(total_principal)
        # Expressed in years (divide by 12)
        if total_principal > 0:
            wal = (cf_df['month'] * cf_df['total_principal']).sum() / total_principal / 12
        else:
            wal = 0.0

        results.append({
            'scenario': name,
            'cdr': cdr,
            'cpr': cpr,
            'loss_severity': loss_severity,
            'irr': irr,
            'total_interest': round(total_interest, 2),
            'total_principal': round(total_principal, 2),
            'total_losses': round(total_losses, 2),
            'total_recoveries': round(total_recoveries, 2),
            'weighted_avg_life': round(wal, 2),
        })

    return pd.DataFrame(results)


# ===========================================================================
# State-Transition Scenario Analysis
# ===========================================================================

def build_scenarios_transition(
    base_probs: pd.DataFrame,
    stress_pct: float = 0.15,
    upside_pct: float = 0.15,
) -> dict:
    """
    Build three scenario probability sets from base transition probabilities.

    Stress logic:
    - Current→Delinquent × (1 + stress_pct) — more delinquency
    - Current→Fully Paid × (1 - stress_pct) — less prepayment
    - All cure rates (→Current from non-Current) × (1 - stress_pct)
    - Late_3→Charged Off NOT directly stressed (increases via re-normalization)
    - After multipliers: clamp to [0, 1], then adjust residual prob to make row sum to 1

    Upside logic: opposite multipliers.

    Loss severity is FIXED — only transition probs change.

    Parameters
    ----------
    base_probs : pd.DataFrame
        Output of compute_age_transition_probabilities(states='7state').
    stress_pct : float
        Multiplicative shift for stress scenario (default 0.15).
    upside_pct : float
        Multiplicative shift for upside scenario (default 0.15).

    Returns
    -------
    dict
        {'Base': base_probs, 'Stress': stressed_probs, 'Upside': upside_probs}
    """
    pct_cols = [c for c in base_probs.columns if c.endswith('_pct')]

    def _apply_shift(probs_df: pd.DataFrame, shift_sign: float,
                     pct: float) -> pd.DataFrame:
        """Apply multiplicative shift to transition probabilities.

        shift_sign: +1 for stress, -1 for upside.
        """
        result = probs_df.copy()

        for idx, row in result.iterrows():
            from_status = row['from_status']

            if from_status in ('Charged Off', 'Fully Paid'):
                continue  # Absorbing states don't change

            if from_status == 'Current':
                # Stress: increase delinquency, decrease prepayment
                if 'to_delinquent_0_30_pct' in result.columns:
                    result.at[idx, 'to_delinquent_0_30_pct'] = np.clip(
                        row['to_delinquent_0_30_pct'] * (1 + shift_sign * pct),
                        0, 1,
                    )
                if 'to_fully_paid_pct' in result.columns:
                    result.at[idx, 'to_fully_paid_pct'] = np.clip(
                        row['to_fully_paid_pct'] * (1 - shift_sign * pct),
                        0, 1,
                    )
            else:
                # Non-current states: adjust cure rate (→Current)
                if 'to_current_pct' in result.columns:
                    result.at[idx, 'to_current_pct'] = np.clip(
                        row['to_current_pct'] * (1 - shift_sign * pct),
                        0, 1,
                    )

            # Re-normalize: adjust the "stay" or "roll forward" prob
            # to make row sum to 1
            total = sum(result.at[idx, c] for c in pct_cols)
            if total > 0:
                # Find the residual column (Current→Current for Current,
                # or the roll-forward column for delinquent states)
                if from_status == 'Current':
                    residual_col = 'to_current_pct'
                elif from_status == 'Delinquent (0-30)':
                    residual_col = 'to_late_1_pct'
                elif from_status == 'Late_1':
                    residual_col = 'to_late_2_pct'
                elif from_status == 'Late_2':
                    residual_col = 'to_late_3_pct'
                elif from_status == 'Late_3':
                    residual_col = 'to_charged_off_pct'
                else:
                    residual_col = None

                if residual_col and residual_col in result.columns:
                    other_sum = sum(
                        result.at[idx, c] for c in pct_cols if c != residual_col
                    )
                    result.at[idx, residual_col] = np.clip(
                        1.0 - other_sum, 0, 1
                    )

        return result

    stressed_probs = _apply_shift(base_probs, +1, stress_pct)
    upside_probs = _apply_shift(base_probs, -1, upside_pct)

    return {
        'Base': base_probs,
        'Stress': stressed_probs,
        'Upside': upside_probs,
    }


def build_scenarios_from_percentiles(
    raw_age_probs: pd.DataFrame,
    pool_cdr: float,
    base_cpr: float,
    scenario_cdrs: dict[str, float],
    scenario_cprs: dict[str, float],
) -> dict[str, pd.DataFrame]:
    """
    Build scenario probability sets from percentile-derived CDR/CPR values.

    For each scenario, scales delinquency/cure transition probabilities
    by the ratio of scenario CDR to pool CDR, then scales age-specific
    Current→Fully Paid rates by the ratio of scenario CPR to the
    pool-level CPR. This preserves the age-dependent shape of the
    prepayment curve while shifting its overall level.

    CDR scaling logic:
    - Current rows: to_delinquent_0_30_pct *= cdr_ratio
    - Non-Current non-absorbing rows: to_current_pct *= 1/cdr_ratio (cure inversely scaled)
    - Late_3→Charged Off NOT directly scaled (increases via re-normalization)
    - After multipliers: clamp to [0, 1], adjust residual column for row sum = 1.0

    CPR scaling logic:
    - Current rows: to_fully_paid_pct *= cpr_ratio at every age bucket
    - Shift difference into to_current_pct to maintain row sum = 1.0
    - Clamp to [0, 1] after scaling

    Parameters
    ----------
    raw_age_probs : pd.DataFrame
        Output of compute_age_transition_probabilities(states='7state',
        bucket_size=1). Empirical rates used directly (not CPR-adjusted).
    pool_cdr : float
        Pool-level CDR (denominator for the CDR scaling ratio).
    base_cpr : float
        Pool-level CPR including both full payoffs and curtailments
        (denominator for the CPR scaling ratio).
    scenario_cdrs : dict[str, float]
        Mapping scenario name → CDR value (e.g., {'Base': 0.07, ...}).
    scenario_cprs : dict[str, float]
        Mapping scenario name → CPR value (e.g., {'Base': 0.015, ...}).

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping scenario name → adjusted probability DataFrame, directly
        usable by compare_scenarios_transition().
    """
    pct_cols = [c for c in raw_age_probs.columns if c.endswith('_pct')]

    result = {}
    for name in scenario_cdrs:
        scenario_cdr = scenario_cdrs[name]
        scenario_cpr = scenario_cprs[name]

        # CDR ratio: how much to scale delinquency relative to pool level
        if pool_cdr > 0:
            cdr_ratio = scenario_cdr / pool_cdr
        else:
            cdr_ratio = 1.0

        # CPR ratio: how much to scale prepayment relative to pool-level CPR
        if base_cpr > 0:
            cpr_ratio = scenario_cpr / base_cpr
        else:
            cpr_ratio = 1.0

        adjusted = raw_age_probs.copy()

        for idx, row in adjusted.iterrows():
            from_status = row['from_status']

            if from_status in ('Charged Off', 'Fully Paid'):
                continue  # Absorbing states don't change

            if from_status == 'Current':
                # Scale delinquency rate by CDR ratio
                if 'to_delinquent_0_30_pct' in adjusted.columns:
                    adjusted.at[idx, 'to_delinquent_0_30_pct'] = np.clip(
                        row['to_delinquent_0_30_pct'] * cdr_ratio, 0, 1,
                    )

                # Scale prepayment rate by CPR ratio (preserves age-dependent shape)
                if 'to_fully_paid_pct' in adjusted.columns:
                    old_fp = row['to_fully_paid_pct']
                    new_fp = np.clip(old_fp * cpr_ratio, 0, 1)
                    adjusted.at[idx, 'to_fully_paid_pct'] = new_fp
            else:
                # Non-current: inversely scale cure rate
                if 'to_current_pct' in adjusted.columns:
                    inverse_ratio = (1.0 / cdr_ratio) if cdr_ratio > 0 else 1.0
                    adjusted.at[idx, 'to_current_pct'] = np.clip(
                        row['to_current_pct'] * inverse_ratio, 0, 1,
                    )

            # Re-normalize: adjust residual column to make row sum = 1.0
            if from_status == 'Current':
                residual_col = 'to_current_pct'
            elif from_status == 'Delinquent (0-30)':
                residual_col = 'to_late_1_pct'
            elif from_status == 'Late_1':
                residual_col = 'to_late_2_pct'
            elif from_status == 'Late_2':
                residual_col = 'to_late_3_pct'
            elif from_status == 'Late_3':
                residual_col = 'to_charged_off_pct'
            else:
                residual_col = None

            if residual_col and residual_col in adjusted.columns:
                other_sum = sum(
                    adjusted.at[idx, c] for c in pct_cols if c != residual_col
                )
                adjusted.at[idx, residual_col] = np.clip(
                    1.0 - other_sum, 0, 1
                )

        result[name] = adjusted

    return result


def compare_scenarios_transition(
    pool_state: dict,
    scenario_probs: dict,
    loss_severity: float,
    recovery_rate: float,
    pool_chars: dict,
    num_months: int,
    purchase_price: float,
    scenario_curtailment_rates: dict[str, dict[int, float]] | None = None,
) -> pd.DataFrame:
    """
    Run transition projections and IRR for each scenario.

    Parameters
    ----------
    pool_state : dict
        Output of build_pool_state().
    scenario_probs : dict
        Output of build_scenarios_transition(). Maps scenario name to probs DataFrame.
    loss_severity : float
        Loss severity fraction (FIXED across all scenarios).
    recovery_rate : float
        Recovery rate fraction (FIXED across all scenarios).
    pool_chars : dict
        Pool characteristics.
    num_months : int
        Number of months to project.
    purchase_price : float
        Purchase price as fraction of UPB.
    scenario_curtailment_rates : dict[str, dict[int, float]] | None
        Per-scenario curtailment rates. Maps scenario name → {age: smm}.
        If None, no curtailments applied (backward compatible).

    Returns
    -------
    pd.DataFrame
        Columns: scenario, irr, total_interest, total_principal, total_losses,
        total_recoveries, total_defaults, weighted_avg_life.
    """
    results = []

    for name, probs_df in scenario_probs.items():
        curtailment = scenario_curtailment_rates.get(name) if scenario_curtailment_rates else None
        cf_df = project_cashflows_transition(
            pool_state, probs_df, loss_severity, recovery_rate,
            pool_chars, num_months, curtailment_rates=curtailment,
        )
        irr = calculate_irr(cf_df, pool_chars, purchase_price)

        total_interest = cf_df['interest'].sum()
        total_principal = cf_df['total_principal'].sum()
        total_losses = cf_df['loss'].sum()
        total_recoveries = cf_df['recovery'].sum()
        total_defaults = cf_df['defaults'].sum()

        if total_principal > 0:
            wal = (cf_df['month'] * cf_df['total_principal']).sum() / total_principal / 12
        else:
            wal = 0.0

        results.append({
            'scenario': name,
            'loss_severity': loss_severity,
            'irr': irr,
            'total_interest': round(total_interest, 2),
            'total_principal': round(total_principal, 2),
            'total_losses': round(total_losses, 2),
            'total_recoveries': round(total_recoveries, 2),
            'total_defaults': round(total_defaults, 2),
            'weighted_avg_life': round(wal, 2),
        })

    return pd.DataFrame(results)
