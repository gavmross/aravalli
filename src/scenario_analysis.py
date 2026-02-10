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


def compare_scenarios_transition(
    pool_state: dict,
    scenario_probs: dict,
    loss_severity: float,
    recovery_rate: float,
    pool_chars: dict,
    num_months: int,
    purchase_price: float,
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

    Returns
    -------
    pd.DataFrame
        Columns: scenario, irr, total_interest, total_principal, total_losses,
        total_recoveries, total_defaults, weighted_avg_life.
    """
    results = []

    for name, probs_df in scenario_probs.items():
        cf_df = project_cashflows_transition(
            pool_state, probs_df, loss_severity, recovery_rate,
            pool_chars, num_months,
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
