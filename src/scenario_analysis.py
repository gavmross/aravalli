"""
Scenario analysis for base/stress/upside comparison.

Functions:
    compute_base_assumptions — extract base-case CDR, CPR, loss severity
    build_scenarios — apply multiplicative shifts for stress/upside
    compare_scenarios — run projections and IRR for each scenario
"""

import logging

import pandas as pd

from src.cashflow_engine import (
    compute_pool_assumptions,
    project_cashflows,
    calculate_irr,
)

logger = logging.getLogger(__name__)


def compute_base_assumptions(df_all: pd.DataFrame,
                             df_current: pd.DataFrame) -> dict:
    """
    Wrapper that calls compute_pool_assumptions() and returns the base case.

    Parameters
    ----------
    df_all : pd.DataFrame
        All loans in the filtered strata (all statuses).
    df_current : pd.DataFrame
        Current loans only with last_pymnt_d == 2019-03-01, with calc_amort columns.

    Returns
    -------
    dict with keys: 'cdr', 'cpr', 'loss_severity'
    """
    assumptions = compute_pool_assumptions(df_all, df_current)
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
