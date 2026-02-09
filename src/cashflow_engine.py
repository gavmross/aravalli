"""
Cash flow projection engine for pool-level loan portfolio analysis.

Functions:
    compute_pool_assumptions  — CDR, CPR, loss severity from historical data
    compute_pool_characteristics — pool-level aggregates (UPB, WAC, WAM, payment)
    project_cashflows — monthly projected cash flows with defaults, prepayments, recoveries
    calculate_irr — annualized IRR from projected cash flows
    solve_price — find purchase price that achieves a target IRR
"""

import logging
from typing import Optional

import numpy as np
import numpy_financial as npf
import pandas as pd
from scipy.optimize import brentq

logger = logging.getLogger(__name__)


def compute_pool_assumptions(df_all: pd.DataFrame,
                             df_current: pd.DataFrame) -> dict:
    """
    Compute base-case CDR, CPR, and loss severity for a filtered cohort.

    Parameters
    ----------
    df_all : pd.DataFrame
        All loans in the filtered strata (all statuses). Used for CDR and loss severity.
    df_current : pd.DataFrame
        Current loans only with last_pymnt_d == 2019-03-01, with calc_amort columns
        already applied. Used for CPR.

    Returns
    -------
    dict with keys: 'cdr', 'cdr_cumulative', 'cpr', 'loss_severity', 'recovery_rate'
        cdr is the annualized rate (used in cash flow projections).
        cdr_cumulative is the raw lifetime rate (retained for display/reference).
    """
    # --- CDR ---
    # Step 1: Cumulative CDR = sum(defaulted UPB) / sum(total originated UPB)
    charged_off = df_all[df_all['loan_status'] == 'Charged Off']
    defaulted_upb = (charged_off['funded_amnt'] - charged_off['total_rec_prncp']).sum()
    total_originated_upb = df_all['funded_amnt'].sum()
    cdr_cumulative = defaulted_upb / total_originated_upb if total_originated_upb > 0 else 0.0

    # Step 2: Annualize using compound survival formula
    # CDR_annual = 1 - (1 - CDR_cumulative)^(12 / WALA_months)
    # WALA_months = funded_amnt-weighted average loan age from issue_d to snapshot
    if cdr_cumulative > 0 and len(df_all) > 0:
        snapshot = pd.Timestamp('2019-03-01')
        issue_dates = pd.to_datetime(df_all['issue_d'])
        age_months = (snapshot - issue_dates).dt.days / 30.44  # approximate months
        wala_months = np.average(age_months, weights=df_all['funded_amnt'])
        if wala_months > 0:
            cdr = 1 - (1 - cdr_cumulative) ** (12 / wala_months)
        else:
            cdr = 0.0
    else:
        cdr = 0.0

    # --- CPR ---
    # Same logic as pool_cpr_current in calculate_performance_metrics
    paid_current = df_current[df_current['last_pmt_beginning_balance'] > 0]
    if len(paid_current) > 0:
        total_beginning_balance = paid_current['last_pmt_beginning_balance'].sum()
        total_scheduled_principal = paid_current['last_pmt_scheduled_principal'].sum()
        total_unscheduled_principal = paid_current['last_pmt_unscheduled_principal'].sum()
        denominator = total_beginning_balance - total_scheduled_principal
        if denominator > 0:
            smm = total_unscheduled_principal / denominator
            cpr = 1 - (1 - smm) ** 12
        else:
            cpr = 0.0
    else:
        cpr = 0.0

    # --- Loss Severity ---
    # From Charged Off loans with positive exposure, capped recoveries
    if len(charged_off) > 0:
        exposure = charged_off['funded_amnt'] - charged_off['total_rec_prncp']
        mask = exposure > 0
        if mask.sum() > 0:
            exposure_valid = exposure[mask]
            recoveries_valid = charged_off.loc[mask, 'recoveries']
            # Cap recoveries at exposure (can't recover more than was lost)
            capped_recoveries = np.minimum(recoveries_valid, exposure_valid)
            capped_upb_lost = (exposure_valid - capped_recoveries).clip(lower=0)
            loss_severity = capped_upb_lost.sum() / exposure_valid.sum()
            recovery_rate = capped_recoveries.sum() / exposure_valid.sum()
        else:
            loss_severity = 0.0
            recovery_rate = 0.0
    else:
        loss_severity = 0.0
        recovery_rate = 0.0

    return {
        'cdr': cdr,
        'cdr_cumulative': cdr_cumulative,
        'cpr': cpr,
        'loss_severity': loss_severity,
        'recovery_rate': recovery_rate,
    }


def compute_pool_characteristics(df_current: pd.DataFrame) -> dict:
    """
    Extract pool-level aggregates from Current loans (March 2019 last payment only).

    Parameters
    ----------
    df_current : pd.DataFrame
        Current loans with calc_amort columns applied.

    Returns
    -------
    dict with keys: 'total_upb', 'wac', 'wam', 'monthly_payment'
    """
    total_upb = df_current['out_prncp'].sum()

    if total_upb > 0:
        wac = np.average(df_current['int_rate'].values,
                         weights=df_current['out_prncp'].values)
        wam = int(round(np.average(df_current['updated_remaining_term'].values,
                                   weights=df_current['out_prncp'].values)))
    else:
        wac = 0.0
        wam = 0

    monthly_payment = df_current['installment'].sum()

    return {
        'total_upb': total_upb,
        'wac': wac,
        'wam': wam,
        'monthly_payment': monthly_payment,
    }


def project_cashflows(pool_chars: dict,
                      cdr: float,
                      cpr: float,
                      loss_severity: float,
                      purchase_price: float) -> pd.DataFrame:
    """
    Project monthly cash flows for a pool of Current loans.

    Parameters
    ----------
    pool_chars : dict
        Output of compute_pool_characteristics().
    cdr : float
        Annual default rate (e.g., 0.10 for 10%).
    cpr : float
        Annual prepayment rate (e.g., 0.12 for 12%).
    loss_severity : float
        Fraction of defaulted UPB that is lost (e.g., 0.85 for 85%).
    purchase_price : float
        Purchase price as fraction of UPB (e.g., 0.95 for 95 cents on the dollar).

    Returns
    -------
    pd.DataFrame with columns: month, date, beginning_balance, defaults, loss,
        recovery, interest, scheduled_principal, prepayments, total_principal,
        ending_balance, total_cashflow
    """
    total_upb = pool_chars['total_upb']
    wac = pool_chars['wac']
    wam = pool_chars['wam']
    monthly_payment = pool_chars['monthly_payment']

    # Convert annual rates to monthly
    # MDR = 1 - (1 - CDR)^(1/12)
    mdr = 1 - (1 - cdr) ** (1 / 12)
    # SMM = 1 - (1 - CPR)^(1/12)
    smm = 1 - (1 - cpr) ** (1 / 12)
    # Monthly interest rate
    monthly_rate = wac / 12

    # Start date: March 2019 (t=0), first payment at April 2019 (t=1)
    start_date = pd.Timestamp('2019-03-01')

    rows = []
    prev_ending_balance = total_upb

    for t in range(1, wam + 1):
        beginning_balance = prev_ending_balance
        if beginning_balance <= 0:
            break

        # 1. Defaults happen first
        defaults = beginning_balance * mdr
        loss = defaults * loss_severity
        recovery = defaults * (1 - loss_severity)

        # 2. Performing balance after removing defaults
        performing_balance = beginning_balance - defaults

        # 3. Interest on performing balance
        interest = performing_balance * monthly_rate

        # 4. Scheduled principal (from regular amortization payment)
        scheduled_principal = monthly_payment - interest
        # Can't exceed performing balance
        scheduled_principal = min(scheduled_principal, performing_balance)
        # Can't be negative (interest might exceed payment in edge cases)
        scheduled_principal = max(scheduled_principal, 0.0)

        # 5. Prepayments on remaining balance after scheduled principal
        prepayments = (performing_balance - scheduled_principal) * smm

        # 6. Total principal and ending balance
        total_principal = scheduled_principal + prepayments
        ending_balance = performing_balance - total_principal
        ending_balance = max(ending_balance, 0.0)  # Floor at zero

        # 7. Total cash flow to investor
        total_cashflow = interest + total_principal + recovery

        # Date for this month
        date = start_date + pd.DateOffset(months=t)

        rows.append({
            'month': t,
            'date': date,
            'beginning_balance': round(beginning_balance, 2),
            'defaults': round(defaults, 2),
            'loss': round(loss, 2),
            'recovery': round(recovery, 2),
            'interest': round(interest, 2),
            'scheduled_principal': round(scheduled_principal, 2),
            'prepayments': round(prepayments, 2),
            'total_principal': round(total_principal, 2),
            'ending_balance': round(ending_balance, 2),
            'total_cashflow': round(total_cashflow, 2),
        })

        prev_ending_balance = ending_balance

    return pd.DataFrame(rows)


def calculate_irr(cashflows_df: pd.DataFrame,
                  pool_chars: dict,
                  purchase_price: float) -> float:
    """
    Compute annualized IRR from projected cash flows.

    Parameters
    ----------
    cashflows_df : pd.DataFrame
        Output of project_cashflows().
    pool_chars : dict
        Output of compute_pool_characteristics().
    purchase_price : float
        Purchase price as fraction of UPB.

    Returns
    -------
    float — annualized IRR (e.g., 0.12 for 12%)
    """
    # Build cash flow array: month 0 = purchase outlay, months 1..N = inflows
    cf = np.zeros(len(cashflows_df) + 1)
    cf[0] = -(purchase_price * pool_chars['total_upb'])  # Initial outlay (negative)
    cf[1:] = cashflows_df['total_cashflow'].values  # Monthly inflows

    # Compute monthly IRR
    monthly_irr = npf.irr(cf)

    if np.isnan(monthly_irr):
        logger.warning("npf.irr returned NaN — IRR could not be computed")
        return float('nan')

    # Annualize: (1 + monthly_irr)^12 - 1
    annual_irr = (1 + monthly_irr) ** 12 - 1

    # Verify: NPV at this rate should be ≈ 0
    npv_check = npf.npv(monthly_irr, cf)
    if abs(npv_check) > max(1.0, abs(cf[0]) * 1e-6):
        logger.warning("IRR verification: NPV = %.4f (expected ≈ 0)", npv_check)

    return annual_irr


def solve_price(pool_chars: dict,
                target_irr: float,
                cdr: float,
                cpr: float,
                loss_severity: float) -> Optional[float]:
    """
    Find the purchase price that achieves a target IRR.

    Uses scipy.optimize.brentq to solve for price where IRR(price) == target_irr.

    Parameters
    ----------
    pool_chars : dict
        Output of compute_pool_characteristics().
    target_irr : float
        Target annualized IRR (e.g., 0.12 for 12%).
    cdr : float
        Annual default rate.
    cpr : float
        Annual prepayment rate.
    loss_severity : float
        Loss severity.

    Returns
    -------
    float — purchase price (e.g., 0.92 for 92 cents), or None if no solution.
    """
    def objective(price: float) -> float:
        cf_df = project_cashflows(pool_chars, cdr, cpr, loss_severity, price)
        irr = calculate_irr(cf_df, pool_chars, price)
        if np.isnan(irr):
            return float('nan')
        return irr - target_irr

    try:
        solved_price = brentq(objective, 0.50, 1.50, xtol=1e-6)
    except ValueError:
        logger.warning(
            "solve_price: no solution in [0.50, 1.50] for target IRR = %.4f",
            target_irr
        )
        return None

    return solved_price
