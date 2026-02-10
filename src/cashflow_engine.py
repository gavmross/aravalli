"""
Cash flow projection engine for pool-level loan portfolio analysis.

Functions:
    compute_pool_assumptions  — CDR, CPR, loss severity from historical data
    compute_pool_characteristics — pool-level aggregates (UPB, WAC, WAM, payment)
    project_cashflows — monthly projected cash flows with defaults, prepayments, recoveries
    calculate_irr — annualized IRR from projected cash flows
    solve_price — find purchase price that achieves a target IRR
    build_pool_state — construct initial 7-state pool from loan data
    project_cashflows_transition — state-transition monthly cash flow projection
    solve_price_transition — find purchase price for target IRR (transition model)
"""

import logging
from typing import Optional

import numpy as np
import numpy_financial as npf
import pandas as pd
from scipy.optimize import brentq

from src.amortization import calc_monthly_payment, calc_balance

logger = logging.getLogger(__name__)


def compute_pool_assumptions(df_all: pd.DataFrame,
                             df_active: pd.DataFrame) -> dict:
    """
    Compute base-case CDR, CPR, and loss severity for a filtered cohort.

    CDR uses the dv01 conditional methodology: trailing 12-month average MDR,
    annualized via CDR = 1 - (1 - avg_MDR)^12.

    Parameters
    ----------
    df_all : pd.DataFrame
        All loans in the filtered strata (all statuses). Must have
        'default_month' and 'payoff_month' columns from reconstruct_loan_timeline().
        Used for CDR and loss severity.
    df_active : pd.DataFrame
        Active loans (Current + In Grace + Late) with last_pymnt_d == 2019-03-01,
        with calc_amort columns already applied. Used for CPR.

    Returns
    -------
    dict with keys: 'cdr', 'cumulative_default_rate', 'avg_mdr', 'monthly_mdrs',
                    'cpr', 'loss_severity', 'recovery_rate'
        cdr is the conditional annualized rate (used in cash flow projections).
        cumulative_default_rate is the raw lifetime rate (display/reference only).
        avg_mdr is the average monthly default rate (un-annualized).
        monthly_mdrs is a list of 12 individual monthly MDRs.
    """
    # --- CDR (Conditional, trailing 12-month) ---
    snapshot = pd.Timestamp('2019-03-01')
    issue_dates = pd.to_datetime(df_all['issue_d'])

    # Parse default_month and payoff_month once
    default_month = pd.to_datetime(df_all['default_month'], errors='coerce')
    payoff_month = pd.to_datetime(df_all['payoff_month'], errors='coerce')

    monthly_mdrs = []
    for m in range(1, 13):  # trailing 12 months: Apr 2018 – Mar 2019
        month_start = snapshot - pd.DateOffset(months=m)
        month_end = month_start + pd.DateOffset(months=1)

        # Defaults in this month: loans whose default_month falls in [month_start, month_end)
        defaults_mask = (default_month >= month_start) & (default_month < month_end)
        default_upb = (
            df_all.loc[defaults_mask, 'funded_amnt'] -
            df_all.loc[defaults_mask, 'total_rec_prncp']
        ).clip(lower=0).sum()

        # Performing balance at start of month:
        # originated before month_start, not yet defaulted, not yet paid off
        originated = issue_dates <= month_start
        not_defaulted = default_month.isna() | (default_month > month_start)
        not_paid_off = payoff_month.isna() | (payoff_month > month_start)
        performing_mask = originated & not_defaulted & not_paid_off

        perf_idx = performing_mask.values.nonzero()[0]

        if len(perf_idx) > 0:
            funded = df_all['funded_amnt'].values[perf_idx].astype(np.float64)
            rates = df_all['int_rate'].values[perf_idx].astype(np.float64)
            terms = df_all['term_months'].values[perf_idx].astype(np.float64)

            perf_issue = issue_dates.values[perf_idx]
            td = np.datetime64(month_start) - perf_issue
            age_at_month = np.round(
                td.astype('timedelta64[D]').astype(np.float64) / 30.44
            ).astype(int).clip(min=0)

            pmt = calc_monthly_payment(funded, rates, terms)
            est_balance, _, _ = calc_balance(funded, rates, pmt,
                                             age_at_month.astype(np.float64))
            performing_balance = np.clip(est_balance, 0, None).sum()
        else:
            performing_balance = 0.0

        mdr = default_upb / performing_balance if performing_balance > 0 else 0.0
        monthly_mdrs.append(mdr)

    avg_mdr = float(np.mean(monthly_mdrs))
    cdr = 1 - (1 - avg_mdr) ** 12

    # Cumulative default rate for reference display only (NOT called CDR)
    charged_off = df_all[df_all['loan_status'] == 'Charged Off']
    defaulted_upb_cum = (
        charged_off['funded_amnt'] - charged_off['total_rec_prncp']
    ).clip(lower=0).sum()
    total_originated = df_all['funded_amnt'].sum()
    cumulative_default_rate = defaulted_upb_cum / total_originated if total_originated > 0 else 0.0

    # --- CPR ---
    # Same logic as pool_cpr in calculate_performance_metrics
    paid_active = df_active[df_active['last_pmt_beginning_balance'] > 0]
    if len(paid_active) > 0:
        total_beginning_balance = paid_active['last_pmt_beginning_balance'].sum()
        total_scheduled_principal = paid_active['last_pmt_scheduled_principal'].sum()
        total_unscheduled_principal = paid_active['last_pmt_unscheduled_principal'].sum()
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
        'cumulative_default_rate': cumulative_default_rate,
        'avg_mdr': avg_mdr,
        'monthly_mdrs': monthly_mdrs,
        'cpr': cpr,
        'loss_severity': loss_severity,
        'recovery_rate': recovery_rate,
    }


def compute_pool_characteristics(df_active: pd.DataFrame) -> dict:
    """
    Extract pool-level aggregates from active loans (March 2019 last payment).

    Parameters
    ----------
    df_active : pd.DataFrame
        Active loans (Current + In Grace + Late) with calc_amort columns applied.

    Returns
    -------
    dict with keys: 'total_upb', 'wac', 'wam', 'monthly_payment'
    """
    total_upb = df_active['out_prncp'].sum()

    if total_upb > 0:
        wac = np.average(df_active['int_rate'].values,
                         weights=df_active['out_prncp'].values)
        wam = int(round(np.average(df_active['updated_remaining_term'].values,
                                   weights=df_active['out_prncp'].values)))
    else:
        wac = 0.0
        wam = 0

    monthly_payment = df_active['installment'].sum()

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
    Project monthly cash flows for a pool of active loans.

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


# ===========================================================================
# State-Transition Cash Flow Engine (7-State Model)
# ===========================================================================

from src.portfolio_analytics import TRANSITION_STATES_7

# Map LC loan_status to 7-state model states
_STATUS_MAP_7 = {
    'Current': 'Current',
    'In Grace Period': 'Delinquent (0-30)',
    'Late (16-30 days)': 'Delinquent (0-30)',
    'Late (31-120 days)': None,  # Assigned to Late_1/2/3 based on timing
}


def adjust_prepayment_rates(age_probs: pd.DataFrame,
                            cpr: float) -> pd.DataFrame:
    """
    Replace empirical Current → Fully Paid rates with CPR-derived SMM.

    The empirical Current → Fully Paid transition rate from historical data
    includes all payoffs (maturity + voluntary prepayment), which overstates
    the forward-looking prepayment rate for currently performing loans.
    This replaces it with the SMM derived from the observed pool-level CPR
    and re-normalizes the Current row so probabilities sum to 1.0.

    Parameters
    ----------
    age_probs : pd.DataFrame
        Output of compute_age_transition_probabilities(states='7state').
    cpr : float
        Annual conditional prepayment rate (e.g., 0.0149 for 1.49%).

    Returns
    -------
    pd.DataFrame
        Copy with Current → Fully Paid set to SMM and Current → Current
        adjusted to maintain row sum = 1.0.
    """
    smm = 1 - (1 - cpr) ** (1 / 12)

    adjusted = age_probs.copy()
    mask = adjusted['from_status'] == 'Current'

    if mask.any():
        old_fp = adjusted.loc[mask, 'to_fully_paid_pct'].copy()
        adjusted.loc[mask, 'to_fully_paid_pct'] = smm

        # Shift the difference into "stay Current" to keep row sum = 1.0
        delta = old_fp - smm
        adjusted.loc[mask, 'to_current_pct'] += delta

        # Safety clamp
        for col in adjusted.columns:
            if col.startswith('to_') and col.endswith('_pct'):
                adjusted.loc[mask, col] = adjusted.loc[mask, col].clip(0, 1)

    return adjusted


def build_pool_state(df: pd.DataFrame,
                     include_statuses: list[str] | None = None) -> dict:
    """
    Construct initial pool state from loan DataFrame for the 7-state model.

    Maps each loan's LC status to one of the 7 model states, grouped by
    individual loan age in months.

    Parameters
    ----------
    df : pd.DataFrame
        Loan data with columns: loan_status, out_prncp, int_rate,
        updated_remaining_term, installment, loan_age_months.
        For Late (31-120 days) loans, also needs late_31_120_month, last_pymnt_d.
    include_statuses : list[str] | None
        LC statuses to include. Defaults to ['Current'].

    Returns
    -------
    dict
        Keys: 'states' (dict[str, dict[int, float]] — {state: {age: upb}}),
        'total_upb', 'wac', 'wam', 'monthly_payment'.
    """
    if include_statuses is None:
        include_statuses = [
            'Current', 'In Grace Period',
            'Late (16-30 days)', 'Late (31-120 days)',
        ]

    df_pool = df[df['loan_status'].isin(include_statuses)].copy()

    if len(df_pool) == 0:
        return {
            'states': {s: {} for s in TRANSITION_STATES_7},
            'total_upb': 0.0,
            'wac': 0.0,
            'wam': 0,
            'monthly_payment': 0.0,
        }

    # Initialize state dict
    states = {s: {} for s in TRANSITION_STATES_7}

    for _, loan in df_pool.iterrows():
        lc_status = loan['loan_status']
        upb = float(loan['out_prncp'])
        age = int(loan['loan_age_months'])

        if upb <= 0:
            continue

        if lc_status in ('Current',):
            model_state = 'Current'
        elif lc_status in ('In Grace Period', 'Late (16-30 days)'):
            model_state = 'Delinquent (0-30)'
        elif lc_status == 'Late (31-120 days)':
            # Determine Late sub-state based on how long in Late
            if pd.notna(loan.get('late_31_120_month')) and pd.notna(loan.get('last_pymnt_d')):
                late_start = pd.Timestamp(loan['late_31_120_month'])
                snapshot = pd.Timestamp('2019-03-01')
                months_in_late = max(0, int(round(
                    (snapshot - late_start).days / 30.44
                )))
                if months_in_late <= 0:
                    model_state = 'Late_1'
                elif months_in_late == 1:
                    model_state = 'Late_2'
                else:
                    model_state = 'Late_3'
            else:
                model_state = 'Late_1'  # Default to first sub-state
        else:
            continue

        states[model_state][age] = states[model_state].get(age, 0.0) + upb

    # Pool-level aggregates
    total_upb = df_pool['out_prncp'].sum()
    if total_upb > 0:
        wac = float(np.average(df_pool['int_rate'].values,
                               weights=df_pool['out_prncp'].values))
        wam = int(round(np.average(df_pool['updated_remaining_term'].values,
                                   weights=df_pool['out_prncp'].values)))
    else:
        wac = 0.0
        wam = 0
    monthly_payment = df_pool['installment'].sum()

    return {
        'states': states,
        'total_upb': total_upb,
        'wac': wac,
        'wam': wam,
        'monthly_payment': monthly_payment,
    }


def _build_prob_lookup(age_probs: pd.DataFrame) -> dict:
    """
    Build a lookup dict from age_probs DataFrame.

    Returns
    -------
    dict
        {(from_status, age_int): {to_state: prob}}
    """
    lookup = {}
    max_age_by_status = {}

    for _, row in age_probs.iterrows():
        from_status = row['from_status']
        try:
            age = int(row['age_bucket'])
        except (ValueError, TypeError):
            continue  # Skip non-integer bucket labels

        probs = {}
        for col in age_probs.columns:
            if col.startswith('to_') and col.endswith('_pct'):
                # Convert column name back to state name
                state_part = col[3:-4]  # strip 'to_' and '_pct'
                state_map = {
                    'current': 'Current',
                    'delinquent_0_30': 'Delinquent (0-30)',
                    'late_1': 'Late_1',
                    'late_2': 'Late_2',
                    'late_3': 'Late_3',
                    'charged_off': 'Charged Off',
                    'fully_paid': 'Fully Paid',
                }
                to_state = state_map.get(state_part)
                if to_state is not None:
                    probs[to_state] = float(row[col])

            lookup[(from_status, age)] = probs

        # Track max age per from_status
        if from_status not in max_age_by_status:
            max_age_by_status[from_status] = age
        else:
            max_age_by_status[from_status] = max(max_age_by_status[from_status], age)

    return lookup, max_age_by_status


def _get_probs(lookup: dict, max_ages: dict, from_status: str,
               age: int) -> dict:
    """Get transition probs for (from_status, age), using nearest available if missing."""
    key = (from_status, age)
    if key in lookup:
        return lookup[key]

    # Use nearest available age (cap at max known age)
    max_age = max_ages.get(from_status, 0)
    capped_age = min(age, max_age)
    key = (from_status, capped_age)
    if key in lookup:
        return lookup[key]

    # Search downward from capped_age
    for a in range(capped_age, -1, -1):
        if (from_status, a) in lookup:
            return lookup[(from_status, a)]

    # No data at all — return identity (stay in same state)
    return {from_status: 1.0}


def project_cashflows_transition(
    pool_state: dict,
    age_probs: pd.DataFrame,
    loss_severity: float,
    recovery_rate: float,
    pool_chars: dict,
    num_months: int,
) -> pd.DataFrame:
    """
    Project monthly cash flows using the 7-state transition model.

    Tracks pool balance across 7 states (Current, Delinquent (0-30),
    Late_1, Late_2, Late_3, Charged Off, Fully Paid) with age-specific
    empirical transition probabilities. Defaults flow through a realistic
    5-month delinquency pipeline instead of hitting immediately.

    Parameters
    ----------
    pool_state : dict
        Output of build_pool_state(). Contains 'states', 'total_upb',
        'wac', 'wam', 'monthly_payment'.
    age_probs : pd.DataFrame
        Output of compute_age_transition_probabilities(states='7state',
        bucket_size=1).
    loss_severity : float
        Fraction of defaulted UPB that is lost.
    recovery_rate : float
        Fraction of defaulted UPB that is recovered.
    pool_chars : dict
        Pool characteristics (wac, monthly_payment used for interest/principal).
    num_months : int
        Number of months to project.

    Returns
    -------
    pd.DataFrame
        Monthly projection with columns compatible with calculate_irr().
    """
    # Build lookup dict
    lookup, max_ages = _build_prob_lookup(age_probs)

    # Initialize state balances: {state: {age: upb}}
    current_states = {}
    for state in TRANSITION_STATES_7:
        current_states[state] = dict(pool_state['states'].get(state, {}))

    wac = pool_chars.get('wac', pool_state.get('wac', 0.0))
    monthly_rate = wac / 12
    pool_monthly_payment = pool_chars.get('monthly_payment',
                                          pool_state.get('monthly_payment', 0.0))
    # Notional balance tracks standard amortization (unaffected by transitions).
    # Used to scale the pool payment: when no loans leave Current, payment stays
    # fixed and the pool amortizes to $0 exactly like the Simple model.  When
    # loans transition out, current_upb < notional_upb and the payment scales down.
    notional_upb = sum(current_states['Current'].values())

    start_date = pd.Timestamp('2019-03-01')
    cumulative_default = 0.0
    cumulative_paid = 0.0

    rows = []

    for t in range(1, num_months + 1):
        # Sum current UPB by state
        current_upb = sum(current_states['Current'].values())
        delinq_upb = sum(current_states['Delinquent (0-30)'].values())
        late1_upb = sum(current_states['Late_1'].values())
        late2_upb = sum(current_states['Late_2'].values())
        late3_upb = sum(current_states['Late_3'].values())

        beginning_balance = current_upb + delinq_upb + late1_upb + late2_upb + late3_upb

        if beginning_balance <= 0.01:
            break

        # New state balances for next period
        new_states = {s: {} for s in TRANSITION_STATES_7}

        # Track cash flow components
        prepayments = 0.0
        new_defaults = 0.0

        # Process transitions for each active state
        for state in ['Current', 'Delinquent (0-30)', 'Late_1', 'Late_2', 'Late_3']:
            for age, upb in current_states[state].items():
                if upb <= 0:
                    continue

                probs = _get_probs(lookup, max_ages, state, age)

                for to_state in TRANSITION_STATES_7:
                    p = probs.get(to_state, 0.0)
                    if p <= 0:
                        continue

                    flow = upb * p
                    new_age = age + 1

                    if to_state == 'Charged Off':
                        new_defaults += flow
                    elif to_state == 'Fully Paid':
                        if state == 'Current':
                            prepayments += flow
                    else:
                        new_states[to_state][new_age] = (
                            new_states[to_state].get(new_age, 0.0) + flow
                        )

        # Absorbing states carry forward
        for age, upb in current_states['Charged Off'].items():
            new_states['Charged Off'][age] = (
                new_states['Charged Off'].get(age, 0.0) + upb
            )
        for age, upb in current_states['Fully Paid'].items():
            new_states['Fully Paid'][age] = (
                new_states['Fully Paid'].get(age, 0.0) + upb
            )

        # Cash flows from Current UPB
        # Scale payment by current_upb / notional_upb so that:
        #   - If no transitions: fraction=1, payment is fixed → standard amortization
        #   - If loans leave Current: fraction<1, payment scales down proportionally
        fraction = current_upb / notional_upb if notional_upb > 0 else 0.0
        scaled_payment = pool_monthly_payment * fraction
        interest = current_upb * monthly_rate
        sched_principal = scaled_payment - interest
        sched_principal = max(0.0, min(sched_principal, current_upb))

        # Advance notional balance by standard amortization (independent of transitions)
        notional_interest = notional_upb * monthly_rate
        notional_principal = pool_monthly_payment - notional_interest
        notional_principal = max(0.0, min(notional_principal, notional_upb))
        notional_upb = max(0.0, notional_upb - notional_principal)

        # Scale scheduled principal: we need to subtract it from new Current balances
        new_current_upb = sum(new_states['Current'].values())
        if new_current_upb > 0 and sched_principal > 0:
            scale = max(0.0, 1.0 - sched_principal / new_current_upb)
            for age in new_states['Current']:
                new_states['Current'][age] *= scale

        # Losses and recoveries from new defaults
        losses = new_defaults * loss_severity
        recoveries = new_defaults * recovery_rate

        cumulative_default += new_defaults
        cumulative_paid += prepayments + sched_principal

        # Ending balance = all non-absorbing states
        ending_current = sum(new_states['Current'].values())
        ending_delinq = sum(new_states['Delinquent (0-30)'].values())
        ending_l1 = sum(new_states['Late_1'].values())
        ending_l2 = sum(new_states['Late_2'].values())
        ending_l3 = sum(new_states['Late_3'].values())
        ending_balance = ending_current + ending_delinq + ending_l1 + ending_l2 + ending_l3

        total_cashflow = interest + sched_principal + prepayments + recoveries

        rows.append({
            'month': t,
            'date': start_date + pd.DateOffset(months=t),
            'beginning_balance': round(beginning_balance, 2),
            'interest': round(interest, 2),
            'scheduled_principal': round(sched_principal, 2),
            'prepayments': round(prepayments, 2),
            'defaults': round(new_defaults, 2),
            'loss': round(losses, 2),
            'recovery': round(recoveries, 2),
            'total_principal': round(sched_principal + prepayments, 2),
            'ending_balance': round(ending_balance, 2),
            'total_cashflow': round(total_cashflow, 2),
            'current_upb': round(ending_current, 2),
            'delinquent_upb': round(ending_delinq, 2),
            'late_1_upb': round(ending_l1, 2),
            'late_2_upb': round(ending_l2, 2),
            'late_3_upb': round(ending_l3, 2),
            'default_upb': round(cumulative_default, 2),
            'fully_paid_upb': round(cumulative_paid, 2),
        })

        current_states = new_states

    return pd.DataFrame(rows)


def solve_price_transition(
    pool_state: dict,
    age_probs: pd.DataFrame,
    loss_severity: float,
    recovery_rate: float,
    pool_chars: dict,
    num_months: int,
    target_irr: float,
) -> Optional[float]:
    """
    Find the purchase price that achieves a target IRR (transition model).

    Since the projection doesn't depend on purchase price (price only affects
    the initial outlay in IRR), projects once and solves for price.

    Parameters
    ----------
    pool_state : dict
        Output of build_pool_state().
    age_probs : pd.DataFrame
        Age-specific transition probabilities (7-state, monthly).
    loss_severity : float
        Loss severity fraction.
    recovery_rate : float
        Recovery rate fraction.
    pool_chars : dict
        Pool characteristics.
    num_months : int
        Number of months to project.
    target_irr : float
        Target annualized IRR (e.g., 0.12 for 12%).

    Returns
    -------
    float | None
        Purchase price as fraction of UPB, or None if no solution.
    """
    cf_df = project_cashflows_transition(
        pool_state, age_probs, loss_severity, recovery_rate,
        pool_chars, num_months,
    )

    if len(cf_df) == 0:
        return None

    def objective(price: float) -> float:
        irr = calculate_irr(cf_df, pool_chars, price)
        if np.isnan(irr):
            return float('nan')
        return irr - target_irr

    try:
        solved_price = brentq(objective, 0.50, 1.50, xtol=1e-6)
    except ValueError:
        logger.warning(
            "solve_price_transition: no solution in [0.50, 1.50] for "
            "target IRR = %.4f", target_irr
        )
        return None

    return solved_price
