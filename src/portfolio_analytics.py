"""
Portfolio analytics functions for credit metrics, performance metrics, and transition matrix.

Extracted from scripts/analysis.ipynb — DO NOT MODIFY LOGIC.
Only imports were added.
"""

import numpy as np
import pandas as pd


def calculate_credit_metrics(df: pd.DataFrame,
                            strata_col: str,
                            verbose: bool = True) -> pd.DataFrame:
    """
    Calculate credit metrics for each pool strata.

    Parameters:
    -----------
    df : pd.DataFrame
        Loan data with required columns
    strata_col : str
        Column to stratify by (e.g., 'grade', 'term_months', 'purpose', 'addr_state', 'issue_quarter')
    verbose : bool
        Print results (default: True)

    Returns:
    --------
    results_df : pd.DataFrame
        Credit metrics by strata

    Metrics Calculated:
    -------------------
    ORIGINAL (all loans, weighted by funded_amnt):
    - orig_total_upb_mm: Total original balance (in millions)
    - orig_loan_count: Number of loans
    - orig_wac: Weighted average coupon
    - orig_wam: Weighted average maturity
    - orig_avg_fico: Weighted average FICO
    - orig_avg_dti: Weighted average DTI

    CURRENT/ACTIVE (Current + In Grace + Late loans, weighted by out_prncp):
    - curr_total_upb_mm: Total current balance (active loans only, in millions)
    - curr_upb_current_mm: UPB of "Current" loans (in millions)
    - curr_upb_grace_mm: UPB of "In Grace Period" loans (in millions)
    - curr_upb_late_16_30_mm: UPB of "Late (16-30 days)" loans (in millions)
    - curr_upb_late_31_120_mm: UPB of "Late (31-120 days)" loans (in millions)
    - curr_wac: Weighted average coupon (Current loans only)
    - curr_wam: Weighted average remaining maturity (Current loans only)
    - curr_wala: Weighted average loan age (Current loans only)
    - curr_avg_fico: Weighted average FICO (Current loans only)
    - curr_avg_dti: Weighted average DTI (Current loans only)

    TERMINAL STATUS:
    - upb_fully_paid_mm: Sum of total_rec_prncp for "Fully Paid" loans (in millions)
    - upb_defaulted_mm: Sum of lgd for "Charged Off" loans (in millions)
    - upb_recovered_mm: Sum of recoveries for "Charged Off" loans (in millions)
    """

    # Required columns
    required_cols = ['funded_amnt', 'out_prncp', 'int_rate', 'term_months',
                     'updated_remaining_term', 'orig_exp_payments_made',
                     'original_fico', 'latest_fico', 'dti_clean', 'loan_status',
                     'total_rec_prncp', 'recoveries']

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if strata_col not in df.columns:
        raise ValueError(f"Strata column '{strata_col}' not found in dataframe")

    results = []

    # 1. Calculate aggregate metrics for entire portfolio (strata_value = 'ALL')
    agg_metrics = _calculate_metrics_for_group(
        df,
        strata_col=strata_col,
        strata_value='ALL'
    )
    results.append(agg_metrics)

    # 2. Calculate metrics for each strata value
    strata_values = df[strata_col].dropna().unique()
    for strata_value in sorted(strata_values):
        strata_df = df[df[strata_col] == strata_value].copy()

        strata_metrics = _calculate_metrics_for_group(
            strata_df,
            strata_col=strata_col,
            strata_value=strata_value
        )
        results.append(strata_metrics)

    # Combine all results
    results_df = pd.DataFrame(results)

    # Convert orig_total_upb to millions
    if 'orig_total_upb' in results_df.columns:
        results_df['orig_total_upb_mm'] = results_df['orig_total_upb'] / 1e6
        results_df = results_df.drop(columns=['orig_total_upb'])

    # Convert active_total_upb to millions
    if 'active_total_upb' in results_df.columns:
        results_df['active_total_upb_mm'] = results_df['active_total_upb'] / 1e6
        results_df = results_df.drop(columns=['active_total_upb'])

    # Reorder columns for readability
    col_order = [
        'strata_type', 'strata_value',
        'orig_total_upb_mm', 'orig_loan_count', 'orig_wac', 'orig_wam', 'orig_avg_fico', 'orig_avg_dti',
        'active_total_upb_mm', 'active_upb_current_perc', 'active_upb_grace_perc', 'active_upb_late_16_30_perc', 'active_upb_late_31_120_perc',
        'curr_wac', 'curr_wam', 'curr_wala', 'curr_avg_fico', 'curr_avg_dti',
        'upb_fully_paid_perc', 'upb_lost_perc'
    ]

    results_df = results_df[col_order]

    if verbose:
        print(f"="*120)
        print(f"CREDIT METRICS BY {strata_col.upper()}")
        print(f"="*120)

        # Format for display
        display_df = results_df.copy()

        # Format percentage columns
        pct_cols = [col for col in display_df.columns if col.endswith('_perc')]
        for col in pct_cols:
            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%")

        # Format WAC columns as percentages
        wac_cols = ['orig_wac', 'curr_wac']
        for col in wac_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%")

        print(display_df.to_string(index=False))

    return results_df


def _calculate_metrics_for_group(df: pd.DataFrame,
                                 strata_col: str,
                                 strata_value: any) -> dict:
    """
    Helper function to calculate credit metrics for a specific group.
    """

    metrics = {
        'strata_type': strata_col,
        'strata_value': strata_value
    }

    # =========================================================================
    # ORIGINAL METRICS (all loans, weighted by funded_amnt)
    # =========================================================================

    orig_weight = df['funded_amnt'].values
    orig_total_weight = orig_weight.sum()

    metrics['orig_total_upb'] = orig_total_weight
    metrics['orig_loan_count'] = len(df)

    if orig_total_weight > 0:
        # WAC (Weighted Average Coupon)
        metrics['orig_wac'] = np.average(df['int_rate'].values, weights=orig_weight)

        # WAM (Weighted Average Maturity)
        metrics['orig_wam'] = np.average(df['term_months'].values, weights=orig_weight)

        # Average FICO (original)
        metrics['orig_avg_fico'] = np.average(df['original_fico'].fillna(0).values, weights=orig_weight)

        # Average DTI
        metrics['orig_avg_dti'] = np.average(df['dti_clean'].fillna(0).values, weights=orig_weight)
    else:
        metrics['orig_wac'] = 0
        metrics['orig_wam'] = 0
        metrics['orig_avg_fico'] = 0
        metrics['orig_avg_dti'] = 0

    # =========================================================================
    # CURRENT/ACTIVE METRICS (active loans only, weighted by out_prncp)
    # =========================================================================

    # Define active statuses
    active_statuses = ['Current', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
    active_loans = df[df['loan_status'].isin(active_statuses)].copy()

    # Total active UPB
    active_total_upb = active_loans['out_prncp'].sum()
    metrics['active_total_upb'] = active_total_upb

    # Breakdown by status (as percentages of active UPB)
    if active_total_upb > 0:
        metrics['active_upb_current_perc'] = active_loans[active_loans['loan_status'] == 'Current']['out_prncp'].sum() / active_total_upb
        metrics['active_upb_grace_perc'] = active_loans[active_loans['loan_status'] == 'In Grace Period']['out_prncp'].sum() / active_total_upb
        metrics['active_upb_late_16_30_perc'] = active_loans[active_loans['loan_status'] == 'Late (16-30 days)']['out_prncp'].sum() / active_total_upb
        metrics['active_upb_late_31_120_perc'] = active_loans[active_loans['loan_status'] == 'Late (31-120 days)']['out_prncp'].sum() / active_total_upb
    else:
        metrics['active_upb_current_perc'] = 0
        metrics['active_upb_grace_perc'] = 0
        metrics['active_upb_late_16_30_perc'] = 0
        metrics['active_upb_late_31_120_perc'] = 0

    # Current metrics (weighted by out_prncp for Current loans only)
    current_only = active_loans[active_loans['loan_status'] == 'Current']
    current_upb = current_only['out_prncp'].sum()
    if len(current_only) > 0 and current_upb > 0:
        curr_weight = current_only['out_prncp'].values

        # WAC (current)
        metrics['curr_wac'] = np.average(current_only['int_rate'].values, weights=curr_weight)

        # WAM (current - remaining term)
        metrics['curr_wam'] = np.average(current_only['updated_remaining_term'].values, weights=curr_weight)

        # WALA (current)
        metrics['curr_wala'] = np.average(current_only['orig_exp_payments_made'].values, weights=curr_weight)

        # Average FICO (current)
        metrics['curr_avg_fico'] = np.average(current_only['latest_fico'].fillna(0).values, weights=curr_weight)

        # Average DTI (current)
        metrics['curr_avg_dti'] = np.average(current_only['dti_clean'].fillna(0).values, weights=curr_weight)
    else:
        metrics['curr_wac'] = 0
        metrics['curr_wam'] = 0
        metrics['curr_wala'] = 0
        metrics['curr_avg_fico'] = 0
        metrics['curr_avg_dti'] = 0

    # =========================================================================
    # TERMINAL STATUS METRICS (as percentages of original UPB)
    # =========================================================================

    # Fully Paid: Sum of total_rec_prncp as % of original UPB
    fully_paid = df[df['loan_status'] == 'Fully Paid']
    if orig_total_weight > 0:
        metrics['upb_fully_paid_perc'] = fully_paid['total_rec_prncp'].sum() / orig_total_weight
    else:
        metrics['upb_fully_paid_perc'] = 0

    # Charged Off: Calculate upb_lost = funded_amnt - total_rec_prncp - recoveries
    charged_off = df[df['loan_status'] == 'Charged Off']
    upb_lost = (charged_off['funded_amnt'] - charged_off['total_rec_prncp'] - charged_off['recoveries']).clip(lower=0).sum()

    if orig_total_weight > 0:
        metrics['upb_lost_perc'] = upb_lost / orig_total_weight
    else:
        metrics['upb_lost_perc'] = 0

    return metrics


def calculate_performance_metrics(df: pd.DataFrame,
                                  vintage_col: str = 'issue_quarter',
                                  verbose: bool = True) -> pd.DataFrame:
    """
    Calculate performance metrics by vintage.

    Parameters:
    -----------
    df : pd.DataFrame
        Loan data with required columns
    vintage_col : str
        Column to group by vintage (default: 'issue_quarter')
    verbose : bool
        Print results (default: True)

    Returns:
    --------
    results_df : pd.DataFrame
        Performance metrics by vintage

    Metrics Calculated:
    -------------------
    COUNTS & UPB:
    - orig_loan_count: Total number of loans
    - orig_upb_mm: Total original UPB (in millions)

    LOAN STATUS PERCENTAGES (% of original loan count):
    - pct_active: % Current + Late loans
    - pct_current: % Current only
    - pct_fully_paid: % Fully Paid
    - pct_charged_off: % Charged Off

    DEFAULT METRICS:
    - defaulted_loan_count: Number of charged off loans
    - pct_defaulted_count: % of loans charged off
    - defaulted_upb_mm: Defaulted UPB (in millions)
    - pct_defaulted_upb: % of original UPB defaulted

    PREPAYMENT METRICS:
    - pool_cpr_active: CPR for all active loans (Current + Late)
    - pool_cpr_current: CPR for Current loans only
    - pct_prepaid_current: Total prepaid principal / expected principal (Current loans only)

    LOSS METRICS:
    - loss_severity: lgd / (funded_amnt - total_rec_prncp) for charged off loans
    """

    # Required columns
    required_cols = ['funded_amnt', 'out_prncp', 'total_rec_prncp', 'loan_status',
                     'last_pmt_beginning_balance', 'last_pmt_scheduled_principal',
                     'last_pmt_unscheduled_principal', 'orig_exp_principal_paid',
                     'recoveries']

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if vintage_col not in df.columns:
        raise ValueError(f"Vintage column '{vintage_col}' not found in dataframe")

    results = []

    # Get unique vintages
    vintages = df[vintage_col].dropna().unique()

    for vintage in sorted(vintages):
        vintage_df = df[df[vintage_col] == vintage].copy()

        metrics = {
            'vintage': vintage
        }

        # =====================================================================
        # COUNTS & UPB
        # =====================================================================

        total_loans = len(vintage_df)
        total_upb = vintage_df['funded_amnt'].sum()

        metrics['orig_loan_count'] = total_loans
        metrics['orig_upb_mm'] = total_upb / 1e6  # Convert to millions

        # =====================================================================
        # LOAN STATUS PERCENTAGES
        # =====================================================================

        # Active loans: Current + Late
        active_statuses = ['Current', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
        active_count = len(vintage_df[vintage_df['loan_status'].isin(active_statuses)])
        current_count = len(vintage_df[vintage_df['loan_status'] == 'Current'])
        fully_paid_count = len(vintage_df[vintage_df['loan_status'] == 'Fully Paid'])
        charged_off_count = len(vintage_df[vintage_df['loan_status'] == 'Charged Off'])

        metrics['pct_active'] = active_count / total_loans if total_loans > 0 else 0
        metrics['pct_current'] = current_count / total_loans if total_loans > 0 else 0
        metrics['pct_fully_paid'] = fully_paid_count / total_loans if total_loans > 0 else 0
        metrics['pct_charged_off'] = charged_off_count / total_loans if total_loans > 0 else 0

        # =====================================================================
        # DEFAULT METRICS
        # =====================================================================

        charged_off = vintage_df[vintage_df['loan_status'] == 'Charged Off']

        metrics['defaulted_loan_count'] = len(charged_off)
        metrics['pct_defaulted_count'] = len(charged_off) / total_loans if total_loans > 0 else 0

        # Defaulted UPB = funded_amnt - total_rec_prncp
        defaulted_upb = (charged_off['funded_amnt'] - charged_off['total_rec_prncp']).sum()
        metrics['defaulted_upb_mm'] = defaulted_upb / 1e6
        metrics['pct_defaulted_upb'] = defaulted_upb / total_upb if total_upb > 0 else 0

        # =====================================================================
        # CPR - ACTIVE LOANS (Current + Late)
        # =====================================================================

        active_loans = vintage_df[vintage_df['loan_status'].isin(active_statuses)].copy()
        paid_active = active_loans[active_loans['last_pmt_beginning_balance'] > 0].copy()

        if len(paid_active) > 0:
            total_beginning_balance = paid_active['last_pmt_beginning_balance'].sum()
            total_scheduled_principal = paid_active['last_pmt_scheduled_principal'].sum()
            total_unscheduled_principal = paid_active['last_pmt_unscheduled_principal'].sum()

            denominator = total_beginning_balance - total_scheduled_principal
            if denominator > 0:
                smm_active = total_unscheduled_principal / denominator
                cpr_active = 1 - (1 - smm_active) ** 12
            else:
                cpr_active = 0
        else:
            cpr_active = 0

        metrics['pool_cpr_active'] = cpr_active

        # =====================================================================
        # CPR - CURRENT LOANS ONLY
        # =====================================================================

        current_loans = vintage_df[vintage_df['loan_status'] == 'Current'].copy()
        paid_current = current_loans[current_loans['last_pmt_beginning_balance'] > 0].copy()

        if len(paid_current) > 0:
            total_beginning_balance = paid_current['last_pmt_beginning_balance'].sum()
            total_scheduled_principal = paid_current['last_pmt_scheduled_principal'].sum()
            total_unscheduled_principal = paid_current['last_pmt_unscheduled_principal'].sum()

            denominator = total_beginning_balance - total_scheduled_principal
            if denominator > 0:
                smm_current = total_unscheduled_principal / denominator
                cpr_current = 1 - (1 - smm_current) ** 12
            else:
                cpr_current = 0
        else:
            cpr_current = 0

        metrics['pool_cpr_current'] = cpr_current

        # =====================================================================
        # PREPAYMENT - CURRENT LOANS ONLY
        # =====================================================================

        # Calculate prepayment for Current loans only
        current_loans['prepaid_principal'] = (
            current_loans['total_rec_prncp'] - current_loans['orig_exp_principal_paid']
        ).clip(lower=0)

        total_prepaid_principal = current_loans['prepaid_principal'].sum()
        total_expected_principal = current_loans['orig_exp_principal_paid'].sum()

        if total_expected_principal > 0:
            pct_prepaid_current = total_prepaid_principal / total_expected_principal
        else:
            pct_prepaid_current = 0

        metrics['pct_prepaid_current'] = pct_prepaid_current

        # =====================================================================
        # LOSS SEVERITY & RECOVERY RATE
        # =====================================================================

        if len(charged_off) > 0:
            # Create explicit copy to avoid SettingWithCopyWarning
            charged_off_copy = charged_off.copy()

            # Calculate upb_lost = funded_amnt - total_rec_prncp - recoveries
            charged_off_copy['upb_lost'] = (
                charged_off_copy['funded_amnt'] -
                charged_off_copy['total_rec_prncp'] -
                charged_off_copy['recoveries']
            ).clip(lower=0)

            # Calculate exposure = funded_amnt - total_rec_prncp
            charged_off_copy['exposure'] = charged_off_copy['funded_amnt'] - charged_off_copy['total_rec_prncp']

            # Only calculate for loans with positive exposure
            valid_mask = charged_off_copy['exposure'] > 0

            if valid_mask.sum() > 0:
                # Cap recoveries at exposure (recovery rate can't exceed 100%)
                charged_off_copy['capped_recoveries'] = charged_off_copy[['recoveries', 'exposure']].min(axis=1)

                # Recalculate upb_lost with capped recoveries
                charged_off_copy['capped_upb_lost'] = (
                    charged_off_copy['funded_amnt'] -
                    charged_off_copy['total_rec_prncp'] -
                    charged_off_copy['capped_recoveries']
                ).clip(lower=0)

                # Loss severity = capped_upb_lost / exposure
                total_upb_lost = charged_off_copy.loc[valid_mask, 'capped_upb_lost'].sum()
                total_exposure = charged_off_copy.loc[valid_mask, 'exposure'].sum()
                loss_severity = total_upb_lost / total_exposure if total_exposure > 0 else 0

                # Recovery rate = capped_recoveries / exposure
                total_recoveries = charged_off_copy.loc[valid_mask, 'capped_recoveries'].sum()
                recovery_rate = total_recoveries / total_exposure if total_exposure > 0 else 0
            else:
                loss_severity = 0
                recovery_rate = 0
        else:
            loss_severity = 0
            recovery_rate = 0

        metrics['loss_severity'] = loss_severity
        metrics['recovery_rate'] = recovery_rate

        results.append(metrics)

    # Combine all results
    results_df = pd.DataFrame(results)

    # Reorder columns
    col_order = [
        'vintage', 'orig_loan_count', 'orig_upb_mm',
        'pct_active', 'pct_current', 'pct_fully_paid', 'pct_charged_off',
        'pct_defaulted_count', 'pct_defaulted_upb',
        'pool_cpr_active', 'pool_cpr_current', 'pct_prepaid_current',
        'loss_severity', 'recovery_rate'
    ]
    results_df = results_df[col_order]

    if verbose:
        print(f"="*120)
        print(f"PERFORMANCE METRICS BY VINTAGE")
        print(f"="*120)

        # Format for display
        display_df = results_df.copy()
        display_df['pct_active'] = display_df['pct_active'].apply(lambda x: f"{x*100:.2f}%")
        display_df['pct_current'] = display_df['pct_current'].apply(lambda x: f"{x*100:.2f}%")
        display_df['pct_fully_paid'] = display_df['pct_fully_paid'].apply(lambda x: f"{x*100:.2f}%")
        display_df['pct_charged_off'] = display_df['pct_charged_off'].apply(lambda x: f"{x*100:.2f}%")
        display_df['pct_defaulted_count'] = display_df['pct_defaulted_count'].apply(lambda x: f"{x*100:.2f}%")
        display_df['pct_defaulted_upb'] = display_df['pct_defaulted_upb'].apply(lambda x: f"{x*100:.2f}%")
        display_df['pool_cpr_active'] = display_df['pool_cpr_active'].apply(lambda x: f"{x*100:.2f}%")
        display_df['pool_cpr_current'] = display_df['pool_cpr_current'].apply(lambda x: f"{x*100:.2f}%")
        display_df['pct_prepaid_current'] = display_df['pct_prepaid_current'].apply(lambda x: f"{x*100:.2f}%")
        display_df['loss_severity'] = display_df['loss_severity'].apply(lambda x: f"{x*100:.2f}%")
        display_df['recovery_rate'] = display_df['recovery_rate'].apply(lambda x: f"{x*100:.2f}%")

        print(display_df.to_string(index=False))

    return results_df


def calculate_transition_matrix(df: pd.DataFrame,
                                strata_col: str = None,
                                verbose: bool = True) -> pd.DataFrame:
    """
    Calculate delinquency transition flow showing loan progression through states.

    Creates a flow-based transition analysis showing:
    - From Current: Where do loans go?
    - From Grace: What happens to loans that became delinquent?
    - From Late 16-30: Do they cure or progress?
    - From Late 31-120: What's the final outcome?

    Key assumption: Loans with curr_paid_late1_flag = 1 reached Late 16-30 and then cured.

    States (in order of severity):
    - Current (starting point - 100% of loans)
    - In Grace Period (first delinquency)
    - Late (16-30 days) (late fees charged here)
    - Late (31-120 days) (severe delinquency)
    - Charged Off (terminal - bad)
    - Fully Paid (terminal - good)

    Parameters:
    -----------
    df : pd.DataFrame
        Loan data with loan_status and curr_paid_late1_flag columns
    strata_col : str, optional
        Column to stratify by (e.g., 'grade'). If None, calculates for all loans.
    verbose : bool
        Print results (default: True)

    Returns:
    --------
    results_df : pd.DataFrame
        Transition flow probabilities for each stage
    """

    # Check for required column
    if 'curr_paid_late1_flag' not in df.columns:
        raise ValueError("Column 'curr_paid_late1_flag' not found in dataframe")

    results = []

    if strata_col is None:
        # Calculate for all loans
        flow = _calculate_transition_flow(df, strata_value='ALL')
        results.append(flow)
    else:
        # Validate strata column
        if strata_col not in df.columns:
            raise ValueError(f"Strata column '{strata_col}' not found in dataframe")

        # Calculate for all loans first
        flow_all = _calculate_transition_flow(df, strata_value='ALL')
        results.append(flow_all)

        # Calculate for each strata value
        strata_values = df[strata_col].dropna().unique()
        for strata_value in sorted(strata_values):
            strata_df = df[df[strata_col] == strata_value].copy()
            flow = _calculate_transition_flow(strata_df, strata_value=strata_value)
            results.append(flow)

    # Combine results
    results_df = pd.DataFrame(results)

    # Reorder columns
    col_order = [
        'strata_value', 'total_loans',
        'from_current_to_fully_paid_clean', 'from_current_to_current_clean', 'from_current_to_delinquent',
        'from_grace_still_in_grace', 'from_grace_progressed',
        'from_late16_cured', 'from_late16_still_in_late16', 'from_late16_progressed',
        'from_late31_still_in_late31', 'from_late31_charged_off'
    ]

    results_df = results_df[col_order]

    if verbose:
        print(f"="*120)
        print(f"DELINQUENCY TRANSITION FLOW")
        print(f"="*120)

        # Format for display
        display_df = results_df.copy()

        # Format percentages
        pct_cols = [col for col in display_df.columns if col.startswith('from_')]
        for col in pct_cols:
            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%")

        print(display_df.to_string(index=False))

        # Print flow summary
        print(f"\n{'='*120}")
        print("FLOW SUMMARY")
        print(f"{'='*120}")

        for idx, row in display_df.iterrows():
            strata = row['strata_value']
            print(f"\n{strata}:")
            print(f"  From Current (100%):")
            print(f"    → Fully Paid (clean): {row['from_current_to_fully_paid_clean']}")
            print(f"    → Current (clean): {row['from_current_to_current_clean']}")
            print(f"    → Delinquent: {row['from_current_to_delinquent']}")
            print(f"  From Grace:")
            print(f"    → Still in Grace: {row['from_grace_still_in_grace']}")
            print(f"    → Progressed: {row['from_grace_progressed']}")
            print(f"  From Late 16-30:")
            print(f"    → Cured: {row['from_late16_cured']}")
            print(f"    → Still in Late 16-30: {row['from_late16_still_in_late16']}")
            print(f"    → Progressed: {row['from_late16_progressed']}")
            print(f"  From Late 31-120:")
            print(f"    → Still in Late 31-120: {row['from_late31_still_in_late31']}")
            print(f"    → Charged Off: {row['from_late31_charged_off']}")

    return results_df


def _calculate_transition_flow(df: pd.DataFrame, strata_value: any) -> dict:
    """
    Helper function to calculate transition flow for a specific group.
    """

    metrics = {
        'strata_value': strata_value,
        'total_loans': len(df)
    }

    total_loans = len(df)

    # =========================================================================
    # FROM CURRENT (100% of loans start here)
    # =========================================================================

    # Clean Fully Paid: Fully Paid with no late fees
    fully_paid_clean = len(df[(df['loan_status'] == 'Fully Paid') & (df['curr_paid_late1_flag'] == 0)])

    # Clean Current: Current with no late fees
    current_clean = len(df[(df['loan_status'] == 'Current') & (df['curr_paid_late1_flag'] == 0)])

    # Delinquent: Everyone else (Grace + Late 16-30 + Late 31-120 + Charged Off + cured with flag)
    delinquent = total_loans - fully_paid_clean - current_clean

    metrics['from_current_to_fully_paid_clean'] = fully_paid_clean / total_loans if total_loans > 0 else 0
    metrics['from_current_to_current_clean'] = current_clean / total_loans if total_loans > 0 else 0
    metrics['from_current_to_delinquent'] = delinquent / total_loans if total_loans > 0 else 0

    # =========================================================================
    # FROM GRACE PERIOD (those who became delinquent)
    # =========================================================================

    # Total that reached Grace = all delinquent loans
    reached_grace = delinquent

    # Still in Grace
    still_in_grace = len(df[df['loan_status'] == 'In Grace Period'])

    # Progressed beyond Grace = Late 16-30 + Late 31-120 + Charged Off + cured (flag=1)
    progressed_from_grace = reached_grace - still_in_grace

    metrics['from_grace_still_in_grace'] = still_in_grace / reached_grace if reached_grace > 0 else 0
    metrics['from_grace_progressed'] = progressed_from_grace / reached_grace if reached_grace > 0 else 0

    # =========================================================================
    # FROM LATE 16-30 (those who progressed beyond Grace)
    # =========================================================================

    # Total that reached Late 16-30 = curr_paid_late1_flag=1 + Late 16-30 + Late 31-120 + Charged Off
    cured_from_late16 = len(df[df['curr_paid_late1_flag'] == 1])
    currently_late16 = len(df[df['loan_status'] == 'Late (16-30 days)'])
    currently_late31 = len(df[df['loan_status'] == 'Late (31-120 days)'])
    charged_off = len(df[df['loan_status'] == 'Charged Off'])

    reached_late16 = cured_from_late16 + currently_late16 + currently_late31 + charged_off

    # Still in Late 16-30
    still_in_late16 = currently_late16

    # Progressed beyond Late 16-30 = Late 31-120 + Charged Off
    progressed_from_late16 = currently_late31 + charged_off

    metrics['from_late16_cured'] = cured_from_late16 / reached_late16 if reached_late16 > 0 else 0
    metrics['from_late16_still_in_late16'] = still_in_late16 / reached_late16 if reached_late16 > 0 else 0
    metrics['from_late16_progressed'] = progressed_from_late16 / reached_late16 if reached_late16 > 0 else 0

    # =========================================================================
    # FROM LATE 31-120 (those who progressed beyond Late 16-30)
    # =========================================================================

    # Total that reached Late 31-120 = Late 31-120 + Charged Off
    reached_late31 = currently_late31 + charged_off

    # Still in Late 31-120
    still_in_late31 = currently_late31

    # Charged Off
    metrics['from_late31_still_in_late31'] = still_in_late31 / reached_late31 if reached_late31 > 0 else 0
    metrics['from_late31_charged_off'] = charged_off / reached_late31 if reached_late31 > 0 else 0

    return metrics


# ===========================================================================
# NEW DISPLAY FUNCTIONS — Backsolve Transitions, Age-Weighted Matrix,
# Default Timing, Loan Age Status Distribution
# ===========================================================================

import logging

logger = logging.getLogger(__name__)

# Snapshot date for the Lending Club dataset
SNAPSHOT_DATE = pd.Timestamp('2019-03-01')

# States used in the age-weighted transition matrix
TRANSITION_STATES = [
    'Current',
    'Delinquent (0-30)',
    'Late (31-120)',
    'Charged Off',
    'Fully Paid',
]


def reconstruct_loan_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each loan, estimate key transition months using backsolve logic.

    Uses last_pymnt_d and loan_status to reconstruct when each loan
    entered delinquency, late status, default, or payoff. This is the
    foundation for the age-weighted transition matrix and default timing.

    Parameters
    ----------
    df : pd.DataFrame
        Loan data with columns: loan_status, issue_d, last_pymnt_d,
        curr_paid_late1_flag, funded_amnt, total_rec_prncp, out_prncp.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns: loan_age_months, age_bucket,
        age_bucket_label, delinquent_month, late_31_120_month, default_month,
        payoff_month, delinquent_age, late_31_120_age, default_age,
        payoff_age, cured_from_late.
    """
    df = df.copy()

    # Ensure datetime columns
    if not pd.api.types.is_datetime64_any_dtype(df['issue_d']):
        df['issue_d'] = pd.to_datetime(df['issue_d'])
    if not pd.api.types.is_datetime64_any_dtype(df['last_pymnt_d']):
        df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'])

    # Loan age at snapshot (months since origination), capped at term - 1
    # A 36-month loan caps at age 35 (month 0 through 35 = 36 months).
    # This keeps all ages within clean 6-month buckets (0-5 through 54-59).
    raw_age = (
        (SNAPSHOT_DATE - df['issue_d']).dt.days / 30.44
    ).round().astype(int)
    if 'term_months' in df.columns:
        df['loan_age_months'] = np.minimum(raw_age, df['term_months'].astype(int) - 1)
    else:
        df['loan_age_months'] = raw_age

    # 6-month age bucket for grouping
    df['age_bucket'] = (df['loan_age_months'] // 6) * 6
    df['age_bucket_label'] = df['age_bucket'].apply(lambda x: f"{x}-{x + 5}")

    # Initialize transition columns as NaT / NaN
    for col in ['delinquent_month', 'late_31_120_month', 'default_month', 'payoff_month']:
        df[col] = pd.NaT
    for col in ['delinquent_age', 'late_31_120_age', 'default_age', 'payoff_age']:
        df[col] = np.nan

    # === Delinquent / Charged Off loans ===
    delinquent_statuses = [
        'In Grace Period', 'Late (16-30 days)',
        'Late (31-120 days)', 'Charged Off',
    ]
    delinq_mask = df['loan_status'].isin(delinquent_statuses)

    if delinq_mask.any():
        # The month they missed payment (1 month after last payment)
        df.loc[delinq_mask, 'delinquent_month'] = (
            df.loc[delinq_mask, 'last_pymnt_d'] + pd.DateOffset(months=1)
        )

        # 31+ days past due begins the following month
        df.loc[delinq_mask, 'late_31_120_month'] = (
            df.loc[delinq_mask, 'delinquent_month'] + pd.DateOffset(months=1)
        )

    # Default at ~120 days (4 months after first missed) — terminal
    charged_off_mask = df['loan_status'] == 'Charged Off'
    if charged_off_mask.any():
        df.loc[charged_off_mask, 'default_month'] = (
            df.loc[charged_off_mask, 'delinquent_month'] + pd.DateOffset(months=4)
        )

    # === Fully Paid loans ===
    fully_paid_mask = df['loan_status'] == 'Fully Paid'
    if fully_paid_mask.any():
        df.loc[fully_paid_mask, 'payoff_month'] = df.loc[fully_paid_mask, 'last_pymnt_d']

    # === Loan age at each transition (capped at term_months - 1) ===
    term_cap = df['term_months'].astype(int) - 1 if 'term_months' in df.columns else None
    for col in ['delinquent_month', 'late_31_120_month', 'default_month', 'payoff_month']:
        age_col = col.replace('_month', '_age')
        mask = df[col].notna()
        if mask.any():
            raw = (
                (df.loc[mask, col] - df.loc[mask, 'issue_d']).dt.days / 30.44
            ).round().astype(int)
            if term_cap is not None:
                raw = np.minimum(raw, term_cap.loc[mask])
            df.loc[mask, age_col] = raw

    # === Cured loans ===
    df['cured_from_late'] = (
        (df['curr_paid_late1_flag'] == 1)
        & (df['loan_status'].isin(['Current', 'Fully Paid']))
    )

    return df


def get_loan_status_at_age(df: pd.DataFrame, age: int) -> pd.Series:
    """
    Return each loan's estimated status at a specific loan age.

    Uses the reconstructed timeline columns from reconstruct_loan_timeline()
    to determine what state each loan was in at the given number of months
    since origination.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with reconstructed timeline columns (loan_age_months,
        delinquent_age, late_31_120_age, default_age, payoff_age).
    age : int
        Loan age in months to evaluate.

    Returns
    -------
    pd.Series
        Status strings ('Current', 'Delinquent (0-30)', 'Late (31-120)',
        'Charged Off', 'Fully Paid') or None for loans not yet at this age.
    """
    # Start with all None (loans not yet originated at this age)
    result = pd.Series(None, index=df.index, dtype=object)

    # Only evaluate loans that existed at this age
    valid = df['loan_age_months'] >= age

    # Default to Current for valid loans
    result[valid] = 'Current'

    # Apply transitions in reverse priority order (last assignment wins)
    # so higher-priority states overwrite lower-priority ones

    # Delinquent (0-30): at delinquent_age (lasts 1 month before progressing)
    delinq = valid & df['delinquent_age'].notna() & (age >= df['delinquent_age'])
    result[delinq] = 'Delinquent (0-30)'

    # Late (31-120): from late_31_120_age onward (until default or snapshot)
    late = valid & df['late_31_120_age'].notna() & (age >= df['late_31_120_age'])
    result[late] = 'Late (31-120)'

    # Fully Paid: from payoff_age onward (absorbing)
    paid = valid & df['payoff_age'].notna() & (age >= df['payoff_age'])
    result[paid] = 'Fully Paid'

    # Charged Off: from default_age onward (absorbing, highest priority)
    default = valid & df['default_age'].notna() & (age >= df['default_age'])
    result[default] = 'Charged Off'

    return result


def compute_age_transition_probabilities(
    df: pd.DataFrame,
    bucket_size: int = 6,
) -> pd.DataFrame:
    """
    Compute empirical transition probabilities at each loan age bucket.

    For each consecutive pair of ages (N, N+1), determines the from→to
    status transition for every loan, then aggregates into age buckets
    and normalizes to probabilities.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with reconstructed timelines (from reconstruct_loan_timeline).
        Should include ALL loans in the cohort (historical, not just current).
    bucket_size : int
        Months per age bucket (default 6).

    Returns
    -------
    pd.DataFrame
        Columns: age_bucket, from_status, to_current_pct,
        to_delinquent_0_30_pct, to_late_31_120_pct, to_charged_off_pct,
        to_fully_paid_pct, observation_count.
    """
    max_age = int(df['loan_age_months'].max())

    # Collect all transition observations
    transitions = []

    # Cache: compute status at age 0, then iterate
    prev_status = get_loan_status_at_age(df, 0)

    for age_n in range(0, max_age):
        next_status = get_loan_status_at_age(df, age_n + 1)

        # Only count loans that were valid (not None) at both ages
        valid = prev_status.notna() & next_status.notna()
        if valid.sum() == 0:
            prev_status = next_status
            continue

        # Determine age bucket for this age N
        age_bucket = (age_n // bucket_size) * bucket_size
        bucket_label = f"{age_bucket}-{age_bucket + bucket_size - 1}"

        from_vals = prev_status[valid]
        to_vals = next_status[valid]

        # Group by (from, to) and count
        pair_counts = pd.DataFrame({
            'from_status': from_vals,
            'to_status': to_vals,
        }).groupby(['from_status', 'to_status']).size().reset_index(name='count')
        pair_counts['age_bucket'] = bucket_label

        transitions.append(pair_counts)

        # Reuse for next iteration
        prev_status = next_status

    if not transitions:
        return pd.DataFrame(columns=[
            'age_bucket', 'from_status',
            'to_current_pct', 'to_delinquent_0_30_pct',
            'to_late_31_120_pct', 'to_charged_off_pct',
            'to_fully_paid_pct', 'observation_count',
        ])

    all_trans = pd.concat(transitions, ignore_index=True)

    # Aggregate by age_bucket, from_status, to_status
    agg = (
        all_trans
        .groupby(['age_bucket', 'from_status', 'to_status'])['count']
        .sum()
        .reset_index()
    )

    # Pivot to get one row per (age_bucket, from_status)
    pivot = agg.pivot_table(
        index=['age_bucket', 'from_status'],
        columns='to_status',
        values='count',
        fill_value=0,
    ).reset_index()

    # Flatten column names
    pivot.columns.name = None

    # Ensure all target states exist as columns
    for state in TRANSITION_STATES:
        if state not in pivot.columns:
            pivot[state] = 0

    # Compute total observations per row
    state_cols = [c for c in TRANSITION_STATES if c in pivot.columns]
    pivot['observation_count'] = pivot[state_cols].sum(axis=1)

    # Normalize to probabilities
    col_map = {
        'Current': 'to_current_pct',
        'Delinquent (0-30)': 'to_delinquent_0_30_pct',
        'Late (31-120)': 'to_late_31_120_pct',
        'Charged Off': 'to_charged_off_pct',
        'Fully Paid': 'to_fully_paid_pct',
    }
    for state, pct_col in col_map.items():
        if state in pivot.columns:
            pivot[pct_col] = pivot[state] / pivot['observation_count']
        else:
            pivot[pct_col] = 0.0

    # Select and order output columns
    result = pivot[
        ['age_bucket', 'from_status']
        + list(col_map.values())
        + ['observation_count']
    ].copy()

    # Sort by bucket then from_status
    bucket_order = sorted(result['age_bucket'].unique(),
                          key=lambda x: int(x.split('-')[0]))
    result['_bucket_sort'] = result['age_bucket'].map(
        {b: i for i, b in enumerate(bucket_order)}
    )
    status_order = {s: i for i, s in enumerate(TRANSITION_STATES)}
    result['_status_sort'] = result['from_status'].map(status_order)
    result = (
        result
        .sort_values(['_bucket_sort', '_status_sort'])
        .drop(columns=['_bucket_sort', '_status_sort'])
        .reset_index(drop=True)
    )

    return result


def compute_pool_transition_matrix(
    df_current: pd.DataFrame,
    age_probs: pd.DataFrame,
) -> dict:
    """
    Apply age-specific transition probabilities to the current pool's UPB.

    Produces a single aggregate dollar-flow matrix showing expected
    next-month flows for Current loans, weighted by each age bucket's UPB.

    Parameters
    ----------
    df_current : pd.DataFrame
        Current loans with March 2019 last payment, with
        reconstruct_loan_timeline() columns applied.
    age_probs : pd.DataFrame
        Output from compute_age_transition_probabilities().

    Returns
    -------
    dict
        Keys: 'aggregate_matrix' ($ flows), 'aggregate_matrix_pct'
        (percentage flows), 'breakdown_by_age' (DataFrame).
    """
    # Get transition probs for Current from_status only
    current_probs = age_probs[age_probs['from_status'] == 'Current'].copy()

    # UPB per age bucket in the current pool
    bucket_upb = (
        df_current
        .groupby('age_bucket_label')['out_prncp']
        .sum()
        .reset_index()
        .rename(columns={'out_prncp': 'upb'})
    )

    pct_cols = [
        'to_current_pct', 'to_delinquent_0_30_pct',
        'to_late_31_120_pct', 'to_charged_off_pct', 'to_fully_paid_pct',
    ]
    dollar_cols = [
        'to_current_$', 'to_delinquent_$',
        'to_late_31_120_$', 'to_charged_off_$', 'to_fully_paid_$',
    ]

    # Merge UPB with transition probabilities
    breakdown = bucket_upb.merge(
        current_probs[['age_bucket'] + pct_cols],
        left_on='age_bucket_label',
        right_on='age_bucket',
        how='left',
    )

    # Fill missing age buckets (no historical data) with 0 transition probs
    for col in pct_cols:
        breakdown[col] = breakdown[col].fillna(0)

    # Compute dollar flows per bucket
    for pct, dollar in zip(pct_cols, dollar_cols):
        breakdown[dollar] = breakdown['upb'] * breakdown[pct]

    # Sort by age bucket
    breakdown = breakdown.sort_values(
        'age_bucket_label',
        key=lambda s: s.str.extract(r'(\d+)', expand=False).astype(int),
    ).reset_index(drop=True)

    # Clean up columns for output
    breakdown_out = breakdown[
        ['age_bucket_label', 'upb'] + dollar_cols + pct_cols
    ].copy()
    # Rename pct cols for display
    breakdown_out = breakdown_out.rename(columns={
        'to_current_pct': 'to_current_rate',
        'to_delinquent_0_30_pct': 'to_delinquent_rate',
        'to_late_31_120_pct': 'to_late_31_120_rate',
        'to_charged_off_pct': 'to_charged_off_rate',
        'to_fully_paid_pct': 'to_fully_paid_rate',
    })

    # Aggregate: sum dollar flows across all buckets
    agg_dollars = {}
    state_names = ['Current', 'Delinquent (0-30)', 'Late (31-120)',
                   'Charged Off', 'Fully Paid']
    for dollar_col, state in zip(dollar_cols, state_names):
        agg_dollars[state] = breakdown[dollar_col].sum()

    aggregate_matrix = {'Current': agg_dollars}

    # Also include reference rows for other states from age_probs
    for from_state in TRANSITION_STATES:
        if from_state == 'Current':
            continue
        state_probs = age_probs[age_probs['from_status'] == from_state]
        if len(state_probs) == 0:
            continue
        # Weighted average probabilities across all buckets (for reference)
        avg_probs = {}
        for pct, state in zip(pct_cols, state_names):
            if pct in state_probs.columns:
                total_obs = state_probs['observation_count'].sum()
                if total_obs > 0:
                    avg_probs[state] = (
                        (state_probs[pct] * state_probs['observation_count']).sum()
                        / total_obs
                    )
                else:
                    avg_probs[state] = 0.0
            else:
                avg_probs[state] = 0.0
        aggregate_matrix[from_state] = avg_probs

    # Compute percentage version
    aggregate_matrix_pct = {}
    for from_state, flows in aggregate_matrix.items():
        total = sum(flows.values())
        if total > 0:
            aggregate_matrix_pct[from_state] = {
                to_state: v / total for to_state, v in flows.items()
            }
        else:
            aggregate_matrix_pct[from_state] = {
                to_state: 0.0 for to_state in flows
            }

    return {
        'aggregate_matrix': aggregate_matrix,
        'aggregate_matrix_pct': aggregate_matrix_pct,
        'breakdown_by_age': breakdown_out,
    }


def compute_default_timing(
    df: pd.DataFrame,
    group_col: str = None,
) -> pd.DataFrame:
    """
    Build a default timing distribution by loan age at default.

    Shows at what loan age (months since origination) defaults occur,
    useful for understanding seasoning patterns and peak default timing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with reconstruct_loan_timeline() columns applied.
    group_col : str, optional
        Column to group by (e.g., 'grade', 'term_months'). If None,
        computes a single aggregate distribution.

    Returns
    -------
    pd.DataFrame
        Columns: default_age_months, loan_count, upb_amount,
        pct_of_defaults, cumulative_pct. If group_col is provided,
        an additional column with the group value.
    """
    # Filter to Charged Off loans with valid default_age
    charged_off = df[
        (df['loan_status'] == 'Charged Off') & df['default_age'].notna()
    ].copy()

    if len(charged_off) == 0:
        cols = ['default_age_months', 'loan_count', 'upb_amount',
                'pct_of_defaults', 'cumulative_pct']
        if group_col:
            cols.insert(0, group_col)
        return pd.DataFrame(columns=cols)

    # Compute exposure (defaulted UPB)
    charged_off['_default_upb'] = (
        charged_off['funded_amnt'] - charged_off['total_rec_prncp']
    ).clip(lower=0)

    charged_off['default_age_months'] = charged_off['default_age'].astype(int)

    group_keys = [group_col, 'default_age_months'] if group_col else ['default_age_months']

    agg = (
        charged_off
        .groupby(group_keys)
        .agg(
            loan_count=('default_age_months', 'size'),
            upb_amount=('_default_upb', 'sum'),
        )
        .reset_index()
        .sort_values(group_keys)
    )

    # Compute pct_of_defaults and cumulative within each group
    if group_col:
        group_totals = agg.groupby(group_col)['loan_count'].transform('sum')
        agg['pct_of_defaults'] = agg['loan_count'] / group_totals
        agg['cumulative_pct'] = agg.groupby(group_col)['pct_of_defaults'].cumsum()
    else:
        total = agg['loan_count'].sum()
        agg['pct_of_defaults'] = agg['loan_count'] / total
        agg['cumulative_pct'] = agg['pct_of_defaults'].cumsum()

    return agg.reset_index(drop=True)


def compute_loan_age_status_matrix(
    df: pd.DataFrame,
    bucket_size: int = 6,
) -> pd.DataFrame:
    """
    Cross-sectional snapshot: status distribution at each loan age bucket.

    For loans at each age bucket (as of the snapshot date), shows what
    percentage are Current, Fully Paid, Charged Off, or Late/Grace.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with loan_age_months column (from reconstruct_loan_timeline).
    bucket_size : int
        Months per age bucket (default 6). Set to 1 for monthly.

    Returns
    -------
    pd.DataFrame
        Columns: age_bucket, total_loans, total_upb, pct_current,
        pct_fully_paid, pct_charged_off, pct_late_grace.
    """
    # Extract only the columns we need to avoid pandas consolidation errors
    # on the large enriched DataFrame with mixed dtypes
    status_map = {
        'Current': 'current',
        'Fully Paid': 'fully_paid',
        'Charged Off': 'charged_off',
        'In Grace Period': 'late_grace',
        'Late (16-30 days)': 'late_grace',
        'Late (31-120 days)': 'late_grace',
    }

    slim = pd.DataFrame({
        '_age_group': df['age_bucket_label'] if bucket_size > 1 else df['loan_age_months'],
        '_sort': df['age_bucket'] if bucket_size > 1 else df['loan_age_months'],
        '_status_cat': df['loan_status'].map(status_map).fillna('other'),
        '_upb': df['out_prncp'] if 'out_prncp' in df.columns else 0,
    })

    # Group by age bucket
    grouped = slim.groupby('_age_group')

    result_rows = []
    for name, group in grouped:
        total = len(group)
        total_upb = group['_upb'].sum()

        counts = group['_status_cat'].value_counts()
        sort_val = group['_sort'].iloc[0]

        result_rows.append({
            'age_bucket': name,
            '_sort': sort_val,
            'total_loans': total,
            'total_upb': total_upb,
            'pct_current': counts.get('current', 0) / total,
            'pct_fully_paid': counts.get('fully_paid', 0) / total,
            'pct_charged_off': counts.get('charged_off', 0) / total,
            'pct_late_grace': counts.get('late_grace', 0) / total,
        })

    result = pd.DataFrame(result_rows)
    if len(result) > 0:
        result = result.sort_values('_sort').drop(columns='_sort').reset_index(drop=True)

    return result
