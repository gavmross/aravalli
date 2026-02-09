"""
Amortization calculations for loan portfolio analysis.

Extracted from scripts/analysis.ipynb — DO NOT MODIFY LOGIC.
Only imports were added.
"""

import numpy as np
import pandas as pd


def calc_monthly_payment(principal: np.ndarray,
                         annual_rate: np.ndarray,
                         term_months: np.ndarray) -> np.ndarray:
    """
    Vectorized calculation of fixed monthly payment for fully amortizing loans.

    Formula: PMT = P * [r(1+r)^n] / [(1+r)^n - 1]
    """
    monthly_rate = annual_rate / 12.0

    # Handle zero interest rate
    zero_rate_mask = monthly_rate == 0

    # Calculate for non-zero rates
    numerator = monthly_rate * (1 + monthly_rate) ** term_months
    denominator = (1 + monthly_rate) ** term_months - 1

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        payment = principal * (numerator / denominator)

    # For zero rates, payment is simple division
    payment[zero_rate_mask] = principal[zero_rate_mask] / term_months[zero_rate_mask]

    return payment


def calc_balance(principal: np.ndarray,
                 annual_rate: np.ndarray,
                 monthly_payment: np.ndarray,
                 payments_made: np.ndarray) -> tuple:
    """
    Vectorized calculation of remaining balance after specified payments.
    """
    monthly_rate = annual_rate / 12.0

    # Initialize arrays (ensure float64)
    remaining_balance = principal.copy().astype(np.float64)
    total_principal_paid = np.zeros_like(principal, dtype=np.float64)
    total_interest_paid = np.zeros_like(principal, dtype=np.float64)

    # For loans with no payments, return initial values
    no_payments_mask = payments_made == 0

    # For loans with payments, calculate iteratively
    max_payments = int(np.max(payments_made)) if len(payments_made) > 0 else 0

    for payment_num in range(1, max_payments + 1):
        # Mask for loans that have made at least this many payments
        active_mask = payments_made >= payment_num

        if not active_mask.any():
            break

        # Calculate interest for this period
        interest = remaining_balance * monthly_rate

        # Calculate principal payment
        principal_pmt = monthly_payment - interest

        # Handle final payment (may be less than regular payment)
        principal_pmt = np.minimum(principal_pmt, remaining_balance)

        # Update only for active loans
        remaining_balance[active_mask] -= principal_pmt[active_mask]
        total_principal_paid[active_mask] += principal_pmt[active_mask]
        total_interest_paid[active_mask] += interest[active_mask]

    # Ensure no negative balances
    remaining_balance = np.maximum(remaining_balance, 0.0)

    return remaining_balance, total_principal_paid, total_interest_paid


def calc_payment_num(start_dates: pd.Series,
                     end_dates: pd.Series) -> np.ndarray:
    """
    Vectorized calculation of months between two date series.
    """
    # Calculate month difference
    months = (end_dates.dt.year - start_dates.dt.year) * 12 + \
             (end_dates.dt.month - start_dates.dt.month)

    # Convert to numpy and handle NaN/negative
    months = months.fillna(0).values
    months = np.maximum(months, 0)

    return months.astype(int)


def calc_amort(df: pd.DataFrame,
               principal_col: str = 'funded_amnt',
               rate_col: str = 'int_rate',
               term_col: str = 'term_months',
               issue_date_col: str = 'issue_d',
               as_of_date_col: str = 'last_pymnt_d',
               verbose: bool = False) -> pd.DataFrame:
    """
    Calculate expected monthly payment and balance as of last_pymnt_d for each loan.

    OPTIMIZED for large datasets (2M+ loans) using vectorized operations.

    Returns:
    --------
    df_with_schedule : pd.DataFrame
        Original dataframe with added columns (all rounded to 2 decimals):

        ORIGINAL AMORTIZATION (orig_exp_*):
        - orig_exp_monthly_payment: Expected fixed monthly payment from origination
        - orig_exp_payments_made: Number of payments made
        - orig_exp_balance: Expected remaining balance
        - orig_exp_principal_paid: Expected cumulative principal paid
        - orig_exp_interest_paid: Expected cumulative interest paid
        - orig_exp_monthly_principal: Expected principal portion at last_pymnt_d
        - orig_exp_monthly_interest: Expected interest portion at last_pymnt_d
        - orig_exp_balance_diff: Actual out_prncp - orig_exp_balance
        - orig_exp_installment_diff: installment - orig_exp_monthly_payment

        LAST PAYMENT ANALYSIS (last_pmt_*):
        - last_pmt_beginning_balance: UPB before last payment
        - last_pmt_interest: Interest portion of last payment
        - last_pmt_actual_principal: Actual principal paid in last payment
        - last_pmt_scheduled_principal: Expected principal (from installment)
        - last_pmt_unscheduled_principal: Prepayment amount
        - last_pmt_smm: Single Monthly Mortality
        - last_pmt_cpr: Conditional Prepayment Rate (annualized)

        NEXT PAYMENT PROJECTION (next_pmt_*):
        - next_pmt_principal: Principal portion of next payment
        - next_pmt_interest: Interest portion of next payment
        - updated_remaining_term: Remaining months based on current balance
        - updated_maturity_date: Projected maturity date
    """

    df_sched = df.copy()

    # Validate required columns
    required_cols = [principal_col, rate_col, term_col, issue_date_col, as_of_date_col]
    missing_cols = [col for col in required_cols if col not in df_sched.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Convert dates to datetime if needed
    if df_sched[issue_date_col].dtype == 'object':
        df_sched[issue_date_col] = pd.to_datetime(df_sched[issue_date_col], errors='coerce')
    if df_sched[as_of_date_col].dtype == 'object':
        df_sched[as_of_date_col] = pd.to_datetime(df_sched[as_of_date_col], errors='coerce')

    # Check if rate is percentage or decimal and convert if needed
    sample_rate = df_sched[rate_col].dropna().iloc[0] if len(df_sched) > 0 else 0
    if sample_rate > 1:
        df_sched[rate_col] = df_sched[rate_col] / 100.0

    # Extract numpy arrays for vectorized operations
    principal = df_sched[principal_col].values
    annual_rate = df_sched[rate_col].fillna(0).values
    term_months = df_sched[term_col].fillna(0).values.astype(int)
    monthly_rate = annual_rate / 12.0

    # =========================================================================
    # PART 1: ORIGINAL AMORTIZATION (exp_* columns)
    # =========================================================================

    # Step 1: Calculate expected monthly payment (vectorized)
    monthly_payment = calc_monthly_payment(principal, annual_rate, term_months)
    df_sched['orig_exp_monthly_payment'] = np.round(monthly_payment, 2)

    # Step 2: Calculate number of payments made (vectorized)
    payments_made = calc_payment_num(
        df_sched[issue_date_col],
        df_sched[as_of_date_col]
    )
    payments_made = np.minimum(payments_made, term_months)
    df_sched['orig_exp_payments_made'] = payments_made

    # Step 3: Calculate expected balance (vectorized)
    remaining_balance, principal_paid, interest_paid = calc_balance(
        principal, annual_rate, monthly_payment, payments_made
    )

    df_sched['orig_exp_balance'] = np.round(remaining_balance, 2)
    df_sched['orig_exp_principal_paid'] = np.round(principal_paid, 2)
    df_sched['orig_exp_interest_paid'] = np.round(interest_paid, 2)

    # Step 4: Calculate principal/interest split for last payment made
    has_payments_mask = payments_made > 0

    balance_before_last, _, _ = calc_balance(
        principal, annual_rate, monthly_payment, np.maximum(payments_made - 1, 0)
    )

    exp_monthly_interest = balance_before_last * monthly_rate
    exp_monthly_principal = monthly_payment - exp_monthly_interest
    exp_monthly_principal = np.minimum(exp_monthly_principal, balance_before_last)
    exp_monthly_interest = monthly_payment - exp_monthly_principal

    exp_monthly_principal[~has_payments_mask] = 0.0
    exp_monthly_interest[~has_payments_mask] = 0.0

    df_sched['orig_exp_monthly_principal'] = np.round(exp_monthly_principal, 2)
    df_sched['orig_exp_monthly_interest'] = np.round(exp_monthly_interest, 2)

    # Step 5: Calculate differences
    if 'out_prncp' in df_sched.columns:
        df_sched['orig_exp_balance_diff'] = np.round(
            df_sched['out_prncp'] - df_sched['orig_exp_balance'], 2
        )

    if 'installment' in df_sched.columns:
        df_sched['orig_exp_installment_diff'] = np.round(
            df_sched['installment'] - df_sched['orig_exp_monthly_payment'], 2
        )

    # =========================================================================
    # PART 2: LAST PAYMENT ANALYSIS (last_pmt_* columns)
    # Using actual data: out_prncp, last_pymnt_amnt, installment, int_rate
    # =========================================================================

    if 'out_prncp' in df_sched.columns and 'last_pymnt_amnt' in df_sched.columns and 'installment' in df_sched.columns:

        current_upb = df_sched['out_prncp'].values
        last_payment = df_sched['last_pymnt_amnt'].values
        installment = df_sched['installment'].values

        # Has payment mask
        has_last_payment = (last_payment > 0) & (payments_made > 0)

        # Calculate beginning balance (before last payment)
        # Formula: beginning_balance = (current_upb + last_payment) / (1 + monthly_rate)
        with np.errstate(divide='ignore', invalid='ignore'):
            beginning_balance = (current_upb + last_payment) / (1 + monthly_rate)

        df_sched['last_pmt_beginning_balance'] = np.round(beginning_balance, 2)

        # Calculate interest portion of last payment
        last_pmt_interest = beginning_balance * monthly_rate
        df_sched['last_pmt_interest'] = np.round(last_pmt_interest, 2)

        # Calculate actual principal paid in last payment
        last_pmt_actual_principal = last_payment - last_pmt_interest
        df_sched['last_pmt_actual_principal'] = np.round(last_pmt_actual_principal, 2)

        # Calculate scheduled principal (from installment)
        last_pmt_scheduled_principal = installment - last_pmt_interest
        df_sched['last_pmt_scheduled_principal'] = np.round(last_pmt_scheduled_principal, 2)

        # Calculate unscheduled principal (prepayment)
        last_pmt_unscheduled_principal = last_pmt_actual_principal - last_pmt_scheduled_principal
        df_sched['last_pmt_unscheduled_principal'] = np.round(last_pmt_unscheduled_principal, 2)

        # Calculate SMM (Single Monthly Mortality)
        # SMM = unscheduled_principal / (beginning_balance - scheduled_principal)
        with np.errstate(divide='ignore', invalid='ignore'):
            denominator = beginning_balance - last_pmt_scheduled_principal
            smm = last_pmt_unscheduled_principal / denominator
            smm = np.where(denominator > 0, smm, 0)
            smm = np.where(has_last_payment, smm, 0)

        df_sched['last_pmt_smm'] = np.round(smm, 6)

        # Calculate CPR (Conditional Prepayment Rate)
        # CPR = 1 - (1 - SMM)^12
        cpr = 1 - (1 - smm) ** 12
        cpr = np.where(has_last_payment, cpr, 0)

        df_sched['last_pmt_cpr'] = np.round(cpr, 6)

    # =========================================================================
    # PART 3: NEXT PAYMENT PROJECTION (next_pmt_* columns)
    # Using current balance and installment to project forward
    # =========================================================================

    if 'out_prncp' in df_sched.columns and 'installment' in df_sched.columns:

        current_balance = df_sched['out_prncp'].values
        actual_payment = df_sched['installment'].values

        has_balance_mask = current_balance > 0

        # Calculate interest for next payment
        next_pmt_interest = current_balance * monthly_rate

        # Calculate principal for next payment
        next_pmt_principal = actual_payment - next_pmt_interest
        next_pmt_principal = np.minimum(next_pmt_principal, current_balance)

        # For loans with no balance, set to zero
        next_pmt_principal[~has_balance_mask] = 0.0
        next_pmt_interest[~has_balance_mask] = 0.0

        df_sched['next_pmt_principal'] = np.round(next_pmt_principal, 2)
        df_sched['next_pmt_interest'] = np.round(next_pmt_interest, 2)

        # Calculate remaining term
        # Formula: n = -ln(1 - r*P/PMT) / ln(1+r)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = (monthly_rate * current_balance) / actual_payment
            ratio = np.minimum(ratio, 0.9999)  # Prevent issues when payment < interest

            remaining_term = -np.log(1 - ratio) / np.log(1 + monthly_rate)

            # Handle zero rate case
            zero_rate_mask = monthly_rate == 0
            remaining_term[zero_rate_mask] = current_balance[zero_rate_mask] / actual_payment[zero_rate_mask]

            # Round up to nearest whole month
            remaining_term = np.ceil(remaining_term)

        # Set to 0 for loans with no balance
        remaining_term[~has_balance_mask] = 0

        # Cap at reasonable maximum
        max_reasonable_term = 600  # 50 years
        remaining_term = np.minimum(remaining_term, max_reasonable_term)

        df_sched['updated_remaining_term'] = remaining_term.astype(int)

        # Calculate updated maturity date using period addition
        df_sched['updated_maturity_date'] = (
            df_sched[as_of_date_col].dt.to_period("M")
            + df_sched['updated_remaining_term']
        ).dt.to_timestamp()

    if verbose:
        print(f"Calculated amortization schedule for {len(df_sched):,} loans")
        print(f"Mean expected payment: ${df_sched['orig_exp_monthly_payment'].mean():.2f}")
        print(f"Total expected balance: ${df_sched['orig_exp_balance'].sum():,.2f}")

    return df_sched
