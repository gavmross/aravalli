"""Tests for src/portfolio_analytics.py"""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_analytics import (
    calculate_credit_metrics,
    _calculate_metrics_for_group,
    calculate_performance_metrics,
    calculate_transition_matrix,
    _calculate_transition_flow,
    reconstruct_loan_timeline,
    get_loan_status_at_age,
    compute_age_transition_probabilities,
    compute_pool_transition_matrix,
    compute_default_timing,
    compute_loan_age_status_matrix,
    TRANSITION_STATES_7,
)


@pytest.fixture
def synthetic_loan_df():
    """
    Create a synthetic loan DataFrame with all required columns for both
    credit metrics and performance metrics.

    Pool: 10 loans across 2 grades (A, B) and 2 vintages (2018Q1, 2018Q2).
    Statuses: Current (4), Fully Paid (3), Charged Off (2), In Grace Period (1).
    """
    np.random.seed(42)
    n = 10

    df = pd.DataFrame({
        # Identifiers & terms
        'funded_amnt': [10000, 15000, 20000, 12000, 8000, 25000, 18000, 9000, 11000, 14000],
        'int_rate': [0.07, 0.09, 0.11, 0.08, 0.06, 0.12, 0.10, 0.07, 0.09, 0.08],
        'term_months': [36, 36, 60, 36, 36, 60, 60, 36, 36, 60],
        'out_prncp': [5000, 0, 12000, 6000, 0, 0, 10000, 0, 5500, 7000],

        # Statuses
        'loan_status': [
            'Current', 'Fully Paid', 'Current', 'Current',
            'Fully Paid', 'Charged Off', 'Current', 'Charged Off',
            'In Grace Period', 'Fully Paid'
        ],

        # Strata
        'grade': ['A', 'A', 'B', 'A', 'A', 'B', 'B', 'B', 'A', 'B'],
        'issue_quarter': ['2018Q1', '2018Q1', '2018Q1', '2018Q2', '2018Q2',
                          '2018Q1', '2018Q2', '2018Q2', '2018Q1', '2018Q2'],

        # Credit scores & DTI
        'original_fico': [720, 740, 680, 710, 750, 660, 690, 700, 730, 695],
        'latest_fico': [725, 0, 685, 715, 0, 0, 695, 0, 735, 0],
        'dti_clean': [18.0, 15.0, 22.0, 19.0, 12.0, 25.0, 20.0, 16.0, 17.0, 21.0],

        # Principal & recovery data
        'total_rec_prncp': [5000, 15000, 8000, 6000, 8000, 20000, 8000, 7000, 5500, 14000],
        'recoveries': [0, 0, 0, 0, 0, 2000, 0, 800, 0, 0],

        # Amortization columns (from calc_amort)
        'updated_remaining_term': [18, 0, 36, 20, 0, 0, 30, 0, 17, 0],
        'orig_exp_payments_made': [18, 36, 24, 16, 36, 24, 30, 36, 19, 36],

        # Last payment analysis columns (from calc_amort)
        'last_pmt_beginning_balance': [5200, 350, 12500, 6300, 400, 0, 10400, 0, 5700, 500],
        'last_pmt_scheduled_principal': [200, 350, 180, 220, 400, 0, 200, 0, 210, 500],
        'last_pmt_unscheduled_principal': [50, 0, 100, 30, 0, 0, 80, 0, 40, 0],
        'orig_exp_principal_paid': [4800, 15000, 7500, 5800, 8000, 20000, 7500, 7000, 5200, 14000],

        # Transition matrix flag
        'curr_paid_late1_flag': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],

        # Last payment date (Current=Mar 2019, FP/CO=earlier, IGP=Feb 2019)
        'last_pymnt_d': [
            '2019-03-01', '2018-12-01', '2019-03-01', '2019-03-01',
            '2018-11-01', '2018-06-01', '2019-03-01', '2018-08-01',
            '2019-02-01', '2018-10-01'
        ],
    })

    return df


class TestCalculateCreditMetrics:
    """Test credit metrics calculation."""

    def test_returns_dataframe(self, synthetic_loan_df):
        """Should return a DataFrame."""
        result = calculate_credit_metrics(synthetic_loan_df, 'grade', verbose=False)
        assert isinstance(result, pd.DataFrame)

    def test_has_all_row(self, synthetic_loan_df):
        """First row should be the ALL aggregate."""
        result = calculate_credit_metrics(synthetic_loan_df, 'grade', verbose=False)
        assert result.iloc[0]['strata_value'] == 'ALL'

    def test_has_strata_rows(self, synthetic_loan_df):
        """Should have rows for each unique grade + ALL."""
        result = calculate_credit_metrics(synthetic_loan_df, 'grade', verbose=False)
        assert len(result) == 3  # ALL + A + B

    def test_expected_columns(self, synthetic_loan_df):
        """Should contain all expected output columns."""
        result = calculate_credit_metrics(synthetic_loan_df, 'grade', verbose=False)
        expected = [
            'strata_type', 'strata_value',
            'orig_total_upb_mm', 'orig_loan_count', 'orig_wac', 'orig_wam',
            'orig_avg_fico', 'orig_avg_dti',
            'active_total_upb_mm', 'active_upb_current_perc', 'active_upb_grace_perc',
            'active_upb_late_16_30_perc', 'active_upb_late_31_120_perc',
            'curr_wac', 'curr_wam', 'curr_wala', 'curr_avg_fico', 'curr_avg_dti',
            'upb_fully_paid_perc', 'upb_lost_perc'
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_loan_count_all(self, synthetic_loan_df):
        """ALL row should have total loan count."""
        result = calculate_credit_metrics(synthetic_loan_df, 'grade', verbose=False)
        all_row = result[result['strata_value'] == 'ALL'].iloc[0]
        assert all_row['orig_loan_count'] == 10

    def test_missing_column_raises(self, synthetic_loan_df):
        """Missing required column should raise ValueError."""
        df = synthetic_loan_df.drop(columns=['funded_amnt'])
        with pytest.raises(ValueError, match="Missing required columns"):
            calculate_credit_metrics(df, 'grade', verbose=False)

    def test_invalid_strata_raises(self, synthetic_loan_df):
        """Invalid strata column should raise ValueError."""
        with pytest.raises(ValueError, match="Strata column"):
            calculate_credit_metrics(synthetic_loan_df, 'nonexistent_col', verbose=False)

    def test_wac_is_weighted(self, synthetic_loan_df):
        """WAC should be weighted by funded_amnt, not simple average."""
        result = calculate_credit_metrics(synthetic_loan_df, 'grade', verbose=False)
        all_row = result[result['strata_value'] == 'ALL'].iloc[0]
        # Manually compute weighted average
        weights = synthetic_loan_df['funded_amnt'].values
        expected_wac = np.average(synthetic_loan_df['int_rate'].values, weights=weights)
        assert abs(all_row['orig_wac'] - expected_wac) < 1e-6

    def test_active_upb_percentages_sum_to_one(self, synthetic_loan_df):
        """Active UPB percentage breakdown should sum to ~1.0."""
        result = calculate_credit_metrics(synthetic_loan_df, 'grade', verbose=False)
        all_row = result[result['strata_value'] == 'ALL'].iloc[0]
        total_pct = (all_row['active_upb_current_perc'] +
                     all_row['active_upb_grace_perc'] +
                     all_row['active_upb_late_16_30_perc'] +
                     all_row['active_upb_late_31_120_perc'])
        assert abs(total_pct - 1.0) < 0.01


class TestCalculatePerformanceMetrics:
    """Test performance metrics calculation."""

    def test_returns_dataframe(self, synthetic_loan_df):
        """Should return a DataFrame."""
        result = calculate_performance_metrics(synthetic_loan_df, verbose=False)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_vintage(self, synthetic_loan_df):
        """Should have one row per vintage."""
        result = calculate_performance_metrics(synthetic_loan_df, verbose=False)
        assert len(result) == 2  # 2018Q1 and 2018Q2

    def test_expected_columns(self, synthetic_loan_df):
        """Should contain all expected output columns."""
        result = calculate_performance_metrics(synthetic_loan_df, verbose=False)
        expected = [
            'vintage', 'orig_loan_count', 'orig_upb_mm',
            'pct_active', 'pct_current', 'pct_fully_paid', 'pct_charged_off',
            'pct_defaulted_count', 'pct_defaulted_upb',
            'pool_cpr', 'pct_prepaid',
            'loss_severity', 'recovery_rate'
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_status_percentages_sum_close_to_one(self, synthetic_loan_df):
        """Status percentages should sum to ~1 (accounting for other statuses)."""
        result = calculate_performance_metrics(synthetic_loan_df, verbose=False)
        for _, row in result.iterrows():
            # pct_active includes Current + Grace + Late, pct_fully_paid, pct_charged_off
            # Together they should cover all (or most) loans
            total = row['pct_active'] + row['pct_fully_paid'] + row['pct_charged_off']
            assert total <= 1.01  # Allow small rounding

    def test_loss_severity_between_zero_and_one(self, synthetic_loan_df):
        """Loss severity should be between 0 and 1."""
        result = calculate_performance_metrics(synthetic_loan_df, verbose=False)
        for _, row in result.iterrows():
            assert 0 <= row['loss_severity'] <= 1.0

    def test_recovery_rate_between_zero_and_one(self, synthetic_loan_df):
        """Recovery rate should be between 0 and 1."""
        result = calculate_performance_metrics(synthetic_loan_df, verbose=False)
        for _, row in result.iterrows():
            assert 0 <= row['recovery_rate'] <= 1.0

    def test_loss_plus_recovery_equals_one(self, synthetic_loan_df):
        """Loss severity + recovery rate should equal ~1 for vintages with charged off loans."""
        result = calculate_performance_metrics(synthetic_loan_df, verbose=False)
        for _, row in result.iterrows():
            if row['loss_severity'] > 0:
                assert abs(row['loss_severity'] + row['recovery_rate'] - 1.0) < 0.01

    def test_missing_column_raises(self, synthetic_loan_df):
        """Missing required column should raise ValueError."""
        df = synthetic_loan_df.drop(columns=['funded_amnt'])
        with pytest.raises(ValueError, match="Missing required columns"):
            calculate_performance_metrics(df, verbose=False)

    def test_cpr_non_negative(self, synthetic_loan_df):
        """CPR values should be non-negative."""
        result = calculate_performance_metrics(synthetic_loan_df, verbose=False)
        for _, row in result.iterrows():
            assert row['pool_cpr'] >= 0


class TestCalculateTransitionMatrix:
    """Test transition matrix calculation."""

    def test_returns_dataframe(self, synthetic_loan_df):
        """Should return a DataFrame."""
        result = calculate_transition_matrix(synthetic_loan_df, verbose=False)
        assert isinstance(result, pd.DataFrame)

    def test_all_only_when_no_strata(self, synthetic_loan_df):
        """Without strata_col, should have only ALL row."""
        result = calculate_transition_matrix(synthetic_loan_df, strata_col=None, verbose=False)
        assert len(result) == 1
        assert result.iloc[0]['strata_value'] == 'ALL'

    def test_with_strata(self, synthetic_loan_df):
        """With strata_col, should have ALL + one per strata value."""
        result = calculate_transition_matrix(synthetic_loan_df, strata_col='grade', verbose=False)
        assert len(result) == 3  # ALL + A + B

    def test_expected_columns(self, synthetic_loan_df):
        """Should contain all expected output columns."""
        result = calculate_transition_matrix(synthetic_loan_df, verbose=False)
        expected = [
            'strata_value', 'total_loans',
            'from_current_to_fully_paid_clean', 'from_current_to_current_clean',
            'from_current_to_delinquent',
            'from_grace_still_in_grace', 'from_grace_progressed',
            'from_late16_cured', 'from_late16_still_in_late16', 'from_late16_progressed',
            'from_late31_still_in_late31', 'from_late31_charged_off'
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_current_flows_sum_to_one(self, synthetic_loan_df):
        """From-Current flows should sum to 1.0."""
        result = calculate_transition_matrix(synthetic_loan_df, verbose=False)
        row = result.iloc[0]
        total = (row['from_current_to_fully_paid_clean'] +
                 row['from_current_to_current_clean'] +
                 row['from_current_to_delinquent'])
        assert abs(total - 1.0) < 0.01

    def test_grace_flows_sum_to_one(self, synthetic_loan_df):
        """From-Grace flows should sum to 1.0 (if any loans reached grace)."""
        result = calculate_transition_matrix(synthetic_loan_df, verbose=False)
        row = result.iloc[0]
        if row['from_current_to_delinquent'] > 0:
            total = row['from_grace_still_in_grace'] + row['from_grace_progressed']
            assert abs(total - 1.0) < 0.01

    def test_missing_flag_raises(self, synthetic_loan_df):
        """Missing curr_paid_late1_flag should raise ValueError."""
        df = synthetic_loan_df.drop(columns=['curr_paid_late1_flag'])
        with pytest.raises(ValueError, match="curr_paid_late1_flag"):
            calculate_transition_matrix(df, verbose=False)

    def test_total_loans_correct(self, synthetic_loan_df):
        """Total loans should match input DataFrame length."""
        result = calculate_transition_matrix(synthetic_loan_df, verbose=False)
        assert result.iloc[0]['total_loans'] == len(synthetic_loan_df)


class TestCalculateTransitionFlow:
    """Test the helper transition flow function directly."""

    def test_all_current_no_delinquency(self):
        """All Current loans with no flags → 0% delinquent."""
        df = pd.DataFrame({
            'loan_status': ['Current'] * 5,
            'curr_paid_late1_flag': [0] * 5,
        })
        result = _calculate_transition_flow(df, 'ALL')
        assert result['from_current_to_current_clean'] == 1.0
        assert result['from_current_to_delinquent'] == 0.0

    def test_all_charged_off(self):
        """All Charged Off → 100% delinquent, all progressed through."""
        df = pd.DataFrame({
            'loan_status': ['Charged Off'] * 5,
            'curr_paid_late1_flag': [0] * 5,
        })
        result = _calculate_transition_flow(df, 'ALL')
        assert result['from_current_to_delinquent'] == 1.0
        assert result['from_late31_charged_off'] == 1.0

    def test_empty_df(self):
        """Empty DataFrame should return zeros."""
        df = pd.DataFrame({
            'loan_status': pd.Series([], dtype=str),
            'curr_paid_late1_flag': pd.Series([], dtype=int),
        })
        result = _calculate_transition_flow(df, 'ALL')
        assert result['total_loans'] == 0
        assert result['from_current_to_delinquent'] == 0


# ===========================================================================
# Tests for NEW display functions
# ===========================================================================

@pytest.fixture
def timeline_loan_df():
    """
    Create a synthetic dataset with known timeline properties.

    10 loans with specific statuses and dates for testing backsolve logic.
    """
    return pd.DataFrame({
        'id': range(1, 11),
        'loan_status': [
            'Current',             # 1: performing, March 2019 payment
            'Fully Paid',          # 2: paid off June 2018
            'Charged Off',         # 3: defaulted (last_pymnt Oct 2018)
            'In Grace Period',     # 4: just went delinquent
            'Late (31-120 days)',  # 5: deep delinquency
            'Current',             # 6: performing, March 2019
            'Fully Paid',          # 7: paid off Dec 2018
            'Charged Off',         # 8: early default
            'Current',             # 9: performing with cure flag
            'Late (16-30 days)',   # 10: recently late
        ],
        'issue_d': pd.to_datetime([
            '2017-01-01', '2017-01-01', '2017-01-01', '2018-06-01',
            '2017-06-01', '2018-01-01', '2016-01-01', '2016-06-01',
            '2017-06-01', '2018-03-01',
        ]),
        'last_pymnt_d': pd.to_datetime([
            '2019-03-01', '2018-06-01', '2018-10-01', '2019-02-01',
            '2018-11-01', '2019-03-01', '2018-12-01', '2017-06-01',
            '2019-03-01', '2019-01-01',
        ]),
        'funded_amnt': [10000, 15000, 20000, 12000, 8000,
                        25000, 18000, 9000, 11000, 14000],
        'total_rec_prncp': [5000, 15000, 12000, 6000, 4000,
                            12000, 18000, 3000, 5500, 7000],
        'out_prncp': [5000, 0, 0, 6000, 4000,
                      13000, 0, 0, 5500, 7000],
        'curr_paid_late1_flag': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        'grade': ['A', 'A', 'B', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
        'term_months': [36, 36, 60, 36, 60, 36, 60, 36, 36, 36],
    })


class TestReconstructLoanTimeline:
    """Test loan timeline reconstruction via backsolve logic."""

    def test_returns_dataframe_with_expected_columns(self, timeline_loan_df):
        """Should add all expected timeline columns."""
        result = reconstruct_loan_timeline(timeline_loan_df)
        expected_cols = [
            'loan_age_months', 'age_bucket', 'age_bucket_label',
            'delinquent_month', 'late_31_120_month', 'default_month',
            'payoff_month', 'delinquent_age', 'late_31_120_age',
            'default_age', 'payoff_age', 'cured_from_late',
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_charged_off_loan_transitions(self, timeline_loan_df):
        """Charged Off loan with last_pymnt_d=2018-10-01, issue_d=2017-01-01."""
        result = reconstruct_loan_timeline(timeline_loan_df)
        # Loan 3: Charged Off, last_pymnt_d=2018-10-01, issue_d=2017-01-01
        loan = result[result['id'] == 3].iloc[0]

        # delinquent_month = 2018-11-01
        assert loan['delinquent_month'] == pd.Timestamp('2018-11-01')
        # late_31_120_month = 2018-12-01
        assert loan['late_31_120_month'] == pd.Timestamp('2018-12-01')
        # default_month = 2019-03-01 (4 months after delinquent)
        assert loan['default_month'] == pd.Timestamp('2019-03-01')
        # delinquent_age ≈ 22 months
        assert abs(loan['delinquent_age'] - 22) <= 1
        # default_age ≈ 26 months
        assert abs(loan['default_age'] - 26) <= 1

    def test_current_loan_no_transitions(self, timeline_loan_df):
        """Current loan with March 2019 payment → all transition cols NaT."""
        result = reconstruct_loan_timeline(timeline_loan_df)
        loan = result[result['id'] == 1].iloc[0]

        assert pd.isna(loan['delinquent_month'])
        assert pd.isna(loan['late_31_120_month'])
        assert pd.isna(loan['default_month'])
        assert pd.isna(loan['payoff_month'])

    def test_grace_period_loan(self, timeline_loan_df):
        """Grace Period loan with last_pymnt_d=2019-02-01."""
        result = reconstruct_loan_timeline(timeline_loan_df)
        loan = result[result['id'] == 4].iloc[0]

        # delinquent_month = 2019-03-01
        assert loan['delinquent_month'] == pd.Timestamp('2019-03-01')
        # No default_month (not Charged Off)
        assert pd.isna(loan['default_month'])

    def test_fully_paid_loan(self, timeline_loan_df):
        """Fully Paid loan → payoff_month equals last_pymnt_d."""
        result = reconstruct_loan_timeline(timeline_loan_df)
        loan = result[result['id'] == 2].iloc[0]

        assert loan['payoff_month'] == pd.Timestamp('2018-06-01')
        assert pd.isna(loan['delinquent_month'])

    def test_cured_from_late_flag(self, timeline_loan_df):
        """cured_from_late should be True only for flag=1 AND Current/Fully Paid."""
        result = reconstruct_loan_timeline(timeline_loan_df)

        # Loan 9: curr_paid_late1_flag=1, status=Current → True
        assert bool(result[result['id'] == 9].iloc[0]['cured_from_late']) is True

        # All others should be False
        for lid in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
            assert bool(result[result['id'] == lid].iloc[0]['cured_from_late']) is False

    def test_age_bucket_label(self, timeline_loan_df):
        """Loan age 14 → bucket "12-17"."""
        result = reconstruct_loan_timeline(timeline_loan_df)
        # Loan 6: issue_d=2018-01-01, snapshot=2019-03-01 → age ≈ 14 months
        loan = result[result['id'] == 6].iloc[0]
        assert loan['age_bucket_label'] == '12-17'


class TestGetLoanStatusAtAge:
    """Test status determination at specific loan ages."""

    def test_charged_off_loan_timeline(self, timeline_loan_df):
        """Charged Off loan should show correct status at each age."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        loan_3 = df[df['id'] == 3]

        # At age 21: should still be Current
        status_21 = get_loan_status_at_age(loan_3, 21)
        assert status_21.iloc[0] == 'Current'

        # At delinquent_age (≈22): Delinquent
        delinq_age = int(loan_3.iloc[0]['delinquent_age'])
        status_d = get_loan_status_at_age(loan_3, delinq_age)
        assert status_d.iloc[0] == 'Delinquent (0-30)'

        # At late_31_120_age (≈23): Late (31-120)
        late_age = int(loan_3.iloc[0]['late_31_120_age'])
        status_l = get_loan_status_at_age(loan_3, late_age)
        assert status_l.iloc[0] == 'Late (31-120)'

        # At default_age (≈26): Charged Off
        default_age = int(loan_3.iloc[0]['default_age'])
        status_co = get_loan_status_at_age(loan_3, default_age)
        assert status_co.iloc[0] == 'Charged Off'

    def test_fully_paid_loan(self, timeline_loan_df):
        """Fully Paid loan at payoff age and beyond should be Fully Paid."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        loan_2 = df[df['id'] == 2]

        payoff_age = int(loan_2.iloc[0]['payoff_age'])

        # Before payoff: Current
        if payoff_age > 0:
            status_before = get_loan_status_at_age(loan_2, payoff_age - 1)
            assert status_before.iloc[0] == 'Current'

        # At payoff: Fully Paid
        status_at = get_loan_status_at_age(loan_2, payoff_age)
        assert status_at.iloc[0] == 'Fully Paid'

        # After payoff: still Fully Paid (absorbing)
        status_after = get_loan_status_at_age(loan_2, payoff_age + 5)
        assert status_after.iloc[0] == 'Fully Paid'

    def test_current_loan_stays_current(self, timeline_loan_df):
        """Current loan should be Current at all valid ages."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        loan_1 = df[df['id'] == 1]

        for age in [0, 5, 10, 15, 20]:
            status = get_loan_status_at_age(loan_1, age)
            assert status.iloc[0] == 'Current', f"Expected Current at age {age}"

    def test_none_for_future_ages(self, timeline_loan_df):
        """Loan not yet at given age → None."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        loan_6 = df[df['id'] == 6]  # issue_d=2018-01-01, age ≈ 14

        status = get_loan_status_at_age(loan_6, 100)
        assert pd.isna(status.iloc[0]) or status.iloc[0] is None


class TestComputeAgeTransitionProbabilities:
    """Test age-bucketed transition probabilities."""

    def test_row_sums_to_one(self, timeline_loan_df):
        """Each row of probabilities should sum to ~1.0."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        result = compute_age_transition_probabilities(df, bucket_size=6)

        pct_cols = [
            'to_current_pct', 'to_delinquent_0_30_pct',
            'to_late_31_120_pct', 'to_charged_off_pct', 'to_fully_paid_pct',
        ]
        for _, row in result.iterrows():
            total = sum(row[c] for c in pct_cols)
            assert abs(total - 1.0) < 0.01, (
                f"Row sums to {total} for bucket={row['age_bucket']}, "
                f"from={row['from_status']}"
            )

    def test_absorbing_states(self, timeline_loan_df):
        """Charged Off and Fully Paid should stay at 100%."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        result = compute_age_transition_probabilities(df, bucket_size=6)

        co_rows = result[result['from_status'] == 'Charged Off']
        for _, row in co_rows.iterrows():
            assert abs(row['to_charged_off_pct'] - 1.0) < 0.01

        fp_rows = result[result['from_status'] == 'Fully Paid']
        for _, row in fp_rows.iterrows():
            assert abs(row['to_fully_paid_pct'] - 1.0) < 0.01

    def test_no_skip_states(self, timeline_loan_df):
        """Current → Charged Off should be 0% (cannot skip states)."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        result = compute_age_transition_probabilities(df, bucket_size=6)

        current_rows = result[result['from_status'] == 'Current']
        for _, row in current_rows.iterrows():
            assert row['to_charged_off_pct'] == 0.0, (
                f"Current→Charged Off = {row['to_charged_off_pct']} "
                f"at bucket {row['age_bucket']}"
            )

    def test_late_cannot_cure(self, timeline_loan_df):
        """Late (31-120) → Current should be 0%."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        result = compute_age_transition_probabilities(df, bucket_size=6)

        late_rows = result[result['from_status'] == 'Late (31-120)']
        for _, row in late_rows.iterrows():
            assert row['to_current_pct'] == 0.0, (
                f"Late→Current = {row['to_current_pct']} "
                f"at bucket {row['age_bucket']}"
            )

    def test_returns_expected_columns(self, timeline_loan_df):
        """Output should have all expected columns."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        result = compute_age_transition_probabilities(df, bucket_size=6)
        expected = [
            'age_bucket', 'from_status', 'to_current_pct',
            'to_delinquent_0_30_pct', 'to_late_31_120_pct',
            'to_charged_off_pct', 'to_fully_paid_pct', 'observation_count',
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"


class TestComputePoolTransitionMatrix:
    """Test aggregate dollar-flow transition matrix."""

    def test_breakdown_upb_sums_to_total(self, timeline_loan_df):
        """breakdown_by_age UPB should sum to total current pool UPB."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        df_current = df[
            (df['loan_status'] == 'Current')
            & (df['last_pymnt_d'] == '2019-03-01')
        ]
        age_probs = compute_age_transition_probabilities(df, bucket_size=6)
        result = compute_pool_transition_matrix(df_current, age_probs)

        expected_upb = df_current['out_prncp'].sum()
        actual_upb = result['breakdown_by_age']['upb'].sum()
        assert abs(actual_upb - expected_upb) < 1.0

    def test_current_to_skip_states_zero(self, timeline_loan_df):
        """Current → Late(31-120) and Current → Charged Off should be $0."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        df_current = df[
            (df['loan_status'] == 'Current')
            & (df['last_pymnt_d'] == '2019-03-01')
        ]
        age_probs = compute_age_transition_probabilities(df, bucket_size=6)
        result = compute_pool_transition_matrix(df_current, age_probs)

        agg = result['aggregate_matrix']['Current']
        assert agg.get('Late (31-120)', 0) == 0.0
        assert agg.get('Charged Off', 0) == 0.0

    def test_aggregate_row_sums(self, timeline_loan_df):
        """Dollar flows from Current should sum to total pool UPB."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        df_current = df[
            (df['loan_status'] == 'Current')
            & (df['last_pymnt_d'] == '2019-03-01')
        ]
        age_probs = compute_age_transition_probabilities(df, bucket_size=6)
        result = compute_pool_transition_matrix(df_current, age_probs)

        agg = result['aggregate_matrix']['Current']
        total_flow = sum(agg.values())
        expected = df_current['out_prncp'].sum()
        # Should be close (within rounding)
        assert abs(total_flow - expected) / expected < 0.01

    def test_synthetic_two_buckets(self):
        """Verify dollar flows with two synthetic age buckets and known rates."""
        # Create a minimal age_probs with known rates
        age_probs = pd.DataFrame([
            {'age_bucket': '0-5', 'from_status': 'Current',
             'to_current_pct': 0.95, 'to_delinquent_0_30_pct': 0.03,
             'to_late_31_120_pct': 0.0, 'to_charged_off_pct': 0.0,
             'to_fully_paid_pct': 0.02, 'observation_count': 100},
            {'age_bucket': '6-11', 'from_status': 'Current',
             'to_current_pct': 0.90, 'to_delinquent_0_30_pct': 0.05,
             'to_late_31_120_pct': 0.0, 'to_charged_off_pct': 0.0,
             'to_fully_paid_pct': 0.05, 'observation_count': 100},
        ])

        # Create synthetic current loans
        df_current = pd.DataFrame({
            'loan_status': ['Current'] * 4,
            'out_prncp': [50_000_000, 50_000_000, 40_000_000, 60_000_000],
            'age_bucket_label': ['0-5', '0-5', '6-11', '6-11'],
            'issue_d': pd.to_datetime(['2019-01-01'] * 4),
            'last_pymnt_d': pd.to_datetime(['2019-03-01'] * 4),
            'loan_age_months': [2, 2, 8, 8],
        })

        result = compute_pool_transition_matrix(df_current, age_probs)
        agg = result['aggregate_matrix']['Current']

        # Bucket A (0-5): $100M × 3% delinquent = $3M
        # Bucket B (6-11): $100M × 5% delinquent = $5M
        # Total delinquent: $8M
        assert abs(agg['Delinquent (0-30)'] - 8_000_000) < 1.0


class TestComputeDefaultTiming:
    """Test default timing distribution."""

    def test_pct_sums_to_one(self, timeline_loan_df):
        """pct_of_defaults should sum to 1.0."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        result = compute_default_timing(df)

        if len(result) > 0:
            total = result['pct_of_defaults'].sum()
            assert abs(total - 1.0) < 0.01

    def test_cumulative_reaches_one(self, timeline_loan_df):
        """cumulative_pct should reach 1.0 at the last age."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        result = compute_default_timing(df)

        if len(result) > 0:
            assert abs(result['cumulative_pct'].iloc[-1] - 1.0) < 0.01

    def test_grouped_distributions_sum_to_one(self, timeline_loan_df):
        """Each group's pct_of_defaults should sum to 1.0."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        result = compute_default_timing(df, group_col='grade')

        if len(result) > 0:
            for group_val, group_df in result.groupby('grade'):
                total = group_df['pct_of_defaults'].sum()
                assert abs(total - 1.0) < 0.01, (
                    f"Group {group_val} sums to {total}"
                )

    def test_empty_if_no_defaults(self):
        """No Charged Off loans → empty DataFrame."""
        df = pd.DataFrame({
            'loan_status': ['Current', 'Fully Paid'],
            'issue_d': pd.to_datetime(['2017-01-01', '2017-06-01']),
            'last_pymnt_d': pd.to_datetime(['2019-03-01', '2018-12-01']),
            'funded_amnt': [10000, 15000],
            'total_rec_prncp': [5000, 15000],
            'out_prncp': [5000, 0],
            'curr_paid_late1_flag': [0, 0],
        })
        df = reconstruct_loan_timeline(df)
        result = compute_default_timing(df)
        assert len(result) == 0


class TestComputeLoanAgeStatusMatrix:
    """Test cross-sectional loan age status distribution."""

    def test_pct_sums_to_one(self, timeline_loan_df):
        """pct_current + pct_fully_paid + pct_charged_off + pct_late_grace ≈ 1.0."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        result = compute_loan_age_status_matrix(df, bucket_size=6)

        for _, row in result.iterrows():
            total = (row['pct_current'] + row['pct_fully_paid']
                     + row['pct_charged_off'] + row['pct_late_grace'])
            assert abs(total - 1.0) < 0.01, (
                f"Bucket {row['age_bucket']} sums to {total}"
            )

    def test_bucket_labels_6month(self, timeline_loan_df):
        """bucket_size=6 should produce labels like '0-5', '6-11'."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        result = compute_loan_age_status_matrix(df, bucket_size=6)

        for label in result['age_bucket']:
            parts = str(label).split('-')
            assert len(parts) == 2
            start, end = int(parts[0]), int(parts[1])
            assert end - start == 5

    def test_monthly_granularity(self, timeline_loan_df):
        """bucket_size=1 should produce monthly rows."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        result = compute_loan_age_status_matrix(df, bucket_size=1)

        # Should have more rows than 6-month version
        result_6 = compute_loan_age_status_matrix(df, bucket_size=6)
        assert len(result) >= len(result_6)

    def test_total_loans_correct(self, timeline_loan_df):
        """Sum of total_loans across all buckets should equal total loans."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        result = compute_loan_age_status_matrix(df, bucket_size=6)

        assert result['total_loans'].sum() == len(df)


# ===========================================================================
# Tests for 7-STATE model extensions
# ===========================================================================

class TestReconstructLoanTimelineLateSubStates:
    """Test Late sub-state columns added by reconstruct_loan_timeline."""

    def test_late_sub_state_columns_exist(self, timeline_loan_df):
        """Should add late_1/2/3_month and late_1/2/3_age columns."""
        result = reconstruct_loan_timeline(timeline_loan_df)
        for col in ['late_1_month', 'late_2_month', 'late_3_month',
                     'late_1_age', 'late_2_age', 'late_3_age']:
            assert col in result.columns, f"Missing column: {col}"

    def test_charged_off_loan_late_sub_states(self, timeline_loan_df):
        """Charged Off loan 3: late_1 = late_31_120, late_2 = +1mo, late_3 = +2mo."""
        result = reconstruct_loan_timeline(timeline_loan_df)
        loan = result[result['id'] == 3].iloc[0]

        # late_31_120_month = 2018-12-01 (delinquent + 1 month)
        assert loan['late_1_month'] == loan['late_31_120_month']
        assert loan['late_2_month'] == loan['late_31_120_month'] + pd.DateOffset(months=1)
        assert loan['late_3_month'] == loan['late_31_120_month'] + pd.DateOffset(months=2)

    def test_late_sub_state_ages_computed(self, timeline_loan_df):
        """Late sub-state ages should be consecutive months."""
        result = reconstruct_loan_timeline(timeline_loan_df)
        loan = result[result['id'] == 3].iloc[0]

        # Ages should be roughly consecutive
        assert not np.isnan(loan['late_1_age'])
        assert not np.isnan(loan['late_2_age'])
        assert not np.isnan(loan['late_3_age'])
        assert abs(loan['late_2_age'] - loan['late_1_age'] - 1) <= 1
        assert abs(loan['late_3_age'] - loan['late_2_age'] - 1) <= 1

    def test_current_loan_no_late_sub_states(self, timeline_loan_df):
        """Current loan should have NaT/NaN for all late sub-state columns."""
        result = reconstruct_loan_timeline(timeline_loan_df)
        loan = result[result['id'] == 1].iloc[0]

        assert pd.isna(loan['late_1_month'])
        assert pd.isna(loan['late_2_month'])
        assert pd.isna(loan['late_3_month'])
        assert np.isnan(loan['late_1_age'])

    def test_late_31_120_loan_has_sub_states(self, timeline_loan_df):
        """Late (31-120 days) loan 5 should have late sub-state months set."""
        result = reconstruct_loan_timeline(timeline_loan_df)
        loan = result[result['id'] == 5].iloc[0]

        # Loan 5 is Late (31-120 days) → it reached late_31_120
        assert pd.notna(loan['late_1_month'])


class TestGetLoanStatusAtAge7State:
    """Test 7-state status determination."""

    def test_charged_off_loan_7state_pipeline(self, timeline_loan_df):
        """Charged Off loan should show Late_1/Late_2/Late_3 at correct ages."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        loan_3 = df[df['id'] == 3]

        late_1_age = int(loan_3.iloc[0]['late_1_age'])
        late_2_age = int(loan_3.iloc[0]['late_2_age'])
        late_3_age = int(loan_3.iloc[0]['late_3_age'])
        default_age = int(loan_3.iloc[0]['default_age'])

        status_l1 = get_loan_status_at_age(loan_3, late_1_age, states='7state')
        assert status_l1.iloc[0] == 'Late_1'

        status_l2 = get_loan_status_at_age(loan_3, late_2_age, states='7state')
        assert status_l2.iloc[0] == 'Late_2'

        status_l3 = get_loan_status_at_age(loan_3, late_3_age, states='7state')
        assert status_l3.iloc[0] == 'Late_3'

        status_co = get_loan_status_at_age(loan_3, default_age, states='7state')
        assert status_co.iloc[0] == 'Charged Off'

    def test_delinquent_only_at_exact_age(self, timeline_loan_df):
        """In 7-state, Delinquent (0-30) should only appear at exactly delinquent_age."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        loan_3 = df[df['id'] == 3]

        delinq_age = int(loan_3.iloc[0]['delinquent_age'])

        # At delinquent_age: Delinquent
        status = get_loan_status_at_age(loan_3, delinq_age, states='7state')
        assert status.iloc[0] == 'Delinquent (0-30)'

        # One month before: Current
        if delinq_age > 0:
            status_before = get_loan_status_at_age(
                loan_3, delinq_age - 1, states='7state')
            assert status_before.iloc[0] == 'Current'

    def test_5state_backward_compat(self, timeline_loan_df):
        """states='5state' should produce same results as original function."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        loan_3 = df[df['id'] == 3]

        delinq_age = int(loan_3.iloc[0]['delinquent_age'])
        status_5 = get_loan_status_at_age(loan_3, delinq_age, states='5state')
        assert status_5.iloc[0] == 'Delinquent (0-30)'

    def test_current_stays_current_7state(self, timeline_loan_df):
        """Current loan should be Current at all valid ages in 7-state."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        loan_1 = df[df['id'] == 1]

        for age in [0, 5, 10, 15, 20]:
            status = get_loan_status_at_age(loan_1, age, states='7state')
            assert status.iloc[0] == 'Current', f"Expected Current at age {age}"


class TestComputeAgeTransitionProbabilities7State:
    """Test 7-state age-bucketed transition probabilities."""

    def test_7state_columns(self, timeline_loan_df):
        """7-state output should have all 7-state pct columns."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        result = compute_age_transition_probabilities(
            df, bucket_size=1, states='7state')

        expected_cols = [
            'to_current_pct', 'to_delinquent_0_30_pct',
            'to_late_1_pct', 'to_late_2_pct', 'to_late_3_pct',
            'to_charged_off_pct', 'to_fully_paid_pct',
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_7state_row_sums_to_one(self, timeline_loan_df):
        """Each row of 7-state probabilities should sum to ~1.0."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        result = compute_age_transition_probabilities(
            df, bucket_size=1, states='7state')

        pct_cols = [
            'to_current_pct', 'to_delinquent_0_30_pct',
            'to_late_1_pct', 'to_late_2_pct', 'to_late_3_pct',
            'to_charged_off_pct', 'to_fully_paid_pct',
        ]
        for _, row in result.iterrows():
            total = sum(row[c] for c in pct_cols)
            assert abs(total - 1.0) < 0.01, (
                f"Row sums to {total} for bucket={row['age_bucket']}, "
                f"from={row['from_status']}"
            )

    def test_7state_no_skip(self, timeline_loan_df):
        """Current → Late_1/Late_2/Late_3/Charged Off should be 0%."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        result = compute_age_transition_probabilities(
            df, bucket_size=1, states='7state')

        current_rows = result[result['from_status'] == 'Current']
        for _, row in current_rows.iterrows():
            assert row['to_late_1_pct'] == 0.0, (
                f"Current→Late_1 = {row['to_late_1_pct']} at age {row['age_bucket']}")
            assert row['to_late_2_pct'] == 0.0
            assert row['to_late_3_pct'] == 0.0
            assert row['to_charged_off_pct'] == 0.0

    def test_7state_absorbing_states(self, timeline_loan_df):
        """Charged Off and Fully Paid should stay at 100% in 7-state."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        result = compute_age_transition_probabilities(
            df, bucket_size=6, states='7state')

        co_rows = result[result['from_status'] == 'Charged Off']
        for _, row in co_rows.iterrows():
            assert abs(row['to_charged_off_pct'] - 1.0) < 0.01

        fp_rows = result[result['from_status'] == 'Fully Paid']
        for _, row in fp_rows.iterrows():
            assert abs(row['to_fully_paid_pct'] - 1.0) < 0.01

    def test_monthly_ages_more_granular(self, timeline_loan_df):
        """bucket_size=1 should produce more age buckets than bucket_size=6."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        result_monthly = compute_age_transition_probabilities(
            df, bucket_size=1, states='7state')
        result_6mo = compute_age_transition_probabilities(
            df, bucket_size=6, states='7state')

        monthly_buckets = result_monthly['age_bucket'].nunique()
        six_mo_buckets = result_6mo['age_bucket'].nunique()
        assert monthly_buckets >= six_mo_buckets

    def test_5state_backward_compat(self, timeline_loan_df):
        """states='5state' should produce same columns as original."""
        df = reconstruct_loan_timeline(timeline_loan_df)
        result = compute_age_transition_probabilities(
            df, bucket_size=6, states='5state')

        expected_cols = [
            'to_current_pct', 'to_delinquent_0_30_pct',
            'to_late_31_120_pct', 'to_charged_off_pct', 'to_fully_paid_pct',
        ]
        for col in expected_cols:
            assert col in result.columns
