"""Tests for src/cashflow_engine.py"""

import numpy as np
import numpy_financial as npf
import pandas as pd
import pytest

from src.cashflow_engine import (
    compute_pool_assumptions,
    compute_pool_characteristics,
    project_cashflows,
    calculate_irr,
    solve_price,
    adjust_prepayment_rates,
    build_pool_state,
    project_cashflows_transition,
    solve_price_transition,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df_all():
    """
    Synthetic dataset for compute_pool_assumptions with conditional CDR.

    1000 loans originated Jan 2017 (all 36-month term, 10% rate, $10K each).
    In each of the 12 trailing months (Apr 2018 – Mar 2019), 5 loans default.
    60 total defaults, rest are Current. No payoffs in this fixture.

    Expected:
    - Each month: default_upb = 5 * ($10K - $0 rec_prncp) = $50K
    - Performing balance ≈ ~990 loans at ~$8K est. balance each ≈ ~$8M
    - avg_MDR ≈ $50K / ~$8M ≈ 0.00625
    - CDR = 1 - (1 - 0.00625)^12 ≈ 7.24%
    """
    n_total = 1000
    n_defaults_per_month = 5
    n_defaults = n_defaults_per_month * 12  # 60

    funded = [10000.0] * n_total
    rec_prncp = [0.0] * n_total  # simplify: no principal recovered for defaults
    int_rate = [0.10] * n_total
    term_months = [36] * n_total
    issue_d = ['2017-01-01'] * n_total
    recoveries = [0.0] * n_total

    # out_prncp: Charged Off loans = 0, Current loans ≈ scheduled balance
    # Setting Current loans to ~$3100 (close to amortized balance at 26 months)
    # so prepayment adjustment is minimal and existing test bounds still hold.
    out_prncp = [0.0] * n_defaults + [3100.0] * (n_total - n_defaults)

    # Statuses: first 60 are Charged Off (5 per month × 12 months), rest Current
    loan_status = ['Charged Off'] * n_defaults + ['Current'] * (n_total - n_defaults)

    # Assign default_month: 5 loans per month from Apr 2018 to Mar 2019
    default_months = []
    snapshot = pd.Timestamp('2019-03-01')
    for m in range(12, 0, -1):  # m=12 -> Apr 2018, m=1 -> Mar 2019
        month_start = snapshot - pd.DateOffset(months=m)
        default_months.extend([month_start.strftime('%Y-%m-%d')] * n_defaults_per_month)
    # Rest have no default
    default_months.extend([None] * (n_total - n_defaults))

    # No payoffs
    payoff_months = [None] * n_total

    # Recoveries for charged off loans
    recoveries[:n_defaults] = [500.0] * n_defaults  # some recovery

    # last_pymnt_d: Current = March 2019, Charged Off = earlier date
    last_pymnt_d = ['2018-01-01'] * n_defaults + ['2019-03-01'] * (n_total - n_defaults)

    return pd.DataFrame({
        'funded_amnt': funded,
        'total_rec_prncp': rec_prncp,
        'loan_status': loan_status,
        'recoveries': recoveries,
        'issue_d': issue_d,
        'int_rate': int_rate,
        'term_months': term_months,
        'out_prncp': out_prncp,
        'default_month': default_months,
        'payoff_month': payoff_months,
        'last_pymnt_d': last_pymnt_d,
    })


@pytest.fixture
def synthetic_df_active():
    """
    Synthetic dataset of 5 active loans for CPR and pool characteristics.
    All have last_pymnt_d == 2019-03-01 and calc_amort columns.
    """
    return pd.DataFrame({
        'loan_status': ['Current', 'Current', 'Current', 'Current', 'Current'],
        'out_prncp': [5000.0, 8000.0, 3000.0, 10000.0, 4000.0],
        'int_rate': [0.10, 0.12, 0.08, 0.14, 0.09],
        'installment': [300.0, 500.0, 200.0, 700.0, 250.0],
        'updated_remaining_term': [20, 30, 10, 40, 15],
        'last_pmt_beginning_balance': [5200.0, 8300.0, 3100.0, 10400.0, 4150.0],
        'last_pmt_scheduled_principal': [200.0, 300.0, 180.0, 350.0, 210.0],
        'last_pmt_unscheduled_principal': [50.0, 80.0, 20.0, 100.0, 30.0],
    })


@pytest.fixture
def simple_pool_chars():
    """
    Simple pool for testing project_cashflows: $100K at 10% for 36 months.
    Monthly payment for $100K at 10%/year, 36-month term = $3,226.72
    """
    principal = 100_000.0
    annual_rate = 0.10
    term = 36
    monthly_rate = annual_rate / 12
    # Standard amortization payment formula: PMT = PV * r / (1 - (1+r)^-n)
    pmt = principal * monthly_rate / (1 - (1 + monthly_rate) ** -term)
    return {
        'total_upb': principal,
        'wac': annual_rate,
        'wam': term,
        'monthly_payment': round(pmt, 2),
    }


# ---------------------------------------------------------------------------
# compute_pool_assumptions
# ---------------------------------------------------------------------------

class TestComputePoolAssumptions:

    def test_returns_expected_keys(self, synthetic_df_all, synthetic_df_active):
        result = compute_pool_assumptions(synthetic_df_all, synthetic_df_active)
        expected_keys = {
            'cdr', 'cumulative_default_rate', 'avg_mdr', 'monthly_mdrs',
            'cpr', 'loss_severity', 'recovery_rate',
        }
        assert set(result.keys()) == expected_keys

    def test_monthly_mdrs_has_12_elements(self, synthetic_df_all, synthetic_df_active):
        result = compute_pool_assumptions(synthetic_df_all, synthetic_df_active)
        assert len(result['monthly_mdrs']) == 12

    def test_avg_mdr_known_value(self, synthetic_df_all, synthetic_df_active):
        """5 defaults/month × $10K each = $50K default_upb per month.
        Performing balance ≈ ~990 loans at amortized balance.
        avg_MDR should be roughly 0.006 (0.6%)."""
        result = compute_pool_assumptions(synthetic_df_all, synthetic_df_active)
        # With 1000 loans at $10K, ~990 performing, each ~14-26 months seasoned,
        # balance roughly $7-8K each → performing balance ~$7-8M
        # MDR = $50K / ~$7-8M ≈ 0.006-0.007
        assert 0.003 < result['avg_mdr'] < 0.015

    def test_cdr_from_avg_mdr(self, synthetic_df_all, synthetic_df_active):
        """CDR = 1 - (1 - avg_mdr)^12."""
        result = compute_pool_assumptions(synthetic_df_all, synthetic_df_active)
        expected_cdr = 1 - (1 - result['avg_mdr']) ** 12
        assert abs(result['cdr'] - expected_cdr) < 1e-10

    def test_cdr_round_trip(self, synthetic_df_all, synthetic_df_active):
        """CDR → MDR via projection formula should ≈ avg_mdr from observations."""
        result = compute_pool_assumptions(synthetic_df_all, synthetic_df_active)
        mdr_from_cdr = 1 - (1 - result['cdr']) ** (1 / 12)
        assert abs(mdr_from_cdr - result['avg_mdr']) < 1e-10

    def test_cumulative_default_rate_returned(self, synthetic_df_all, synthetic_df_active):
        """cumulative_default_rate should be returned and differ from conditional CDR."""
        result = compute_pool_assumptions(synthetic_df_all, synthetic_df_active)
        assert 'cumulative_default_rate' in result
        # 60 defaults × ($10K - $0) = $600K / (1000 × $10K) = 0.06
        expected_cum = 60 * 10000 / (1000 * 10000)
        assert abs(result['cumulative_default_rate'] - expected_cum) < 1e-6
        # Conditional CDR and cumulative rate should be different
        assert abs(result['cdr'] - result['cumulative_default_rate']) > 0.001

    def test_loss_severity_with_recoveries(self, synthetic_df_all, synthetic_df_active):
        """60 Charged Off loans, each: exposure = $10K - $0 = $10K, recovery = $500.
        Total exposure = $600K. Total recovery = $30K. UPB lost = $570K.
        Loss severity = $570K / $600K = 0.95."""
        result = compute_pool_assumptions(synthetic_df_all, synthetic_df_active)
        expected_loss_severity = (600000 - 30000) / 600000
        assert abs(result['loss_severity'] - expected_loss_severity) < 1e-4

    def test_loss_plus_recovery_equals_one(self, synthetic_df_all, synthetic_df_active):
        result = compute_pool_assumptions(synthetic_df_all, synthetic_df_active)
        assert abs(result['loss_severity'] + result['recovery_rate'] - 1.0) < 1e-6

    def test_cpr_non_negative(self, synthetic_df_all, synthetic_df_active):
        result = compute_pool_assumptions(synthetic_df_all, synthetic_df_active)
        assert result['cpr'] >= 0

    def test_cpr_includes_fully_paid_march_2019(self, synthetic_df_active):
        """Fully Paid loans with last_pymnt_d == March 2019 should contribute to CPR.

        Run 1: df_all has no FP March loans → CPR from Current only.
        Run 2: df_all has FP March loans with large unscheduled principal → higher CPR.
        """
        # df_all with no FP March 2019 loans
        df_all_no_fp = pd.DataFrame({
            'funded_amnt': [10000.0, 10000.0],
            'total_rec_prncp': [0.0, 5000.0],
            'loan_status': ['Charged Off', 'Current'],
            'recoveries': [500.0, 0.0],
            'issue_d': ['2017-01-01', '2017-01-01'],
            'int_rate': [0.10, 0.10],
            'term_months': [36, 36],
            'out_prncp': [0.0, 5000.0],
            'default_month': ['2018-10-01', None],
            'payoff_month': [None, None],
            'last_pymnt_d': ['2018-10-01', '2019-03-01'],
        })
        result_no_fp = compute_pool_assumptions(df_all_no_fp, synthetic_df_active)

        # df_all with FP March 2019 loans that had big prepayments
        df_all_with_fp = pd.DataFrame({
            'funded_amnt': [10000.0, 10000.0, 10000.0],
            'total_rec_prncp': [0.0, 5000.0, 10000.0],
            'loan_status': ['Charged Off', 'Current', 'Fully Paid'],
            'recoveries': [500.0, 0.0, 0.0],
            'issue_d': ['2017-01-01', '2017-01-01', '2017-01-01'],
            'int_rate': [0.10, 0.10, 0.10],
            'term_months': [36, 36, 36],
            'out_prncp': [0.0, 5000.0, 0.0],
            'default_month': ['2018-10-01', None, None],
            'payoff_month': [None, None, '2019-03-01'],
            'last_pymnt_d': ['2018-10-01', '2019-03-01', '2019-03-01'],
            # calc_amort columns for the FP loan (paid off entire remaining balance)
            'last_pmt_beginning_balance': [0.0, 0.0, 3000.0],
            'last_pmt_scheduled_principal': [0.0, 0.0, 200.0],
            'last_pmt_unscheduled_principal': [0.0, 0.0, 2800.0],
        })
        result_with_fp = compute_pool_assumptions(df_all_with_fp, synthetic_df_active)

        # CPR should be higher when FP March 2019 loans are included
        assert result_with_fp['cpr'] > result_no_fp['cpr'], (
            f"CPR with FP March ({result_with_fp['cpr']:.6f}) should exceed "
            f"CPR without ({result_no_fp['cpr']:.6f})"
        )

    def test_cpr_excludes_fully_paid_non_march(self, synthetic_df_active):
        """Fully Paid loans with last_pymnt_d != March 2019 should NOT affect CPR."""
        # df_all with FP loan that paid off in December 2018
        df_all = pd.DataFrame({
            'funded_amnt': [10000.0, 10000.0],
            'total_rec_prncp': [5000.0, 10000.0],
            'loan_status': ['Current', 'Fully Paid'],
            'recoveries': [0.0, 0.0],
            'issue_d': ['2017-01-01', '2017-01-01'],
            'int_rate': [0.10, 0.10],
            'term_months': [36, 36],
            'out_prncp': [5000.0, 0.0],
            'default_month': [None, None],
            'payoff_month': [None, '2018-12-01'],
            'last_pymnt_d': ['2019-03-01', '2018-12-01'],
            'last_pmt_beginning_balance': [0.0, 3000.0],
            'last_pmt_scheduled_principal': [0.0, 200.0],
            'last_pmt_unscheduled_principal': [0.0, 2800.0],
        })
        result = compute_pool_assumptions(df_all, synthetic_df_active)
        # CPR should come only from synthetic_df_active (Current loans)
        result_baseline = compute_pool_assumptions(
            pd.DataFrame({
                'funded_amnt': [10000.0],
                'total_rec_prncp': [5000.0],
                'loan_status': ['Current'],
                'recoveries': [0.0],
                'issue_d': ['2017-01-01'],
                'int_rate': [0.10],
                'term_months': [36],
                'out_prncp': [5000.0],
                'default_month': [None],
                'payoff_month': [None],
                'last_pymnt_d': ['2019-03-01'],
            }),
            synthetic_df_active,
        )
        assert abs(result['cpr'] - result_baseline['cpr']) < 1e-10

    def test_cdr_zero_when_no_defaults(self, synthetic_df_active):
        """Zero defaults in all 12 months → CDR = 0."""
        df_all = pd.DataFrame({
            'funded_amnt': [10000.0, 10000.0],
            'total_rec_prncp': [10000.0, 5000.0],
            'loan_status': ['Fully Paid', 'Current'],
            'recoveries': [0.0, 0.0],
            'issue_d': ['2017-01-01', '2017-01-01'],
            'int_rate': [0.10, 0.10],
            'term_months': [36, 36],
            'out_prncp': [0.0, 5000.0],
            'default_month': [None, None],
            'payoff_month': ['2018-06-01', None],
            'last_pymnt_d': ['2018-06-01', '2019-03-01'],
        })
        result = compute_pool_assumptions(df_all, synthetic_df_active)
        assert result['cdr'] == 0.0
        assert result['avg_mdr'] == 0.0
        assert all(m == 0.0 for m in result['monthly_mdrs'])
        assert result['loss_severity'] == 0.0

    def test_monthly_mdrs_all_non_negative(self, synthetic_df_all, synthetic_df_active):
        result = compute_pool_assumptions(synthetic_df_all, synthetic_df_active)
        for mdr in result['monthly_mdrs']:
            assert mdr >= 0.0

    def test_cdr_prepayment_adjusted_denominator(self, synthetic_df_active):
        """CDR denominator should be lower when loans have prepaid.

        Two runs with identical defaults but different out_prncp for Current loans:
        - Run 1: out_prncp = scheduled balance → no prepayment adjustment
        - Run 2: out_prncp < scheduled balance → adjustment reduces denominator
        Higher CDR expected in Run 2 (same numerator, smaller denominator).
        """
        n_total = 100
        n_defaults_per_month = 1
        n_defaults = n_defaults_per_month * 12

        funded = [10000.0] * n_total
        rec_prncp = [0.0] * n_total
        int_rate = [0.10] * n_total
        term_months = [36] * n_total
        issue_d = ['2017-01-01'] * n_total
        recoveries = [0.0] * n_total
        loan_status = ['Charged Off'] * n_defaults + ['Current'] * (n_total - n_defaults)

        snapshot = pd.Timestamp('2019-03-01')
        default_months = []
        for m in range(12, 0, -1):
            month_start = snapshot - pd.DateOffset(months=m)
            default_months.extend([month_start.strftime('%Y-%m-%d')] * n_defaults_per_month)
        default_months.extend([None] * (n_total - n_defaults))
        payoff_months = [None] * n_total

        # Compute the scheduled balance at snapshot for Current loans
        # (26 months seasoning at 10%, 36-month term, $10K)
        from src.amortization import calc_monthly_payment, calc_balance
        sched_pmt = calc_monthly_payment(
            np.array([10000.0]), np.array([0.10]), np.array([36.0])
        )[0]
        sched_bal, _, _ = calc_balance(
            np.array([10000.0]), np.array([0.10]),
            np.array([sched_pmt]), np.array([26.0])
        )
        sched_bal_at_snapshot = max(sched_bal[0], 0.0)

        # Run 1: no prepayment (out_prncp = scheduled balance)
        out_prncp_no_prepay = [0.0] * n_defaults + [sched_bal_at_snapshot] * (n_total - n_defaults)
        last_pymnt_d = ['2018-01-01'] * n_defaults + ['2019-03-01'] * (n_total - n_defaults)
        df_all_1 = pd.DataFrame({
            'funded_amnt': funded, 'total_rec_prncp': rec_prncp,
            'loan_status': loan_status, 'recoveries': recoveries,
            'issue_d': issue_d, 'int_rate': int_rate,
            'term_months': term_months, 'out_prncp': out_prncp_no_prepay,
            'default_month': default_months, 'payoff_month': payoff_months,
            'last_pymnt_d': last_pymnt_d,
        })
        result_1 = compute_pool_assumptions(df_all_1, synthetic_df_active)

        # Run 2: significant prepayment (out_prncp = 50% of scheduled balance)
        out_prncp_prepaid = [0.0] * n_defaults + [sched_bal_at_snapshot * 0.5] * (n_total - n_defaults)
        df_all_2 = pd.DataFrame({
            'funded_amnt': funded, 'total_rec_prncp': rec_prncp,
            'loan_status': loan_status, 'recoveries': recoveries,
            'issue_d': issue_d, 'int_rate': int_rate,
            'term_months': term_months, 'out_prncp': out_prncp_prepaid,
            'default_month': default_months, 'payoff_month': payoff_months,
            'last_pymnt_d': last_pymnt_d,
        })
        result_2 = compute_pool_assumptions(df_all_2, synthetic_df_active)

        # Same numerator, smaller denominator → higher CDR
        assert result_2['cdr'] > result_1['cdr'], (
            f"Prepayment-adjusted CDR ({result_2['cdr']:.6f}) should be > "
            f"unadjusted ({result_1['cdr']:.6f})"
        )
        assert result_2['avg_mdr'] > result_1['avg_mdr']

        # Both should still be valid CDRs
        assert 0 < result_1['cdr'] < 1.0
        assert 0 < result_2['cdr'] < 1.0

    def test_cdr_no_adjustment_for_charged_off(self, synthetic_df_active):
        """Charged Off / Fully Paid loans should use unadjusted schedule.
        Setting out_prncp to 0 for Charged Off loans should NOT affect CDR
        (they are not active, so monthly_prepaid stays 0)."""
        df_all = pd.DataFrame({
            'funded_amnt': [10000.0, 10000.0],
            'total_rec_prncp': [0.0, 5000.0],
            'loan_status': ['Charged Off', 'Fully Paid'],
            'recoveries': [500.0, 0.0],
            'issue_d': ['2017-01-01', '2017-01-01'],
            'int_rate': [0.10, 0.10],
            'term_months': [36, 36],
            'out_prncp': [0.0, 0.0],
            'default_month': ['2018-10-01', None],
            'payoff_month': [None, '2018-12-01'],
            'last_pymnt_d': ['2018-10-01', '2018-12-01'],
        })
        result = compute_pool_assumptions(df_all, synthetic_df_active)
        # CDR should be non-zero (there's a default)
        assert result['cdr'] > 0
        # The adjustment should not have kicked in (no active loans in df_all)
        assert len(result['monthly_mdrs']) == 12


# ---------------------------------------------------------------------------
# compute_pool_characteristics
# ---------------------------------------------------------------------------

class TestComputePoolCharacteristics:

    def test_returns_expected_keys(self, synthetic_df_active):
        result = compute_pool_characteristics(synthetic_df_active)
        assert set(result.keys()) == {'total_upb', 'wac', 'wam', 'monthly_payment'}

    def test_total_upb(self, synthetic_df_active):
        result = compute_pool_characteristics(synthetic_df_active)
        assert result['total_upb'] == 30000.0  # 5000+8000+3000+10000+4000

    def test_monthly_payment(self, synthetic_df_active):
        result = compute_pool_characteristics(synthetic_df_active)
        assert result['monthly_payment'] == 1950.0  # 300+500+200+700+250

    def test_wac_is_weighted(self, synthetic_df_active):
        """WAC should be weighted by out_prncp."""
        result = compute_pool_characteristics(synthetic_df_active)
        expected_wac = np.average(
            synthetic_df_active['int_rate'].values,
            weights=synthetic_df_active['out_prncp'].values
        )
        assert abs(result['wac'] - expected_wac) < 1e-6

    def test_wam_is_integer(self, synthetic_df_active):
        result = compute_pool_characteristics(synthetic_df_active)
        assert isinstance(result['wam'], int)

    def test_wam_is_weighted(self, synthetic_df_active):
        result = compute_pool_characteristics(synthetic_df_active)
        expected_wam = round(np.average(
            synthetic_df_active['updated_remaining_term'].values,
            weights=synthetic_df_active['out_prncp'].values
        ))
        assert result['wam'] == expected_wam


# ---------------------------------------------------------------------------
# project_cashflows
# ---------------------------------------------------------------------------

class TestProjectCashflows:

    def test_returns_dataframe(self, simple_pool_chars):
        result = project_cashflows(simple_pool_chars, 0.10, 0.12, 0.85, 0.95)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, simple_pool_chars):
        result = project_cashflows(simple_pool_chars, 0.10, 0.12, 0.85, 0.95)
        expected_cols = [
            'month', 'date', 'beginning_balance', 'defaults', 'loss',
            'recovery', 'interest', 'scheduled_principal', 'prepayments',
            'total_principal', 'ending_balance', 'total_cashflow'
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_zero_cdr_zero_cpr_is_standard_amortization(self, simple_pool_chars):
        """With 0% CDR and 0% CPR, should produce standard amortization.
        Balance after 36 months should be ~0."""
        result = project_cashflows(simple_pool_chars, 0.0, 0.0, 0.85, 0.95)
        assert len(result) == 36
        # Final balance should be near zero
        assert result.iloc[-1]['ending_balance'] < 1.0
        # No defaults or prepayments
        assert result['defaults'].sum() == 0.0
        assert result['prepayments'].sum() == 0.0

    def test_zero_cdr_zero_cpr_total_principal(self, simple_pool_chars):
        """With no defaults or prepayments, total principal paid ≈ original UPB."""
        result = project_cashflows(simple_pool_chars, 0.0, 0.0, 0.85, 0.95)
        total_principal = result['total_principal'].sum()
        assert abs(total_principal - 100_000) < 10.0  # Within $10 due to rounding

    def test_beginning_balance_decreases(self, simple_pool_chars):
        """Beginning balance should decrease over time."""
        result = project_cashflows(simple_pool_chars, 0.05, 0.10, 0.85, 0.95)
        balances = result['beginning_balance'].values
        for i in range(1, len(balances)):
            assert balances[i] <= balances[i - 1]

    def test_ending_balance_non_negative(self, simple_pool_chars):
        """Ending balance should never be negative."""
        result = project_cashflows(simple_pool_chars, 0.10, 0.20, 0.85, 0.95)
        assert (result['ending_balance'] >= 0).all()

    def test_high_cdr_faster_runoff(self, simple_pool_chars):
        """Higher CDR should cause balance to decrease faster."""
        low_cdr = project_cashflows(simple_pool_chars, 0.05, 0.0, 0.85, 0.95)
        high_cdr = project_cashflows(simple_pool_chars, 0.20, 0.0, 0.85, 0.95)
        # At month 12, high CDR should have lower balance
        low_bal = low_cdr[low_cdr['month'] == 12]['ending_balance'].values[0]
        high_bal = high_cdr[high_cdr['month'] == 12]['ending_balance'].values[0]
        assert high_bal < low_bal

    def test_high_cpr_faster_runoff(self, simple_pool_chars):
        """Higher CPR should cause balance to decrease faster."""
        low_cpr = project_cashflows(simple_pool_chars, 0.0, 0.05, 0.85, 0.95)
        high_cpr = project_cashflows(simple_pool_chars, 0.0, 0.30, 0.85, 0.95)
        # High CPR should have fewer rows (balance hits zero earlier)
        assert len(high_cpr) <= len(low_cpr)

    def test_dates_start_april_2019(self, simple_pool_chars):
        """First cash flow date should be April 2019 (t=1)."""
        result = project_cashflows(simple_pool_chars, 0.10, 0.12, 0.85, 0.95)
        assert result.iloc[0]['date'] == pd.Timestamp('2019-04-01')

    def test_total_cashflow_equals_components(self, simple_pool_chars):
        """total_cashflow = interest + total_principal + recovery."""
        result = project_cashflows(simple_pool_chars, 0.10, 0.12, 0.85, 0.95)
        for _, row in result.iterrows():
            expected = row['interest'] + row['total_principal'] + row['recovery']
            assert abs(row['total_cashflow'] - expected) < 0.02  # rounding tolerance

    def test_loss_equals_defaults_times_severity(self, simple_pool_chars):
        """loss = defaults × loss_severity."""
        loss_sev = 0.85
        result = project_cashflows(simple_pool_chars, 0.10, 0.12, loss_sev, 0.95)
        for _, row in result.iterrows():
            if row['defaults'] > 0:
                expected_loss = row['defaults'] * loss_sev
                assert abs(row['loss'] - expected_loss) < 0.02


# ---------------------------------------------------------------------------
# calculate_irr
# ---------------------------------------------------------------------------

class TestCalculateIrr:

    def test_simple_cashflows(self):
        """Simple test: purchase at par with known cash flows."""
        pool_chars = {'total_upb': 100.0, 'wac': 0.12, 'wam': 2, 'monthly_payment': 50.0}
        # Create a simple cashflow dataframe
        cf_df = pd.DataFrame({
            'total_cashflow': [60.0, 60.0],
        })
        irr = calculate_irr(cf_df, pool_chars, 1.0)
        # Cash flows: [-100, 60, 60]
        # Monthly IRR from npf.irr
        expected_monthly = npf.irr([-100, 60, 60])
        expected_annual = (1 + expected_monthly) ** 12 - 1
        assert abs(irr - expected_annual) < 0.001

    def test_npv_at_irr_is_zero(self, simple_pool_chars):
        """NPV at the computed IRR should be approximately zero."""
        cf_df = project_cashflows(simple_pool_chars, 0.05, 0.10, 0.85, 0.95)
        irr = calculate_irr(cf_df, simple_pool_chars, 0.95)
        # Verify NPV ≈ 0
        cf = np.zeros(len(cf_df) + 1)
        cf[0] = -(0.95 * simple_pool_chars['total_upb'])
        cf[1:] = cf_df['total_cashflow'].values
        monthly_irr = (1 + irr) ** (1 / 12) - 1
        npv = npf.npv(monthly_irr, cf)
        assert abs(npv) < 10.0  # Within $10 for a $100K pool

    def test_higher_price_lower_irr(self, simple_pool_chars):
        """Higher purchase price → lower IRR."""
        cf_low = project_cashflows(simple_pool_chars, 0.05, 0.10, 0.85, 0.90)
        cf_high = project_cashflows(simple_pool_chars, 0.05, 0.10, 0.85, 1.00)
        irr_low_price = calculate_irr(cf_low, simple_pool_chars, 0.90)
        irr_high_price = calculate_irr(cf_high, simple_pool_chars, 1.00)
        assert irr_low_price > irr_high_price

    def test_at_par_with_no_defaults(self, simple_pool_chars):
        """Buy at par with no defaults → IRR should be close to WAC."""
        cf_df = project_cashflows(simple_pool_chars, 0.0, 0.0, 0.0, 1.0)
        irr = calculate_irr(cf_df, simple_pool_chars, 1.0)
        # With no defaults and buying at par, IRR ≈ WAC
        assert abs(irr - simple_pool_chars['wac']) < 0.005


# ---------------------------------------------------------------------------
# solve_price
# ---------------------------------------------------------------------------

class TestSolvePrice:

    def test_round_trip(self, simple_pool_chars):
        """Solve for price at target IRR, then verify IRR at that price matches target."""
        target_irr = 0.12  # 12% target
        cdr, cpr, loss_sev = 0.05, 0.10, 0.85
        price = solve_price(simple_pool_chars, target_irr, cdr, cpr, loss_sev)
        assert price is not None
        # Verify round-trip
        cf_df = project_cashflows(simple_pool_chars, cdr, cpr, loss_sev, price)
        actual_irr = calculate_irr(cf_df, simple_pool_chars, price)
        assert abs(actual_irr - target_irr) < 0.001  # Within 10bp

    def test_higher_target_irr_lower_price(self, simple_pool_chars):
        """Higher target IRR should require lower purchase price."""
        cdr, cpr, loss_sev = 0.05, 0.10, 0.85
        price_low_irr = solve_price(simple_pool_chars, 0.08, cdr, cpr, loss_sev)
        price_high_irr = solve_price(simple_pool_chars, 0.15, cdr, cpr, loss_sev)
        assert price_low_irr is not None
        assert price_high_irr is not None
        assert price_high_irr < price_low_irr

    def test_returns_none_for_impossible_target(self, simple_pool_chars):
        """Impossibly high target IRR should return None."""
        price = solve_price(simple_pool_chars, 10.0, 0.05, 0.10, 0.85)
        assert price is None

    def test_price_in_reasonable_range(self, simple_pool_chars):
        """Solved price should be between 0.50 and 1.50."""
        price = solve_price(simple_pool_chars, 0.10, 0.05, 0.10, 0.85)
        assert price is not None
        assert 0.50 <= price <= 1.50


# ---------------------------------------------------------------------------
# State-Transition Model: Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def all_current_pool_state():
    """
    Pool state: 100% Current loans at various ages.
    Total UPB = $1M.
    """
    return {
        'states': {
            'Current': {10: 300_000.0, 20: 400_000.0, 30: 300_000.0},
            'Delinquent (0-30)': {},
            'Late_1': {},
            'Late_2': {},
            'Late_3': {},
            'Charged Off': {},
            'Fully Paid': {},
        },
        'total_upb': 1_000_000.0,
        'wac': 0.10,
        'wam': 30,
        'monthly_payment': 40_000.0,
    }


@pytest.fixture
def transition_pool_chars():
    """Pool characteristics matching all_current_pool_state."""
    return {
        'total_upb': 1_000_000.0,
        'wac': 0.10,
        'wam': 30,
        'monthly_payment': 40_000.0,
    }


@pytest.fixture
def simple_7state_probs():
    """
    Simple 7-state transition probabilities for ages 0-59.
    Current → Current: 93%, → Delinquent: 5%, → Fully Paid: 2%
    Delinquent → Late_1: 50%, → Current (cure): 50%
    Late_1 → Late_2: 80%, → Current (cure): 20%
    Late_2 → Late_3: 90%, → Current (cure): 10%
    Late_3 → Charged Off: 85%, → Current (cure): 15%
    """
    rows = []
    for age in range(60):
        # Current transitions
        rows.append({
            'age_bucket': str(age),
            'from_status': 'Current',
            'to_current_pct': 0.93,
            'to_delinquent_0_30_pct': 0.05,
            'to_late_1_pct': 0.0,
            'to_late_2_pct': 0.0,
            'to_late_3_pct': 0.0,
            'to_charged_off_pct': 0.0,
            'to_fully_paid_pct': 0.02,
            'observation_count': 1000,
        })
        # Delinquent transitions
        rows.append({
            'age_bucket': str(age),
            'from_status': 'Delinquent (0-30)',
            'to_current_pct': 0.50,
            'to_delinquent_0_30_pct': 0.0,
            'to_late_1_pct': 0.50,
            'to_late_2_pct': 0.0,
            'to_late_3_pct': 0.0,
            'to_charged_off_pct': 0.0,
            'to_fully_paid_pct': 0.0,
            'observation_count': 500,
        })
        # Late_1 transitions
        rows.append({
            'age_bucket': str(age),
            'from_status': 'Late_1',
            'to_current_pct': 0.20,
            'to_delinquent_0_30_pct': 0.0,
            'to_late_1_pct': 0.0,
            'to_late_2_pct': 0.80,
            'to_late_3_pct': 0.0,
            'to_charged_off_pct': 0.0,
            'to_fully_paid_pct': 0.0,
            'observation_count': 300,
        })
        # Late_2 transitions
        rows.append({
            'age_bucket': str(age),
            'from_status': 'Late_2',
            'to_current_pct': 0.10,
            'to_delinquent_0_30_pct': 0.0,
            'to_late_1_pct': 0.0,
            'to_late_2_pct': 0.0,
            'to_late_3_pct': 0.90,
            'to_charged_off_pct': 0.0,
            'to_fully_paid_pct': 0.0,
            'observation_count': 200,
        })
        # Late_3 transitions
        rows.append({
            'age_bucket': str(age),
            'from_status': 'Late_3',
            'to_current_pct': 0.15,
            'to_delinquent_0_30_pct': 0.0,
            'to_late_1_pct': 0.0,
            'to_late_2_pct': 0.0,
            'to_late_3_pct': 0.0,
            'to_charged_off_pct': 0.85,
            'to_fully_paid_pct': 0.0,
            'observation_count': 150,
        })
        # Absorbing states
        rows.append({
            'age_bucket': str(age),
            'from_status': 'Charged Off',
            'to_current_pct': 0.0,
            'to_delinquent_0_30_pct': 0.0,
            'to_late_1_pct': 0.0,
            'to_late_2_pct': 0.0,
            'to_late_3_pct': 0.0,
            'to_charged_off_pct': 1.0,
            'to_fully_paid_pct': 0.0,
            'observation_count': 100,
        })
        rows.append({
            'age_bucket': str(age),
            'from_status': 'Fully Paid',
            'to_current_pct': 0.0,
            'to_delinquent_0_30_pct': 0.0,
            'to_late_1_pct': 0.0,
            'to_late_2_pct': 0.0,
            'to_late_3_pct': 0.0,
            'to_charged_off_pct': 0.0,
            'to_fully_paid_pct': 1.0,
            'observation_count': 100,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def zero_default_probs():
    """Probs where Current→Delinquent = 0. No defaults possible."""
    rows = []
    for age in range(60):
        rows.append({
            'age_bucket': str(age),
            'from_status': 'Current',
            'to_current_pct': 0.98,
            'to_delinquent_0_30_pct': 0.0,
            'to_late_1_pct': 0.0,
            'to_late_2_pct': 0.0,
            'to_late_3_pct': 0.0,
            'to_charged_off_pct': 0.0,
            'to_fully_paid_pct': 0.02,
            'observation_count': 1000,
        })
        for state in ['Charged Off', 'Fully Paid']:
            rows.append({
                'age_bucket': str(age),
                'from_status': state,
                'to_current_pct': 0.0,
                'to_delinquent_0_30_pct': 0.0,
                'to_late_1_pct': 0.0,
                'to_late_2_pct': 0.0,
                'to_late_3_pct': 0.0,
                'to_charged_off_pct': 1.0 if state == 'Charged Off' else 0.0,
                'to_fully_paid_pct': 1.0 if state == 'Fully Paid' else 0.0,
                'observation_count': 100,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# build_pool_state
# ---------------------------------------------------------------------------

class TestBuildPoolState:

    def test_all_current_pool(self):
        """All-Current pool: only Current state has balances."""
        df = pd.DataFrame({
            'loan_status': ['Current', 'Current', 'Current'],
            'out_prncp': [100_000.0, 200_000.0, 150_000.0],
            'int_rate': [0.10, 0.12, 0.08],
            'updated_remaining_term': [20, 30, 25],
            'installment': [5000.0, 8000.0, 6000.0],
            'loan_age_months': [10, 20, 15],
        })
        result = build_pool_state(df)

        assert sum(result['states']['Current'].values()) == 450_000.0
        assert sum(result['states']['Delinquent (0-30)'].values()) == 0
        assert sum(result['states']['Late_1'].values()) == 0
        assert result['total_upb'] == 450_000.0

    def test_mixed_statuses(self):
        """Pool with Current + Late → correct state assignment."""
        df = pd.DataFrame({
            'loan_status': ['Current', 'In Grace Period', 'Late (16-30 days)'],
            'out_prncp': [100_000.0, 50_000.0, 30_000.0],
            'int_rate': [0.10, 0.12, 0.09],
            'updated_remaining_term': [20, 25, 18],
            'installment': [5000.0, 3000.0, 2000.0],
            'loan_age_months': [10, 15, 12],
        })
        result = build_pool_state(df, include_statuses=[
            'Current', 'In Grace Period', 'Late (16-30 days)'])

        assert sum(result['states']['Current'].values()) == 100_000.0
        # Grace + Late 16-30 → Delinquent (0-30)
        assert sum(result['states']['Delinquent (0-30)'].values()) == 80_000.0

    def test_upb_sums_match(self):
        """Sum of all state UPBs should equal total_upb."""
        df = pd.DataFrame({
            'loan_status': ['Current', 'Current'],
            'out_prncp': [100_000.0, 200_000.0],
            'int_rate': [0.10, 0.12],
            'updated_remaining_term': [20, 30],
            'installment': [5000.0, 8000.0],
            'loan_age_months': [10, 20],
        })
        result = build_pool_state(df)

        state_total = sum(
            sum(ages.values()) for ages in result['states'].values()
        )
        assert abs(state_total - result['total_upb']) < 1.0

    def test_empty_pool(self):
        """Empty pool → zero UPB and empty states."""
        df = pd.DataFrame({
            'loan_status': ['Fully Paid'],
            'out_prncp': [0.0],
            'int_rate': [0.10],
            'updated_remaining_term': [0],
            'installment': [0.0],
            'loan_age_months': [36],
        })
        result = build_pool_state(df)  # Default include_statuses includes all active
        assert result['total_upb'] == 0.0


# ---------------------------------------------------------------------------
# project_cashflows_transition
# ---------------------------------------------------------------------------

class TestProjectCashflowsTransition:

    def test_pipeline_timing(self, all_current_pool_state, simple_7state_probs,
                             transition_pool_chars):
        """All-Current pool: months 1-4 should have zero defaults."""
        cf = project_cashflows_transition(
            all_current_pool_state, simple_7state_probs,
            0.85, 0.15, transition_pool_chars, 36,
        )

        # Pipeline: Current → Delinq (mo1) → Late_1 (mo2) → Late_2 (mo3)
        # → Late_3 (mo4) → Charged Off (mo5)
        for month in [1, 2, 3, 4]:
            row = cf[cf['month'] == month]
            if len(row) > 0:
                assert row.iloc[0]['defaults'] == 0.0, (
                    f"Month {month} should have zero defaults"
                )

        # Month 5+ should have some defaults
        month5 = cf[cf['month'] == 5]
        if len(month5) > 0:
            assert month5.iloc[0]['defaults'] > 0, (
                "Month 5 should have some defaults"
            )

    def test_state_rampup(self, all_current_pool_state, simple_7state_probs,
                          transition_pool_chars):
        """Delinquent UPB should appear at month 1, Late_1 at month 2, etc."""
        cf = project_cashflows_transition(
            all_current_pool_state, simple_7state_probs,
            0.85, 0.15, transition_pool_chars, 12,
        )

        # Month 1: delinquent > 0
        assert cf.iloc[0]['delinquent_upb'] > 0, "Delinquent UPB should appear at month 1"

        # Month 2: late_1 > 0
        assert cf.iloc[1]['late_1_upb'] > 0, "Late_1 UPB should appear at month 2"

        # Month 3: late_2 > 0
        assert cf.iloc[2]['late_2_upb'] > 0, "Late_2 UPB should appear at month 3"

        # Month 4: late_3 > 0
        assert cf.iloc[3]['late_3_upb'] > 0, "Late_3 UPB should appear at month 4"

    def test_balance_decreases(self, all_current_pool_state, simple_7state_probs,
                               transition_pool_chars):
        """Ending balance (non-absorbing states) should generally decrease."""
        cf = project_cashflows_transition(
            all_current_pool_state, simple_7state_probs,
            0.85, 0.15, transition_pool_chars, 30,
        )

        # Check that ending balance at month 10 < month 1
        if len(cf) >= 10:
            assert cf.iloc[9]['ending_balance'] < cf.iloc[0]['ending_balance']

    def test_cashflow_components(self, all_current_pool_state,
                                 simple_7state_probs, transition_pool_chars):
        """total_cashflow = interest + sched_principal + prepayments + recoveries."""
        cf = project_cashflows_transition(
            all_current_pool_state, simple_7state_probs,
            0.85, 0.15, transition_pool_chars, 12,
        )

        for _, row in cf.iterrows():
            expected = (row['interest'] + row['scheduled_principal']
                        + row['prepayments'] + row['recovery'])
            assert abs(row['total_cashflow'] - expected) < 0.10

    def test_zero_defaults_when_no_delinquency(self, all_current_pool_state,
                                                zero_default_probs,
                                                transition_pool_chars):
        """With 0% Current→Delinquent, there should be zero defaults."""
        cf = project_cashflows_transition(
            all_current_pool_state, zero_default_probs,
            0.85, 0.15, transition_pool_chars, 30,
        )

        assert cf['defaults'].sum() == 0.0
        assert cf['loss'].sum() == 0.0

    def test_irr_compatibility(self, all_current_pool_state,
                               simple_7state_probs, transition_pool_chars):
        """Output should work with calculate_irr()."""
        cf = project_cashflows_transition(
            all_current_pool_state, simple_7state_probs,
            0.85, 0.15, transition_pool_chars, 30,
        )

        irr = calculate_irr(cf, transition_pool_chars, 0.95)
        assert not np.isnan(irr)
        assert -1.0 < irr < 5.0  # Reasonable range

    def test_returns_expected_columns(self, all_current_pool_state,
                                       simple_7state_probs, transition_pool_chars):
        """Output should have all expected columns."""
        cf = project_cashflows_transition(
            all_current_pool_state, simple_7state_probs,
            0.85, 0.15, transition_pool_chars, 12,
        )

        expected_cols = [
            'month', 'date', 'beginning_balance', 'interest',
            'scheduled_principal', 'prepayments', 'defaults', 'loss',
            'recovery', 'total_principal', 'ending_balance', 'total_cashflow',
            'current_upb', 'delinquent_upb', 'late_1_upb', 'late_2_upb',
            'late_3_upb', 'default_upb', 'fully_paid_upb',
        ]
        for col in expected_cols:
            assert col in cf.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# solve_price_transition
# ---------------------------------------------------------------------------

class TestSolvePriceTransition:

    def test_round_trip(self, all_current_pool_state, simple_7state_probs,
                        transition_pool_chars):
        """Solved price produces target IRR (round-trip check)."""
        target_irr = 0.12
        price = solve_price_transition(
            all_current_pool_state, simple_7state_probs,
            0.85, 0.15, transition_pool_chars, 30, target_irr,
        )

        assert price is not None
        assert 0.50 <= price <= 1.50

        # Verify round-trip
        cf = project_cashflows_transition(
            all_current_pool_state, simple_7state_probs,
            0.85, 0.15, transition_pool_chars, 30,
        )
        actual_irr = calculate_irr(cf, transition_pool_chars, price)
        assert abs(actual_irr - target_irr) < 0.01  # Within 100bp


# ---------------------------------------------------------------------------
# adjust_prepayment_rates
# ---------------------------------------------------------------------------

class TestAdjustPrepaymentRates:

    def test_current_rows_fully_paid_equals_smm(self, simple_7state_probs):
        """Current → Fully Paid should equal CPR-derived SMM after adjustment."""
        cpr = 0.0149
        smm = 1 - (1 - cpr) ** (1 / 12)

        adjusted = adjust_prepayment_rates(simple_7state_probs, cpr)
        current_rows = adjusted[adjusted['from_status'] == 'Current']

        for _, row in current_rows.iterrows():
            assert abs(row['to_fully_paid_pct'] - smm) < 1e-10

    def test_non_current_rows_unchanged(self, simple_7state_probs):
        """Non-Current rows should be identical before and after adjustment."""
        cpr = 0.0149
        adjusted = adjust_prepayment_rates(simple_7state_probs, cpr)

        non_current_mask = simple_7state_probs['from_status'] != 'Current'
        pct_cols = [c for c in adjusted.columns if c.startswith('to_') and c.endswith('_pct')]

        pd.testing.assert_frame_equal(
            simple_7state_probs.loc[non_current_mask, pct_cols].reset_index(drop=True),
            adjusted.loc[non_current_mask, pct_cols].reset_index(drop=True),
        )

    def test_current_rows_sum_to_one(self, simple_7state_probs):
        """After adjustment, Current rows should still sum to 1.0."""
        cpr = 0.0149
        adjusted = adjust_prepayment_rates(simple_7state_probs, cpr)
        current_rows = adjusted[adjusted['from_status'] == 'Current']

        pct_cols = [c for c in adjusted.columns if c.startswith('to_') and c.endswith('_pct')]
        row_sums = current_rows[pct_cols].sum(axis=1)

        for s in row_sums:
            assert abs(s - 1.0) < 1e-6

    def test_original_not_mutated(self, simple_7state_probs):
        """adjust_prepayment_rates should return a copy, not mutate the input."""
        original_fp = simple_7state_probs.loc[
            simple_7state_probs['from_status'] == 'Current', 'to_fully_paid_pct'
        ].values.copy()

        adjust_prepayment_rates(simple_7state_probs, 0.0149)

        after_fp = simple_7state_probs.loc[
            simple_7state_probs['from_status'] == 'Current', 'to_fully_paid_pct'
        ].values
        np.testing.assert_array_equal(original_fp, after_fp)

    def test_zero_cpr(self, simple_7state_probs):
        """CPR = 0 → SMM = 0 → no prepayments."""
        adjusted = adjust_prepayment_rates(simple_7state_probs, 0.0)
        current_rows = adjusted[adjusted['from_status'] == 'Current']

        assert (current_rows['to_fully_paid_pct'] == 0.0).all()

    def test_delinquency_rates_preserved(self, simple_7state_probs):
        """Current → Delinquent should be unchanged by CPR adjustment."""
        cpr = 0.0149
        adjusted = adjust_prepayment_rates(simple_7state_probs, cpr)

        current_orig = simple_7state_probs[simple_7state_probs['from_status'] == 'Current']
        current_adj = adjusted[adjusted['from_status'] == 'Current']

        np.testing.assert_array_almost_equal(
            current_orig['to_delinquent_0_30_pct'].values,
            current_adj['to_delinquent_0_30_pct'].values,
        )

    def test_extreme_cpr_row_sums_to_one(self, simple_7state_probs):
        """At CPR=1.0 (100% prepayment), rows should still sum to 1.0."""
        adjusted = adjust_prepayment_rates(simple_7state_probs, 1.0)
        current_rows = adjusted[adjusted['from_status'] == 'Current']
        pct_cols = [c for c in adjusted.columns
                    if c.startswith('to_') and c.endswith('_pct')]
        row_sums = current_rows[pct_cols].sum(axis=1)
        for s in row_sums:
            assert abs(s - 1.0) < 1e-6, f"Row sum {s} != 1.0 at CPR=1.0"

    def test_high_cpr_row_sums_to_one(self, simple_7state_probs):
        """At CPR=0.9999 (near-100%), rows should still sum to 1.0."""
        adjusted = adjust_prepayment_rates(simple_7state_probs, 0.9999)
        current_rows = adjusted[adjusted['from_status'] == 'Current']
        pct_cols = [c for c in adjusted.columns
                    if c.startswith('to_') and c.endswith('_pct')]
        row_sums = current_rows[pct_cols].sum(axis=1)
        for s in row_sums:
            assert abs(s - 1.0) < 1e-6, f"Row sum {s} != 1.0 at CPR=0.9999"
