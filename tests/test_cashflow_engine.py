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
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df_all():
    """
    Synthetic dataset of 10 loans (all statuses) for compute_pool_assumptions.
    Includes Current, Fully Paid, and Charged Off loans with known CDR and loss severity.
    """
    return pd.DataFrame({
        'funded_amnt': [10000] * 10,
        'total_rec_prncp': [5000, 10000, 10000, 5000, 10000, 10000, 5000, 10000, 10000, 5000],
        'loan_status': [
            'Charged Off', 'Fully Paid', 'Fully Paid', 'Charged Off',
            'Fully Paid', 'Current', 'Current', 'Fully Paid',
            'Current', 'Charged Off'
        ],
        'recoveries': [1000, 0, 0, 500, 0, 0, 0, 0, 0, 2000],
        'issue_d': ['2017-03-01'] * 10,  # 24 months before snapshot (2019-03-01)
    })


@pytest.fixture
def synthetic_df_current():
    """
    Synthetic dataset of 5 Current loans for CPR and pool characteristics.
    All have last_pymnt_d == 2019-03-01 and calc_amort columns.
    """
    return pd.DataFrame({
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

    def test_returns_expected_keys(self, synthetic_df_all, synthetic_df_current):
        result = compute_pool_assumptions(synthetic_df_all, synthetic_df_current)
        assert set(result.keys()) == {'cdr', 'cdr_cumulative', 'cpr', 'loss_severity', 'recovery_rate'}

    def test_cdr_cumulative_known_value(self, synthetic_df_all, synthetic_df_current):
        """3 Charged Off loans, each funded 10K with 5K rec_prncp.
        CDR_cumulative = 3*(10000-5000) / (10*10000) = 15000/100000 = 0.15"""
        result = compute_pool_assumptions(synthetic_df_all, synthetic_df_current)
        assert abs(result['cdr_cumulative'] - 0.15) < 1e-6

    def test_cdr_annualized_known_value(self, synthetic_df_all, synthetic_df_current):
        """CDR_cumulative = 0.15 over WALA = 24 months.
        CDR_annual = 1 - (1 - 0.15)^(12/24) = 1 - 0.85^0.5 ≈ 0.07804"""
        result = compute_pool_assumptions(synthetic_df_all, synthetic_df_current)
        expected_annual = 1 - (1 - 0.15) ** (12 / 24)
        assert abs(result['cdr'] - expected_annual) < 1e-4

    def test_loss_severity_with_capped_recoveries(self, synthetic_df_all, synthetic_df_current):
        """Charged Off loans have exposure = 5000 each (3 loans).
        Recoveries: 1000, 500, 2000. All < exposure, so no capping needed.
        Capped UPB lost: 4000 + 4500 + 3000 = 11500
        Total exposure: 15000
        Loss severity = 11500/15000 ≈ 0.7667"""
        result = compute_pool_assumptions(synthetic_df_all, synthetic_df_current)
        expected_loss_severity = 11500 / 15000
        assert abs(result['loss_severity'] - expected_loss_severity) < 1e-4

    def test_loss_plus_recovery_equals_one(self, synthetic_df_all, synthetic_df_current):
        result = compute_pool_assumptions(synthetic_df_all, synthetic_df_current)
        assert abs(result['loss_severity'] + result['recovery_rate'] - 1.0) < 1e-6

    def test_cpr_non_negative(self, synthetic_df_all, synthetic_df_current):
        result = compute_pool_assumptions(synthetic_df_all, synthetic_df_current)
        assert result['cpr'] >= 0

    def test_cdr_zero_when_no_charged_off(self, synthetic_df_current):
        """No Charged Off loans → CDR = 0."""
        df_all = pd.DataFrame({
            'funded_amnt': [10000, 10000],
            'total_rec_prncp': [10000, 5000],
            'loan_status': ['Fully Paid', 'Current'],
            'recoveries': [0, 0],
            'issue_d': ['2017-03-01', '2017-03-01'],
        })
        result = compute_pool_assumptions(df_all, synthetic_df_current)
        assert result['cdr'] == 0.0
        assert result['cdr_cumulative'] == 0.0
        assert result['loss_severity'] == 0.0


# ---------------------------------------------------------------------------
# compute_pool_characteristics
# ---------------------------------------------------------------------------

class TestComputePoolCharacteristics:

    def test_returns_expected_keys(self, synthetic_df_current):
        result = compute_pool_characteristics(synthetic_df_current)
        assert set(result.keys()) == {'total_upb', 'wac', 'wam', 'monthly_payment'}

    def test_total_upb(self, synthetic_df_current):
        result = compute_pool_characteristics(synthetic_df_current)
        assert result['total_upb'] == 30000.0  # 5000+8000+3000+10000+4000

    def test_monthly_payment(self, synthetic_df_current):
        result = compute_pool_characteristics(synthetic_df_current)
        assert result['monthly_payment'] == 1950.0  # 300+500+200+700+250

    def test_wac_is_weighted(self, synthetic_df_current):
        """WAC should be weighted by out_prncp."""
        result = compute_pool_characteristics(synthetic_df_current)
        expected_wac = np.average(
            synthetic_df_current['int_rate'].values,
            weights=synthetic_df_current['out_prncp'].values
        )
        assert abs(result['wac'] - expected_wac) < 1e-6

    def test_wam_is_integer(self, synthetic_df_current):
        result = compute_pool_characteristics(synthetic_df_current)
        assert isinstance(result['wam'], int)

    def test_wam_is_weighted(self, synthetic_df_current):
        result = compute_pool_characteristics(synthetic_df_current)
        expected_wam = round(np.average(
            synthetic_df_current['updated_remaining_term'].values,
            weights=synthetic_df_current['out_prncp'].values
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
