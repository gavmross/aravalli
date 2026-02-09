"""Tests for src/scenario_analysis.py"""

import numpy as np
import pandas as pd
import pytest

from src.scenario_analysis import (
    compute_base_assumptions,
    build_scenarios,
    compare_scenarios,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_assumptions():
    """Known base assumptions for testing build_scenarios."""
    return {
        'cdr': 0.08,
        'cpr': 0.12,
        'loss_severity': 0.85,
    }


@pytest.fixture
def simple_pool_chars():
    """Simple pool for testing compare_scenarios."""
    principal = 100_000.0
    annual_rate = 0.10
    term = 36
    monthly_rate = annual_rate / 12
    pmt = principal * monthly_rate / (1 - (1 + monthly_rate) ** -term)
    return {
        'total_upb': principal,
        'wac': annual_rate,
        'wam': term,
        'monthly_payment': round(pmt, 2),
    }


@pytest.fixture
def synthetic_df_all():
    """Synthetic dataset for compute_base_assumptions."""
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
    """Synthetic Current loans for compute_base_assumptions."""
    return pd.DataFrame({
        'out_prncp': [5000.0, 8000.0, 3000.0],
        'int_rate': [0.10, 0.12, 0.08],
        'installment': [300.0, 500.0, 200.0],
        'updated_remaining_term': [20, 30, 10],
        'last_pmt_beginning_balance': [5200.0, 8300.0, 3100.0],
        'last_pmt_scheduled_principal': [200.0, 300.0, 180.0],
        'last_pmt_unscheduled_principal': [50.0, 80.0, 20.0],
    })


# ---------------------------------------------------------------------------
# compute_base_assumptions
# ---------------------------------------------------------------------------

class TestComputeBaseAssumptions:

    def test_returns_expected_keys(self, synthetic_df_all, synthetic_df_current):
        result = compute_base_assumptions(synthetic_df_all, synthetic_df_current)
        assert set(result.keys()) == {'cdr', 'cpr', 'loss_severity'}

    def test_no_recovery_rate_key(self, synthetic_df_all, synthetic_df_current):
        """compute_base_assumptions returns only cdr, cpr, loss_severity (no recovery_rate)."""
        result = compute_base_assumptions(synthetic_df_all, synthetic_df_current)
        assert 'recovery_rate' not in result

    def test_values_match_pool_assumptions(self, synthetic_df_all, synthetic_df_current):
        """Should match compute_pool_assumptions for cdr, cpr, loss_severity."""
        from src.cashflow_engine import compute_pool_assumptions
        full = compute_pool_assumptions(synthetic_df_all, synthetic_df_current)
        base = compute_base_assumptions(synthetic_df_all, synthetic_df_current)
        assert base['cdr'] == full['cdr']
        assert base['cpr'] == full['cpr']
        assert base['loss_severity'] == full['loss_severity']


# ---------------------------------------------------------------------------
# build_scenarios
# ---------------------------------------------------------------------------

class TestBuildScenarios:

    def test_returns_three_scenarios(self, base_assumptions):
        result = build_scenarios(base_assumptions)
        assert set(result.keys()) == {'Base', 'Stress', 'Upside'}

    def test_base_unchanged(self, base_assumptions):
        """Base scenario should match input exactly."""
        result = build_scenarios(base_assumptions)
        assert result['Base']['cdr'] == 0.08
        assert result['Base']['cpr'] == 0.12
        assert result['Base']['loss_severity'] == 0.85

    def test_stress_cdr_increases(self, base_assumptions):
        """Stress CDR = base × (1 + stress_pct)."""
        result = build_scenarios(base_assumptions, stress_pct=0.15)
        expected_cdr = 0.08 * 1.15  # 0.092
        assert abs(result['Stress']['cdr'] - expected_cdr) < 1e-10

    def test_stress_cpr_decreases(self, base_assumptions):
        """Stress CPR = base × (1 - stress_pct)."""
        result = build_scenarios(base_assumptions, stress_pct=0.15)
        expected_cpr = 0.12 * 0.85  # 0.102
        assert abs(result['Stress']['cpr'] - expected_cpr) < 1e-10

    def test_upside_cdr_decreases(self, base_assumptions):
        """Upside CDR = base × (1 - upside_pct)."""
        result = build_scenarios(base_assumptions, upside_pct=0.15)
        expected_cdr = 0.08 * 0.85  # 0.068
        assert abs(result['Upside']['cdr'] - expected_cdr) < 1e-10

    def test_upside_cpr_increases(self, base_assumptions):
        """Upside CPR = base × (1 + upside_pct)."""
        result = build_scenarios(base_assumptions, upside_pct=0.15)
        expected_cpr = 0.12 * 1.15  # 0.138
        assert abs(result['Upside']['cpr'] - expected_cpr) < 1e-10

    def test_loss_severity_fixed_across_all(self, base_assumptions):
        """Loss severity should be identical in all three scenarios."""
        result = build_scenarios(base_assumptions)
        assert result['Base']['loss_severity'] == 0.85
        assert result['Stress']['loss_severity'] == 0.85
        assert result['Upside']['loss_severity'] == 0.85

    def test_custom_stress_pct(self, base_assumptions):
        """Custom stress_pct should apply correctly."""
        result = build_scenarios(base_assumptions, stress_pct=0.30)
        assert abs(result['Stress']['cdr'] - 0.08 * 1.30) < 1e-10
        assert abs(result['Stress']['cpr'] - 0.12 * 0.70) < 1e-10

    def test_custom_upside_pct(self, base_assumptions):
        """Custom upside_pct should apply correctly."""
        result = build_scenarios(base_assumptions, upside_pct=0.50)
        assert abs(result['Upside']['cdr'] - 0.08 * 0.50) < 1e-10
        assert abs(result['Upside']['cpr'] - 0.12 * 1.50) < 1e-10

    def test_zero_shift(self, base_assumptions):
        """0% shift → all three scenarios identical."""
        result = build_scenarios(base_assumptions, stress_pct=0.0, upside_pct=0.0)
        for scenario in ['Base', 'Stress', 'Upside']:
            assert result[scenario]['cdr'] == 0.08
            assert result[scenario]['cpr'] == 0.12


# ---------------------------------------------------------------------------
# compare_scenarios
# ---------------------------------------------------------------------------

class TestCompareScenarios:

    def test_returns_dataframe(self, simple_pool_chars, base_assumptions):
        scenarios = build_scenarios(base_assumptions)
        result = compare_scenarios(simple_pool_chars, scenarios, 0.95)
        assert isinstance(result, pd.DataFrame)

    def test_three_rows(self, simple_pool_chars, base_assumptions):
        scenarios = build_scenarios(base_assumptions)
        result = compare_scenarios(simple_pool_chars, scenarios, 0.95)
        assert len(result) == 3

    def test_expected_columns(self, simple_pool_chars, base_assumptions):
        scenarios = build_scenarios(base_assumptions)
        result = compare_scenarios(simple_pool_chars, scenarios, 0.95)
        expected_cols = [
            'scenario', 'cdr', 'cpr', 'loss_severity', 'irr',
            'total_interest', 'total_principal', 'total_losses',
            'total_recoveries', 'weighted_avg_life'
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_scenario_names(self, simple_pool_chars, base_assumptions):
        scenarios = build_scenarios(base_assumptions)
        result = compare_scenarios(simple_pool_chars, scenarios, 0.95)
        assert list(result['scenario']) == ['Base', 'Stress', 'Upside']

    def test_stress_irr_lower_than_base(self, simple_pool_chars, base_assumptions):
        """Stress scenario (higher defaults, lower prepayments) → lower IRR."""
        scenarios = build_scenarios(base_assumptions)
        result = compare_scenarios(simple_pool_chars, scenarios, 0.95)
        base_irr = result[result['scenario'] == 'Base']['irr'].values[0]
        stress_irr = result[result['scenario'] == 'Stress']['irr'].values[0]
        assert stress_irr < base_irr

    def test_upside_irr_higher_than_base(self, simple_pool_chars, base_assumptions):
        """Upside scenario (lower defaults, higher prepayments) → higher IRR."""
        scenarios = build_scenarios(base_assumptions)
        result = compare_scenarios(simple_pool_chars, scenarios, 0.95)
        base_irr = result[result['scenario'] == 'Base']['irr'].values[0]
        upside_irr = result[result['scenario'] == 'Upside']['irr'].values[0]
        assert upside_irr > base_irr

    def test_stress_higher_losses(self, simple_pool_chars, base_assumptions):
        """Stress scenario should have higher total losses than base."""
        scenarios = build_scenarios(base_assumptions)
        result = compare_scenarios(simple_pool_chars, scenarios, 0.95)
        base_losses = result[result['scenario'] == 'Base']['total_losses'].values[0]
        stress_losses = result[result['scenario'] == 'Stress']['total_losses'].values[0]
        assert stress_losses > base_losses

    def test_loss_severity_same_across_scenarios(self, simple_pool_chars, base_assumptions):
        """Loss severity should be identical for all scenarios."""
        scenarios = build_scenarios(base_assumptions)
        result = compare_scenarios(simple_pool_chars, scenarios, 0.95)
        severities = result['loss_severity'].unique()
        assert len(severities) == 1

    def test_weighted_avg_life_positive(self, simple_pool_chars, base_assumptions):
        """WAL should be positive for all scenarios."""
        scenarios = build_scenarios(base_assumptions)
        result = compare_scenarios(simple_pool_chars, scenarios, 0.95)
        assert (result['weighted_avg_life'] > 0).all()
