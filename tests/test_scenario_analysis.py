"""Tests for src/scenario_analysis.py"""

import numpy as np
import pandas as pd
import pytest

from src.scenario_analysis import (
    compute_base_assumptions,
    build_scenarios,
    compare_scenarios,
    build_scenarios_transition,
    compare_scenarios_transition,
)
from src.cashflow_engine import (
    project_cashflows_transition,
    calculate_irr,
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
    """Synthetic dataset for compute_base_assumptions.

    10 loans originated Jan 2017. 3 Charged Off with default_month in trailing window.
    Requires default_month, payoff_month, int_rate, term_months for conditional CDR.
    """
    return pd.DataFrame({
        'funded_amnt': [10000.0] * 10,
        'total_rec_prncp': [5000.0, 10000.0, 10000.0, 5000.0, 10000.0,
                            10000.0, 5000.0, 10000.0, 10000.0, 5000.0],
        'loan_status': [
            'Charged Off', 'Fully Paid', 'Fully Paid', 'Charged Off',
            'Fully Paid', 'Current', 'Current', 'Fully Paid',
            'Current', 'Charged Off'
        ],
        'recoveries': [1000.0, 0.0, 0.0, 500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2000.0],
        'issue_d': ['2017-01-01'] * 10,
        'int_rate': [0.10] * 10,
        'term_months': [36] * 10,
        'default_month': [
            '2018-06-01', None, None, '2018-09-01', None,
            None, None, None, None, '2019-01-01'
        ],
        'payoff_month': [
            None, '2018-12-01', '2019-02-01', None, '2018-10-01',
            None, None, '2019-01-01', None, None
        ],
    })


@pytest.fixture
def synthetic_df_active():
    """Synthetic active loans for compute_base_assumptions."""
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

    def test_returns_expected_keys(self, synthetic_df_all, synthetic_df_active):
        result = compute_base_assumptions(synthetic_df_all, synthetic_df_active)
        assert set(result.keys()) == {'cdr', 'cpr', 'loss_severity'}

    def test_no_recovery_rate_key(self, synthetic_df_all, synthetic_df_active):
        """compute_base_assumptions returns only cdr, cpr, loss_severity (no recovery_rate)."""
        result = compute_base_assumptions(synthetic_df_all, synthetic_df_active)
        assert 'recovery_rate' not in result

    def test_values_match_pool_assumptions(self, synthetic_df_all, synthetic_df_active):
        """Should match compute_pool_assumptions for cdr, cpr, loss_severity."""
        from src.cashflow_engine import compute_pool_assumptions
        full = compute_pool_assumptions(synthetic_df_all, synthetic_df_active)
        base = compute_base_assumptions(synthetic_df_all, synthetic_df_active)
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


# ---------------------------------------------------------------------------
# State-Transition Scenario Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def transition_base_probs():
    """Simple 7-state base transition probabilities."""
    rows = []
    for age in range(30):
        rows.append({
            'age_bucket': str(age),
            'from_status': 'Current',
            'to_current_pct': 0.90,
            'to_delinquent_0_30_pct': 0.06,
            'to_late_1_pct': 0.0,
            'to_late_2_pct': 0.0,
            'to_late_3_pct': 0.0,
            'to_charged_off_pct': 0.0,
            'to_fully_paid_pct': 0.04,
            'observation_count': 1000,
        })
        rows.append({
            'age_bucket': str(age),
            'from_status': 'Delinquent (0-30)',
            'to_current_pct': 0.40,
            'to_delinquent_0_30_pct': 0.0,
            'to_late_1_pct': 0.60,
            'to_late_2_pct': 0.0,
            'to_late_3_pct': 0.0,
            'to_charged_off_pct': 0.0,
            'to_fully_paid_pct': 0.0,
            'observation_count': 500,
        })
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
        rows.append({
            'age_bucket': str(age),
            'from_status': 'Late_3',
            'to_current_pct': 0.10,
            'to_delinquent_0_30_pct': 0.0,
            'to_late_1_pct': 0.0,
            'to_late_2_pct': 0.0,
            'to_late_3_pct': 0.0,
            'to_charged_off_pct': 0.90,
            'to_fully_paid_pct': 0.0,
            'observation_count': 150,
        })
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
def transition_pool_state():
    """Simple all-Current pool state."""
    return {
        'states': {
            'Current': {10: 500_000.0, 20: 500_000.0},
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
    """Pool characteristics matching transition_pool_state."""
    return {
        'total_upb': 1_000_000.0,
        'wac': 0.10,
        'wam': 30,
        'monthly_payment': 40_000.0,
    }


# ---------------------------------------------------------------------------
# build_scenarios_transition
# ---------------------------------------------------------------------------

class TestBuildScenariosTransition:

    def test_returns_three_scenarios(self, transition_base_probs):
        result = build_scenarios_transition(transition_base_probs)
        assert set(result.keys()) == {'Base', 'Stress', 'Upside'}

    def test_base_unchanged(self, transition_base_probs):
        """Base scenario should be identical to input."""
        result = build_scenarios_transition(transition_base_probs)
        pd.testing.assert_frame_equal(result['Base'], transition_base_probs)

    def test_stress_increases_delinquency(self, transition_base_probs):
        """Stress: Current→Delinquent should increase."""
        result = build_scenarios_transition(transition_base_probs, stress_pct=0.15)
        stress = result['Stress']
        base = result['Base']

        curr_rows_stress = stress[stress['from_status'] == 'Current']
        curr_rows_base = base[base['from_status'] == 'Current']

        for i in range(len(curr_rows_base)):
            assert (curr_rows_stress.iloc[i]['to_delinquent_0_30_pct']
                    > curr_rows_base.iloc[i]['to_delinquent_0_30_pct'])

    def test_stress_decreases_cure_rates(self, transition_base_probs):
        """Stress: cure rates (→Current from non-Current) should decrease."""
        result = build_scenarios_transition(transition_base_probs, stress_pct=0.15)
        stress = result['Stress']
        base = result['Base']

        for state in ['Delinquent (0-30)', 'Late_1', 'Late_2', 'Late_3']:
            state_stress = stress[stress['from_status'] == state]
            state_base = base[base['from_status'] == state]

            if len(state_base) == 0:
                continue

            for i in range(len(state_base)):
                if state_base.iloc[i]['to_current_pct'] > 0:
                    assert (state_stress.iloc[i]['to_current_pct']
                            < state_base.iloc[i]['to_current_pct'])

    def test_upside_decreases_delinquency(self, transition_base_probs):
        """Upside: Current→Delinquent should decrease."""
        result = build_scenarios_transition(transition_base_probs, upside_pct=0.15)
        upside = result['Upside']
        base = result['Base']

        curr_upside = upside[upside['from_status'] == 'Current']
        curr_base = base[base['from_status'] == 'Current']

        for i in range(len(curr_base)):
            assert (curr_upside.iloc[i]['to_delinquent_0_30_pct']
                    < curr_base.iloc[i]['to_delinquent_0_30_pct'])

    def test_upside_increases_cure_rates(self, transition_base_probs):
        """Upside: cure rates should increase."""
        result = build_scenarios_transition(transition_base_probs, upside_pct=0.15)
        upside = result['Upside']
        base = result['Base']

        for state in ['Delinquent (0-30)', 'Late_1', 'Late_2']:
            state_upside = upside[upside['from_status'] == state]
            state_base = base[base['from_status'] == state]

            if len(state_base) == 0:
                continue

            for i in range(len(state_base)):
                if state_base.iloc[i]['to_current_pct'] > 0:
                    assert (state_upside.iloc[i]['to_current_pct']
                            > state_base.iloc[i]['to_current_pct'])

    def test_rows_sum_to_one(self, transition_base_probs):
        """All rows should sum to 1.0 after stress/upside."""
        result = build_scenarios_transition(transition_base_probs, stress_pct=0.30)
        pct_cols = [c for c in result['Stress'].columns if c.endswith('_pct')]

        for scenario_name in ['Stress', 'Upside']:
            for _, row in result[scenario_name].iterrows():
                total = sum(row[c] for c in pct_cols)
                assert abs(total - 1.0) < 0.02, (
                    f"{scenario_name}: row {row['from_status']} at "
                    f"age {row['age_bucket']} sums to {total}"
                )

    def test_late3_default_not_directly_stressed(self, transition_base_probs):
        """Late_3→Charged Off should NOT be directly multiplied."""
        result = build_scenarios_transition(transition_base_probs, stress_pct=0.15)
        stress = result['Stress']
        base = result['Base']

        # In stress, Late_3→Charged Off changes only via re-normalization
        # (cure rate decreases → more goes to default via residual)
        l3_stress = stress[stress['from_status'] == 'Late_3']
        l3_base = base[base['from_status'] == 'Late_3']

        if len(l3_base) > 0:
            # Stress Late_3→Default should be >= base (not directly stressed
            # but increases because cure rate decreased)
            for i in range(len(l3_base)):
                assert (l3_stress.iloc[i]['to_charged_off_pct']
                        >= l3_base.iloc[i]['to_charged_off_pct'] - 0.001)


# ---------------------------------------------------------------------------
# compare_scenarios_transition
# ---------------------------------------------------------------------------

class TestCompareScenariosTransition:

    def test_returns_dataframe(self, transition_pool_state, transition_base_probs,
                               transition_pool_chars):
        scenario_probs = build_scenarios_transition(transition_base_probs)
        result = compare_scenarios_transition(
            transition_pool_state, scenario_probs,
            0.85, 0.15, transition_pool_chars, 30, 0.95,
        )
        assert isinstance(result, pd.DataFrame)

    def test_three_rows(self, transition_pool_state, transition_base_probs,
                        transition_pool_chars):
        scenario_probs = build_scenarios_transition(transition_base_probs)
        result = compare_scenarios_transition(
            transition_pool_state, scenario_probs,
            0.85, 0.15, transition_pool_chars, 30, 0.95,
        )
        assert len(result) == 3

    def test_scenario_irr_ordering(self, transition_pool_state,
                                    transition_base_probs,
                                    transition_pool_chars):
        """Stress IRR < Base IRR < Upside IRR."""
        scenario_probs = build_scenarios_transition(
            transition_base_probs, stress_pct=0.15, upside_pct=0.15)
        result = compare_scenarios_transition(
            transition_pool_state, scenario_probs,
            0.85, 0.15, transition_pool_chars, 30, 0.95,
        )

        base_irr = result[result['scenario'] == 'Base']['irr'].values[0]
        stress_irr = result[result['scenario'] == 'Stress']['irr'].values[0]
        upside_irr = result[result['scenario'] == 'Upside']['irr'].values[0]

        assert stress_irr < base_irr, f"Stress IRR {stress_irr} >= Base {base_irr}"
        assert upside_irr > base_irr, f"Upside IRR {upside_irr} <= Base {base_irr}"

    def test_loss_severity_fixed(self, transition_pool_state,
                                 transition_base_probs, transition_pool_chars):
        """Loss severity should be identical across all scenarios."""
        scenario_probs = build_scenarios_transition(transition_base_probs)
        result = compare_scenarios_transition(
            transition_pool_state, scenario_probs,
            0.85, 0.15, transition_pool_chars, 30, 0.95,
        )
        severities = result['loss_severity'].unique()
        assert len(severities) == 1
        assert severities[0] == 0.85

    def test_stress_higher_losses(self, transition_pool_state,
                                  transition_base_probs, transition_pool_chars):
        """Stress scenario should have higher total losses than base."""
        scenario_probs = build_scenarios_transition(
            transition_base_probs, stress_pct=0.15)
        result = compare_scenarios_transition(
            transition_pool_state, scenario_probs,
            0.85, 0.15, transition_pool_chars, 30, 0.95,
        )

        base_losses = result[result['scenario'] == 'Base']['total_losses'].values[0]
        stress_losses = result[result['scenario'] == 'Stress']['total_losses'].values[0]
        assert stress_losses > base_losses
