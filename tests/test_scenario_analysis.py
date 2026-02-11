"""Tests for src/scenario_analysis.py"""

import numpy as np
import pandas as pd
import pytest

from src.scenario_analysis import (
    compute_vintage_percentiles,
    compute_base_assumptions,
    build_scenarios,
    compare_scenarios,
    build_scenarios_transition,
    build_scenarios_from_percentiles,
    compare_scenarios_transition,
)
from src.cashflow_engine import (
    compute_pool_assumptions,
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
        'out_prncp': [0.0, 0.0, 0.0, 0.0, 0.0,
                      3100.0, 3100.0, 0.0, 3100.0, 0.0],
        'default_month': [
            '2018-06-01', None, None, '2018-09-01', None,
            None, None, None, None, '2019-01-01'
        ],
        'payoff_month': [
            None, '2018-12-01', '2019-02-01', None, '2018-10-01',
            None, None, '2019-01-01', None, None
        ],
        'last_pymnt_d': [
            '2018-06-01', '2018-12-01', '2019-02-01', '2018-09-01', '2018-10-01',
            '2019-03-01', '2019-03-01', '2019-01-01', '2019-03-01', '2019-01-01'
        ],
    })


@pytest.fixture
def synthetic_df_active():
    """Synthetic active loans for compute_base_assumptions."""
    return pd.DataFrame({
        'loan_status': ['Current', 'Current', 'Current'],
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


# ---------------------------------------------------------------------------
# build_scenarios_from_percentiles
# ---------------------------------------------------------------------------

class TestBuildScenariosFromPercentiles:

    def test_returns_three_scenarios(self, transition_base_probs):
        """Should return Base, Stress, Upside keys."""
        cdrs = {'Base': 0.08, 'Stress': 0.12, 'Upside': 0.05}
        cprs = {'Base': 0.015, 'Stress': 0.010, 'Upside': 0.020}
        result = build_scenarios_from_percentiles(
            transition_base_probs, pool_cdr=0.08,
            base_cpr=1 - (1 - 0.04) ** 12,
            scenario_cdrs=cdrs, scenario_cprs=cprs,
        )
        assert set(result.keys()) == {'Base', 'Stress', 'Upside'}

    def test_rows_sum_to_one(self, transition_base_probs):
        """All rows in each scenario should sum to 1.0."""
        cdrs = {'Base': 0.08, 'Stress': 0.12, 'Upside': 0.05}
        cprs = {'Base': 0.015, 'Stress': 0.010, 'Upside': 0.020}
        result = build_scenarios_from_percentiles(
            transition_base_probs, pool_cdr=0.08,
            base_cpr=1 - (1 - 0.04) ** 12,
            scenario_cdrs=cdrs, scenario_cprs=cprs,
        )
        pct_cols = [c for c in transition_base_probs.columns if c.endswith('_pct')]

        for name, probs_df in result.items():
            for _, row in probs_df.iterrows():
                total = sum(row[c] for c in pct_cols)
                assert abs(total - 1.0) < 0.02, (
                    f"{name}: {row['from_status']} at age {row['age_bucket']} "
                    f"sums to {total}"
                )

    def test_stress_higher_delinquency(self, transition_base_probs):
        """Stress (higher CDR ratio) → higher Current→Delinquent rate."""
        cdrs = {'Base': 0.08, 'Stress': 0.12, 'Upside': 0.05}
        cprs = {'Base': 0.015, 'Stress': 0.010, 'Upside': 0.020}
        result = build_scenarios_from_percentiles(
            transition_base_probs, pool_cdr=0.08,
            base_cpr=1 - (1 - 0.04) ** 12,
            scenario_cdrs=cdrs, scenario_cprs=cprs,
        )
        base_curr = result['Base'][result['Base']['from_status'] == 'Current']
        stress_curr = result['Stress'][result['Stress']['from_status'] == 'Current']

        for i in range(len(base_curr)):
            assert (stress_curr.iloc[i]['to_delinquent_0_30_pct']
                    > base_curr.iloc[i]['to_delinquent_0_30_pct'])

    def test_stress_lower_cure_rates(self, transition_base_probs):
        """Stress → lower cure rates for non-Current states."""
        cdrs = {'Base': 0.08, 'Stress': 0.12, 'Upside': 0.05}
        cprs = {'Base': 0.015, 'Stress': 0.010, 'Upside': 0.020}
        result = build_scenarios_from_percentiles(
            transition_base_probs, pool_cdr=0.08,
            base_cpr=1 - (1 - 0.04) ** 12,
            scenario_cdrs=cdrs, scenario_cprs=cprs,
        )
        for state in ['Delinquent (0-30)', 'Late_1', 'Late_2']:
            base_rows = result['Base'][result['Base']['from_status'] == state]
            stress_rows = result['Stress'][result['Stress']['from_status'] == state]

            for i in range(len(base_rows)):
                if base_rows.iloc[i]['to_current_pct'] > 0:
                    assert (stress_rows.iloc[i]['to_current_pct']
                            < base_rows.iloc[i]['to_current_pct'])

    def test_late3_default_not_directly_scaled(self, transition_base_probs):
        """Late_3→Charged Off increases via re-normalization, not direct scaling."""
        cdrs = {'Base': 0.08, 'Stress': 0.12, 'Upside': 0.05}
        cprs = {'Base': 0.015, 'Stress': 0.010, 'Upside': 0.020}
        result = build_scenarios_from_percentiles(
            transition_base_probs, pool_cdr=0.08,
            base_cpr=1 - (1 - 0.04) ** 12,
            scenario_cdrs=cdrs, scenario_cprs=cprs,
        )
        l3_stress = result['Stress'][result['Stress']['from_status'] == 'Late_3']
        l3_base = result['Base'][result['Base']['from_status'] == 'Late_3']

        for i in range(len(l3_base)):
            # Should increase (via less cure → more default in residual)
            assert (l3_stress.iloc[i]['to_charged_off_pct']
                    >= l3_base.iloc[i]['to_charged_off_pct'] - 0.001)

    def test_different_cpr_per_scenario(self, transition_base_probs):
        """Each scenario should have different Current→Fully Paid rates."""
        cdrs = {'Base': 0.08, 'Stress': 0.08, 'Upside': 0.08}  # same CDR
        cprs = {'Base': 0.015, 'Stress': 0.005, 'Upside': 0.030}  # different CPR
        result = build_scenarios_from_percentiles(
            transition_base_probs, pool_cdr=0.08,
            base_cpr=1 - (1 - 0.04) ** 12,
            scenario_cdrs=cdrs, scenario_cprs=cprs,
        )
        # Get first Current row for each scenario
        base_fp = result['Base'][
            result['Base']['from_status'] == 'Current'
        ].iloc[0]['to_fully_paid_pct']
        stress_fp = result['Stress'][
            result['Stress']['from_status'] == 'Current'
        ].iloc[0]['to_fully_paid_pct']
        upside_fp = result['Upside'][
            result['Upside']['from_status'] == 'Current'
        ].iloc[0]['to_fully_paid_pct']

        # Upside (highest CPR) → highest prepayment rate
        assert upside_fp > base_fp
        assert base_fp > stress_fp

    def test_pool_cdr_zero_no_crash(self, transition_base_probs):
        """pool_cdr=0 → CDR scaling skipped (ratio=1.0), CPR still works."""
        cdrs = {'Base': 0.0, 'Stress': 0.05, 'Upside': 0.0}
        cprs = {'Base': 0.015, 'Stress': 0.010, 'Upside': 0.020}
        result = build_scenarios_from_percentiles(
            transition_base_probs, pool_cdr=0.0,
            base_cpr=1 - (1 - 0.04) ** 12,
            scenario_cdrs=cdrs, scenario_cprs=cprs,
        )
        # Should not crash, and return valid probability DataFrames
        assert set(result.keys()) == {'Base', 'Stress', 'Upside'}
        pct_cols = [c for c in transition_base_probs.columns if c.endswith('_pct')]
        for name, probs_df in result.items():
            for _, row in probs_df.iterrows():
                total = sum(row[c] for c in pct_cols)
                assert abs(total - 1.0) < 0.02

    def test_integration_irr_ordering(self, transition_base_probs,
                                       transition_pool_state,
                                       transition_pool_chars):
        """Stress IRR < Base IRR < Upside IRR (integration test)."""
        cdrs = {'Base': 0.08, 'Stress': 0.12, 'Upside': 0.05}
        cprs = {'Base': 0.015, 'Stress': 0.010, 'Upside': 0.020}
        scenario_probs = build_scenarios_from_percentiles(
            transition_base_probs, pool_cdr=0.08,
            base_cpr=1 - (1 - 0.04) ** 12,
            scenario_cdrs=cdrs, scenario_cprs=cprs,
        )
        comparison = compare_scenarios_transition(
            transition_pool_state, scenario_probs,
            0.85, 0.15, transition_pool_chars, 30, 0.95,
        )
        base_irr = comparison[comparison['scenario'] == 'Base']['irr'].values[0]
        stress_irr = comparison[comparison['scenario'] == 'Stress']['irr'].values[0]
        upside_irr = comparison[comparison['scenario'] == 'Upside']['irr'].values[0]

        assert stress_irr < base_irr, f"Stress {stress_irr} >= Base {base_irr}"
        assert upside_irr > base_irr, f"Upside {upside_irr} <= Base {base_irr}"


# ---------------------------------------------------------------------------
# compute_vintage_percentiles
# ---------------------------------------------------------------------------

def _make_vintage_cohort(
    vintage: str,
    n_total: int,
    n_defaults: int,
    issue_d: str,
    out_prncp_current: float = 3100.0,
    recovery_per_default: float = 500.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Helper: build a synthetic (df_all, df_active) pair for one vintage.

    Defaults spread across trailing 12 months (Apr 2018 – Mar 2019).
    Remaining loans are Current with March 2019 last payment date.
    """
    n_current = n_total - n_defaults

    funded = [10000.0] * n_total
    rec_prncp = [0.0] * n_defaults + [5000.0] * n_current
    int_rate = [0.10] * n_total
    term_months = [36] * n_total
    recoveries = [recovery_per_default] * n_defaults + [0.0] * n_current
    out_prncp = [0.0] * n_defaults + [out_prncp_current] * n_current

    loan_status = ['Charged Off'] * n_defaults + ['Current'] * n_current

    snapshot = pd.Timestamp('2019-03-01')
    default_months = []
    for i in range(n_defaults):
        m = (i % 12) + 1
        month_start = snapshot - pd.DateOffset(months=m)
        default_months.append(month_start.strftime('%Y-%m-%d'))
    default_months.extend([None] * n_current)

    payoff_months = [None] * n_total
    last_pymnt_d = ['2018-01-01'] * n_defaults + ['2019-03-01'] * n_current

    last_pmt_beginning_balance = [0.0] * n_defaults + [5200.0] * n_current
    last_pmt_scheduled_principal = [0.0] * n_defaults + [200.0] * n_current
    last_pmt_unscheduled_principal = [0.0] * n_defaults + [50.0] * n_current

    df_all = pd.DataFrame({
        'funded_amnt': funded,
        'total_rec_prncp': rec_prncp,
        'loan_status': loan_status,
        'recoveries': recoveries,
        'issue_d': [issue_d] * n_total,
        'int_rate': int_rate,
        'term_months': term_months,
        'out_prncp': out_prncp,
        'default_month': default_months,
        'payoff_month': payoff_months,
        'last_pymnt_d': last_pymnt_d,
        'issue_quarter': [vintage] * n_total,
        'last_pmt_beginning_balance': last_pmt_beginning_balance,
        'last_pmt_scheduled_principal': last_pmt_scheduled_principal,
        'last_pmt_unscheduled_principal': last_pmt_unscheduled_principal,
    })

    df_active = df_all[df_all['loan_status'] == 'Current'].copy()
    return df_all, df_active


def _build_multi_vintage_data(n_loans=200):
    """Build 4 vintages with different CDR levels."""
    configs = [
        ('2017-Q1', n_loans, int(n_loans * 0.05), '2017-01-01'),
        ('2017-Q2', n_loans, int(n_loans * 0.10), '2017-04-01'),
        ('2017-Q3', n_loans, int(n_loans * 0.15), '2017-07-01'),
        ('2017-Q4', n_loans, int(n_loans * 0.20), '2017-10-01'),
    ]
    all_dfs, active_dfs = [], []
    for vintage, n_total, n_defaults, issue_d in configs:
        df_a, df_act = _make_vintage_cohort(vintage, n_total, n_defaults, issue_d)
        all_dfs.append(df_a)
        active_dfs.append(df_act)

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_active = pd.concat(active_dfs, ignore_index=True)
    return df_all, df_active


@pytest.fixture
def multi_vintage_data():
    """4 vintages, 200 loans each, varying default counts."""
    return _build_multi_vintage_data(n_loans=200)


@pytest.fixture
def multi_vintage_pool_assumptions(multi_vintage_data):
    """Pool-level assumptions for the multi-vintage data."""
    df_all, df_active = multi_vintage_data
    return compute_pool_assumptions(df_all, df_active)


class TestComputeVintagePercentiles:

    def test_returns_expected_keys(self, multi_vintage_data, multi_vintage_pool_assumptions):
        df_all, df_active = multi_vintage_data
        result = compute_vintage_percentiles(
            df_all, df_active, multi_vintage_pool_assumptions,
            min_loans_cdr=100, min_loans_cpr=100)
        expected_keys = {'fallback', 'vintage_cdrs', 'vintage_cprs',
                         'n_cdr_vintages', 'n_cpr_vintages',
                         'percentiles', 'scenarios', 'vintage_data'}
        assert set(result.keys()) == expected_keys

    def test_correct_qualifying_vintage_count(self, multi_vintage_data,
                                               multi_vintage_pool_assumptions):
        """4 vintages, each with 200 loans (>= 100 min_loans) → 4 qualifying."""
        df_all, df_active = multi_vintage_data
        result = compute_vintage_percentiles(
            df_all, df_active, multi_vintage_pool_assumptions,
            min_loans_cdr=100, min_loans_cpr=100)
        assert result['n_cdr_vintages'] == 4
        assert not result['fallback']

    def test_percentile_ordering(self, multi_vintage_data, multi_vintage_pool_assumptions):
        """cdr_p25 <= cdr_p50 <= cdr_p75, same for CPR."""
        df_all, df_active = multi_vintage_data
        result = compute_vintage_percentiles(
            df_all, df_active, multi_vintage_pool_assumptions,
            min_loans_cdr=100, min_loans_cpr=100)
        p = result['percentiles']
        assert p['cdr_p25'] <= p['cdr_p50'] <= p['cdr_p75']
        assert p['cpr_p25'] <= p['cpr_p50'] <= p['cpr_p75']

    def test_fallback_when_few_vintages(self, multi_vintage_data,
                                         multi_vintage_pool_assumptions):
        """min_loans=300 excludes all vintages (each has 200) → fallback."""
        df_all, df_active = multi_vintage_data
        result = compute_vintage_percentiles(
            df_all, df_active, multi_vintage_pool_assumptions,
            min_loans_cdr=300, min_loans_cpr=300)
        assert result['fallback'] is True
        assert result['percentiles'] is None
        assert result['scenarios'] is None

    def test_min_loans_filter(self, multi_vintage_data, multi_vintage_pool_assumptions):
        """With min_loans=201, no vintage qualifies (each has exactly 200)."""
        df_all, df_active = multi_vintage_data
        result = compute_vintage_percentiles(
            df_all, df_active, multi_vintage_pool_assumptions,
            min_loans_cdr=201, min_loans_cpr=201)
        assert result['fallback'] is True
        assert result['n_cdr_vintages'] == 0

    def test_zero_default_vintage_valid(self):
        """A vintage with zero defaults has CDR=0 and still qualifies."""
        configs = [
            ('2017-Q1', 200, 0, '2017-01-01'),
            ('2017-Q2', 200, 20, '2017-04-01'),
            ('2017-Q3', 200, 30, '2017-07-01'),
            ('2017-Q4', 200, 40, '2017-10-01'),
        ]
        all_dfs, active_dfs = [], []
        for vintage, n_total, n_defaults, issue_d in configs:
            df_a, df_act = _make_vintage_cohort(vintage, n_total, n_defaults, issue_d)
            all_dfs.append(df_a)
            active_dfs.append(df_act)

        df_all = pd.concat(all_dfs, ignore_index=True)
        df_active = pd.concat(active_dfs, ignore_index=True)
        pool_assumptions = compute_pool_assumptions(df_all, df_active)

        result = compute_vintage_percentiles(
            df_all, df_active, pool_assumptions,
            min_loans_cdr=100, min_loans_cpr=100)
        assert result['n_cdr_vintages'] == 4
        assert not result['fallback']
        assert result['percentiles']['cdr_p25'] >= 0.0

    def test_base_uses_pool_level(self, multi_vintage_data, multi_vintage_pool_assumptions):
        """Base scenario uses pool-level CDR/CPR, not P50 median."""
        df_all, df_active = multi_vintage_data
        result = compute_vintage_percentiles(
            df_all, df_active, multi_vintage_pool_assumptions,
            min_loans_cdr=100, min_loans_cpr=100)
        s = result['scenarios']
        assert abs(s['Base']['cdr'] - multi_vintage_pool_assumptions['cdr']) < 1e-10
        assert abs(s['Base']['cpr'] - multi_vintage_pool_assumptions['cpr']) < 1e-10

    def test_stress_upside_use_percentiles(self, multi_vintage_data,
                                            multi_vintage_pool_assumptions):
        """Stress=P75 CDR/P25 CPR, Upside=P25 CDR/P75 CPR."""
        df_all, df_active = multi_vintage_data
        result = compute_vintage_percentiles(
            df_all, df_active, multi_vintage_pool_assumptions,
            min_loans_cdr=100, min_loans_cpr=100)
        p = result['percentiles']
        s = result['scenarios']

        assert abs(s['Stress']['cdr'] - p['cdr_p75']) < 1e-10
        assert abs(s['Stress']['cpr'] - p['cpr_p25']) < 1e-10
        assert abs(s['Upside']['cdr'] - p['cdr_p25']) < 1e-10
        assert abs(s['Upside']['cpr'] - p['cpr_p75']) < 1e-10

    def test_vintage_data_dataframe(self, multi_vintage_data,
                                     multi_vintage_pool_assumptions):
        """vintage_data should be a DataFrame with expected columns."""
        df_all, df_active = multi_vintage_data
        result = compute_vintage_percentiles(
            df_all, df_active, multi_vintage_pool_assumptions,
            min_loans_cdr=100, min_loans_cpr=100)
        vd = result['vintage_data']
        assert isinstance(vd, pd.DataFrame)
        assert set(vd.columns) >= {'vintage', 'cdr', 'cpr', 'loan_count'}
        assert len(vd) == 4


# ---------------------------------------------------------------------------
# compare_scenarios_transition with curtailment rates
# ---------------------------------------------------------------------------

class TestCompareScenariosTransitionCurtailments:

    def test_none_curtailments_backward_compat(self, transition_pool_state,
                                                transition_base_probs,
                                                transition_pool_chars):
        """scenario_curtailment_rates=None → backward compatible (no crash)."""
        scenario_probs = build_scenarios_transition(transition_base_probs)
        result = compare_scenarios_transition(
            transition_pool_state, scenario_probs,
            0.85, 0.15, transition_pool_chars, 30, 0.95,
            scenario_curtailment_rates=None,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_curtailments_increase_total_principal(self, transition_pool_state,
                                                    transition_base_probs,
                                                    transition_pool_chars):
        """With curtailment rates, total_principal should increase."""
        scenario_probs = build_scenarios_transition(transition_base_probs)

        result_no_curt = compare_scenarios_transition(
            transition_pool_state, scenario_probs,
            0.85, 0.15, transition_pool_chars, 30, 0.95,
            scenario_curtailment_rates=None,
        )

        curt_rates = {a: 0.01 for a in range(60)}
        scenario_curt = {name: curt_rates for name in scenario_probs}
        result_with_curt = compare_scenarios_transition(
            transition_pool_state, scenario_probs,
            0.85, 0.15, transition_pool_chars, 30, 0.95,
            scenario_curtailment_rates=scenario_curt,
        )

        base_no = result_no_curt[result_no_curt['scenario'] == 'Base']['total_principal'].values[0]
        base_with = result_with_curt[result_with_curt['scenario'] == 'Base']['total_principal'].values[0]
        assert base_with > base_no
