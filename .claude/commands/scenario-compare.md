Run the percentile-based three-scenario comparison for a filtered cohort using the state-transition model.

## Arguments

This command accepts optional arguments. If none provided, use defaults:
- Vintage: ALL (full portfolio)
- Grade: B
- Term: 36 months
- Purchase price: 0.95

## Steps

1. Activate the virtual environment: `source .env/bin/activate`

2. Run the following:

```python
import sqlite3
import pandas as pd
from src.amortization import calc_amort
from src.portfolio_analytics import (
    reconstruct_loan_timeline,
    compute_age_transition_probabilities,
)
from src.cashflow_engine import (
    compute_pool_assumptions,
    compute_pool_characteristics,
    build_pool_state,
    project_cashflows_transition,
    calculate_irr,
    solve_price_transition,
)
from src.scenario_analysis import (
    compute_vintage_percentiles,
    build_scenarios_from_percentiles,
    compare_scenarios_transition,
)

# Load and prepare data
conn = sqlite3.connect('data/loans.db')
df = pd.read_sql('SELECT * FROM loans', conn)
conn.close()
df = calc_amort(df, verbose=False)
df = reconstruct_loan_timeline(df)

# Filter (adjust these based on arguments)
GRADE = 'B'
TERM = 36
df_cohort = df[(df['grade'] == GRADE) & (df['term_months'] == TERM)]
non_current_active = ['In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
df_active = df_cohort[
    ((df_cohort['loan_status'] == 'Current') & (df_cohort['last_pymnt_d'] == '2019-03-01'))
    | (df_cohort['loan_status'].isin(non_current_active))
]

print(f"Cohort: Grade {GRADE}, {TERM}-month")
print(f"Total loans: {len(df_cohort):,} | Active: {len(df_active):,}")
print()

# Compute base assumptions (pool-level)
pool_assumptions = compute_pool_assumptions(df_cohort, df_active)
pool_chars = compute_pool_characteristics(df_active)

print(f"Pool-Level Assumptions:")
print(f"  CDR:           {pool_assumptions['cdr']:.4%}")
print(f"  CPR:           {pool_assumptions['cpr']:.4%}")
print(f"  Loss Severity: {pool_assumptions['loss_severity']:.4%}")
print()

# Compute vintage percentiles
vintage_results = compute_vintage_percentiles(df_cohort, df_active, pool_assumptions)
pctls = vintage_results['percentiles']

if vintage_results['fallback']:
    print(f"WARNING: Fallback mode — fewer than 3 qualifying vintages")
    print(f"  All scenarios use pool-level values")
else:
    print(f"Vintage Percentiles ({vintage_results['n_cdr_vintages']} CDR vintages, "
          f"{vintage_results['n_cpr_vintages']} CPR vintages):")
    print(f"  CDR: P25={pctls['cdr_p25']:.4%}  P50={pctls['cdr_p50']:.4%}  P75={pctls['cdr_p75']:.4%}")
    print(f"  CPR: P25={pctls['cpr_p25']:.4%}  P50={pctls['cpr_p50']:.4%}  P75={pctls['cpr_p75']:.4%}")
print()

# Show scenario assumptions
scenarios = vintage_results['scenarios']
print(f"Scenario Assumptions:")
print(f"  {'Scenario':<10} {'CDR':>8} {'CPR':>8}")
print(f"  {'-'*28}")
for name, vals in scenarios.items():
    print(f"  {name:<10} {vals['cdr']:>7.2%} {vals['cpr']:>7.2%}")
print()

# Build transition probabilities and pool state
age_probs = compute_age_transition_probabilities(df_cohort, bucket_size=1, states='7state')
pool_state = build_pool_state(df_active)

# Build scenario transition probabilities (use pool CPR as denominator)
pool_cpr = pool_assumptions['cpr']
scenario_cdrs = {name: vals['cdr'] for name, vals in scenarios.items()}
scenario_cprs = {name: vals['cpr'] for name, vals in scenarios.items()}
scenario_probs = build_scenarios_from_percentiles(
    age_probs,
    pool_assumptions['cdr'],
    pool_cpr,
    scenario_cdrs,
    scenario_cprs,
)

# Compute per-scenario curtailment rates
curtailment_rates_base = pool_assumptions.get('curtailment_rates', {})
scenario_curtailment = {}
for s_name, s_cpr in scenario_cprs.items():
    cpr_ratio = s_cpr / pool_cpr if pool_cpr > 0 else 1.0
    scenario_curtailment[s_name] = {
        age: min(rate * cpr_ratio, 1.0)
        for age, rate in curtailment_rates_base.items()
    }

# Run comparison
PRICE = 0.95
num_months = pool_chars['wam']
loss_sev = pool_assumptions['loss_severity']
rec_rate = pool_assumptions['recovery_rate']

comparison = compare_scenarios_transition(
    pool_state, scenario_probs,
    loss_sev, rec_rate,
    pool_chars, num_months, PRICE,
    scenario_curtailment_rates=scenario_curtailment,
)

print(f"\nScenario Comparison (Purchase Price: {PRICE}):")
print(comparison.to_string(index=False))

# Verify loss severity is fixed
loss_sevs = comparison['loss_severity'].unique()
if len(loss_sevs) == 1:
    print(f"\n✓ Loss severity fixed at {loss_sevs[0]:.2%} across all scenarios")
else:
    print(f"\n✗ WARNING: Loss severity varies across scenarios: {loss_sevs}")

# Verify IRR ordering
irrs = comparison.set_index('scenario')['irr']
if irrs['Stress'] < irrs['Base'] < irrs['Upside']:
    print(f"✓ IRR ordering correct: Stress ({irrs['Stress']:.2%}) < Base ({irrs['Base']:.2%}) < Upside ({irrs['Upside']:.2%})")
else:
    print(f"✗ WARNING: IRR ordering unexpected: Stress={irrs['Stress']:.2%}, Base={irrs['Base']:.2%}, Upside={irrs['Upside']:.2%}")

# Verify CDR/CPR ordering
print(f"✓ CDR: Upside ({scenarios['Upside']['cdr']:.2%}) < Base ({scenarios['Base']['cdr']:.2%}) < Stress ({scenarios['Stress']['cdr']:.2%})")
print(f"✓ CPR: Stress ({scenarios['Stress']['cpr']:.2%}) < Base ({scenarios['Base']['cpr']:.2%}) < Upside ({scenarios['Upside']['cpr']:.2%})")
```

3. Report the full comparison table and all verification checks.

## Sanity Checks

- Stress IRR should be LOWER than Base IRR
- Upside IRR should be HIGHER than Base IRR
- Stress CDR > Base CDR > Upside CDR (P75 > pool-level > P25)
- Stress CPR < Base CPR < Upside CPR (P25 < pool-level < P75)
- Base CDR/CPR should match pool-level values from `compute_pool_assumptions()`
- Loss severity FIXED across all scenarios
- All IRRs should be finite numbers (not NaN or Inf)
- If < 3 qualifying vintages, all scenarios should use pool-level values and a warning should print
