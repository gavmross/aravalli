Run the three-scenario comparison for a filtered cohort and display results.

## Arguments

This command accepts optional arguments. If none provided, use defaults:
- Vintage: 2018Q1 (or first available vintage with sufficient Current loans)
- Grade: All grades
- Term: 36 months
- Purchase price: 0.95

## Steps

1. Activate the virtual environment: `source .env/bin/activate`

2. Run the following:

```python
import sqlite3
import pandas as pd
from src.amortization import calc_amort
from src.cashflow_engine import (
    compute_pool_assumptions,
    compute_pool_characteristics,
    project_cashflows,
    calculate_irr,
    solve_price
)
from src.scenario_analysis import build_scenarios, compare_scenarios

# Load and prepare data
conn = sqlite3.connect('data/loans.db')
df = pd.read_sql('SELECT * FROM loans', conn)
conn.close()
df = calc_amort(df, verbose=False)

# Filter (adjust these based on arguments)
VINTAGE = '2018Q1'
TERM = 36
df_cohort = df[(df['issue_quarter'] == VINTAGE) & (df['term_months'] == TERM)]
df_current = df_cohort[
    (df_cohort['loan_status'] == 'Current') &
    (df_cohort['last_pymnt_d'] == '2019-03-01')
]

print(f"Cohort: {VINTAGE}, {TERM}-month")
print(f"Total loans: {len(df_cohort):,} | Current (March 2019): {len(df_current):,}")
print()

# Compute base assumptions
assumptions = compute_pool_assumptions(df_cohort, df_current)
pool_chars = compute_pool_characteristics(df_current)

# Build scenarios (default multipliers)
scenarios = build_scenarios(assumptions)

# Compare at purchase price = 0.95
PRICE = 0.95
comparison = compare_scenarios(scenarios, pool_chars, PRICE)

# Print comparison table
print(f"Purchase Price: {PRICE}")
print()
print(f"{'Scenario':<10} {'CDR':>8} {'CPR':>8} {'Loss Sev':>10} {'IRR':>8} {'Price for 12%':>14}")
print("-" * 62)
for _, row in comparison.iterrows():
    # Solve price for 12% IRR per scenario
    p12 = solve_price(pool_chars, target_irr=0.12,
                      cdr=row['cdr'], cpr=row['cpr'],
                      loss_severity=row['loss_severity'])
    print(f"{row['scenario']:<10} {row['cdr']:>7.2%} {row['cpr']:>7.2%} "
          f"{row['loss_severity']:>9.2%} {row['irr']:>7.2%} {p12:>13.4f}")

print()

# Verify loss severity is fixed
loss_sevs = comparison['loss_severity'].unique()
if len(loss_sevs) == 1:
    print(f"✓ Loss severity fixed at {loss_sevs[0]:.2%} across all scenarios")
else:
    print(f"✗ WARNING: Loss severity varies across scenarios: {loss_sevs}")

# Verify multipliers
base_cdr = comparison.loc[comparison['scenario'] == 'Base', 'cdr'].iloc[0]
stress_cdr = comparison.loc[comparison['scenario'] == 'Stress', 'cdr'].iloc[0]
upside_cdr = comparison.loc[comparison['scenario'] == 'Upside', 'cdr'].iloc[0]
print(f"✓ Stress CDR mult: {stress_cdr/base_cdr:.2f}x (expected 1.50x)")
print(f"✓ Upside CDR mult: {upside_cdr/base_cdr:.2f}x (expected 0.50x)")
```

3. Report the full comparison table and all verification checks.

## Expected Output Shape

```
Cohort: 2018Q1, 36-month
Total loans: XX,XXX | Current (March 2019): X,XXX

Purchase Price: 0.95

Scenario   CDR      CPR    Loss Sev       IRR  Price for 12%
--------------------------------------------------------------
Base       X.XX%   X.XX%     XX.XX%    XX.XX%        X.XXXX
Stress     X.XX%   X.XX%     XX.XX%    XX.XX%        X.XXXX
Upside     X.XX%   X.XX%     XX.XX%    XX.XX%        X.XXXX

✓ Loss severity fixed at XX.XX% across all scenarios
✓ Stress CDR mult: 1.50x (expected 1.50x)
✓ Upside CDR mult: 0.50x (expected 0.50x)
```

## Sanity Checks

- Stress IRR should be LOWER than Base IRR
- Upside IRR should be HIGHER than Base IRR
- Price for 12% should be highest in Upside (can afford to pay more) and lowest in Stress
- All IRRs should be finite numbers (not NaN or Inf)
