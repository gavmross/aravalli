Demonstrate the cash flow projection pipeline end-to-end as a smoke test.

## Steps

1. Activate the virtual environment: `source .env/bin/activate`

2. Run the following Python script:

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

# Step 1: Load data
conn = sqlite3.connect('data/loans.db')
df = pd.read_sql('SELECT * FROM loans', conn)
conn.close()
df = calc_amort(df, verbose=False)

# Step 2: Filter to a sample cohort (2018Q1, Grade B, 36-month)
df_cohort = df[(df['issue_quarter'] == '2018Q1') & (df['grade'] == 'B') & (df['term_months'] == 36)]
non_current_active = ['In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
df_active = df_cohort[
    ((df_cohort['loan_status'] == 'Current') & (df_cohort['last_pymnt_d'] == '2019-03-01'))
    | (df_cohort['loan_status'].isin(non_current_active))
]

print(f"Total loans in cohort: {len(df_cohort):,}")
print(f"Active loans: {len(df_active):,}")

# Step 3: Compute assumptions
assumptions = compute_pool_assumptions(df_cohort, df_active)
print(f"\nBase Assumptions:")
print(f"  CDR:           {assumptions['cdr']:.4%}")
print(f"  CPR:           {assumptions['cpr']:.4%}")
print(f"  Loss Severity: {assumptions['loss_severity']:.4%}")
print(f"  Recovery Rate: {assumptions['recovery_rate']:.4%}")

# Step 4: Compute pool characteristics
pool_chars = compute_pool_characteristics(df_active)
print(f"\nPool Characteristics:")
print(f"  Total UPB:       ${pool_chars['total_upb']:,.2f}")
print(f"  WAC:             {pool_chars['wac']:.4%}")
print(f"  WAM:             {pool_chars['wam']} months")
print(f"  Monthly Payment: ${pool_chars['monthly_payment']:,.2f}")

# Step 5: Project cash flows at 95 cents
cf_df = project_cashflows(pool_chars, assumptions['cdr'], assumptions['cpr'],
                          assumptions['loss_severity'], purchase_price=0.95)

# Step 6: Calculate IRR
irr = calculate_irr(cf_df, pool_chars, purchase_price=0.95)
print(f"\nIRR at 95 cents: {irr:.4%}")

# Step 7: Summary table (first 12 months + totals)
print(f"\nFirst 12 Months:")
print(cf_df.head(12).to_string(index=False,
    float_format=lambda x: f"${x:,.2f}" if abs(x) > 1 else f"{x:.6f}"))

print(f"\nTotals:")
print(f"  Total Interest:   ${cf_df['interest'].sum():,.2f}")
print(f"  Total Principal:  ${cf_df['total_principal'].sum():,.2f}")
print(f"  Total Recovery:   ${cf_df['recovery'].sum():,.2f}")
print(f"  Total Losses:     ${cf_df['loss'].sum():,.2f}")
print(f"  Total Cash Flow:  ${cf_df['total_cashflow'].sum():,.2f}")
print(f"  Final Balance:    ${cf_df['ending_balance'].iloc[-1]:,.2f}")

# Step 8: Solve for price at 12% target
price_12 = solve_price(pool_chars, target_irr=0.12,
                       cdr=assumptions['cdr'], cpr=assumptions['cpr'],
                       loss_severity=assumptions['loss_severity'])
print(f"\nPrice for 12% IRR: {price_12:.4f}")
```

3. Report all output. If any step fails, diagnose the error and report which function broke.

## What This Tests

- Data loading and calc_amort work together
- Filtering logic produces the right populations
- compute_pool_assumptions returns reasonable values (CDR < 50%, CPR < 50%, loss_severity between 0 and 1)
- compute_pool_characteristics returns non-zero UPB with reasonable WAC/WAM
- project_cashflows runs to completion and ending balance reaches ~0
- calculate_irr returns a finite, reasonable number
- solve_price converges and the round-trip is consistent

## If the Cohort Is Empty

If 2018Q1 / Grade B / 36-month has no active loans, try these alternatives in order:
1. 2017Q4, Grade B, 36-month
2. 2018Q1, Grade C, 36-month
3. 2017Q1, all grades, 36-month
