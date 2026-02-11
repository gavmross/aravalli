Demonstrate the cash flow projection pipeline end-to-end as a smoke test using the state-transition model.

## Steps

1. Activate the virtual environment: `source .env/bin/activate`

2. Run the following Python script:

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
    compute_curtailment_rates,
)

# Step 1: Load data
conn = sqlite3.connect('data/loans.db')
df = pd.read_sql('SELECT * FROM loans', conn)
conn.close()
df = calc_amort(df, verbose=False)
df = reconstruct_loan_timeline(df)

# Step 2: Filter to a sample cohort (Grade B, 36-month)
df_cohort = df[(df['grade'] == 'B') & (df['term_months'] == 36)]
non_current_active = ['In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
df_active = df_cohort[
    ((df_cohort['loan_status'] == 'Current') & (df_cohort['last_pymnt_d'] == '2019-03-01'))
    | (df_cohort['loan_status'].isin(non_current_active))
]

print(f"Total loans in cohort: {len(df_cohort):,}")
print(f"Active loans: {len(df_active):,}")

# Step 3: Compute assumptions
assumptions = compute_pool_assumptions(df_cohort, df_active)
pool_chars = compute_pool_characteristics(df_active)

print(f"\nBase Assumptions:")
print(f"  CDR:           {assumptions['cdr']:.4%}")
print(f"  CPR:           {assumptions['cpr']:.4%}")
print(f"  Loss Severity: {assumptions['loss_severity']:.4%}")
print(f"  Recovery Rate: {assumptions['recovery_rate']:.4%}")

print(f"\nPool Characteristics:")
print(f"  Total UPB:       ${pool_chars['total_upb']:,.2f}")
print(f"  WAC:             {pool_chars['wac']:.4%}")
print(f"  WAM:             {pool_chars['wam']} months")
print(f"  Monthly Payment: ${pool_chars['monthly_payment']:,.2f}")

# Step 4: Build transition probabilities and pool state
age_probs = compute_age_transition_probabilities(
    df_cohort, bucket_size=1, states='7state')
pool_state = build_pool_state(df_active)

# Step 5: Project cash flows (state-transition model)
num_months = pool_chars['wam']
curtailment_rates = assumptions.get('curtailment_rates', None)
cf_df = project_cashflows_transition(
    pool_state, age_probs,
    assumptions['loss_severity'], assumptions['recovery_rate'],
    pool_chars, num_months,
    curtailment_rates=curtailment_rates,
)

# Step 6: Calculate IRR
irr = calculate_irr(cf_df, pool_chars, purchase_price=0.95)
print(f"\nIRR at 95 cents: {irr:.4%}")

# Step 7: Summary table (first 12 months + totals)
print(f"\nFirst 12 Months:")
summary_cols = ['month', 'beginning_balance', 'defaults', 'loss', 'recovery',
                'interest', 'scheduled_principal', 'prepayments', 'total_principal',
                'ending_balance', 'total_cashflow']
print(cf_df[summary_cols].head(12).to_string(index=False,
    float_format=lambda x: f"${x:,.2f}" if abs(x) > 1 else f"{x:.6f}"))

print(f"\nTotals:")
print(f"  Total Interest:   ${cf_df['interest'].sum():,.2f}")
print(f"  Total Principal:  ${cf_df['total_principal'].sum():,.2f}")
print(f"  Total Recovery:   ${cf_df['recovery'].sum():,.2f}")
print(f"  Total Losses:     ${cf_df['loss'].sum():,.2f}")
print(f"  Total Cash Flow:  ${cf_df['total_cashflow'].sum():,.2f}")
print(f"  Final Balance:    ${cf_df['ending_balance'].iloc[-1]:,.2f}")

# Step 8: Solve for price at 12% target
price_12 = solve_price_transition(
    pool_state, age_probs,
    assumptions['loss_severity'], assumptions['recovery_rate'],
    pool_chars, num_months, target_irr=0.12,
    curtailment_rates=curtailment_rates,
)
print(f"\nPrice for 12% IRR: {price_12:.4f}")
```

3. Report all output. If any step fails, diagnose the error and report which function broke.

## What This Tests

- Data loading, calc_amort, and reconstruct_loan_timeline work together
- Filtering logic produces the right populations
- compute_pool_assumptions returns reasonable values (CDR < 50%, CPR < 50%, loss_severity between 0 and 1)
- compute_pool_characteristics returns non-zero UPB with reasonable WAC/WAM
- compute_age_transition_probabilities produces valid probability matrices (rows sum to 1.0)
- Empirical age-specific Current→Fully Paid rates vary by age (NOT constant)
- build_pool_state correctly maps loan statuses to 7-state model
- project_cashflows_transition runs to completion and ending balance reaches ~$0
- calculate_irr returns a finite, reasonable number
- solve_price_transition converges and the round-trip is consistent

## If the Cohort Is Empty

If Grade B / 36-month has no active loans, try these alternatives in order:
1. Grade C, 36-month
2. Grade B, 60-month
3. All grades, 36-month
