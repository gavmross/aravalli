# Update Instructions — State-Transition Cash Flow Model (7-State)

Add a state-transition cash flow engine alongside the existing flat CDR/CPR model. The user toggles between "Simple" (flat) and "Advanced" (state-transition) in the dashboard. Do not modify the existing flat model — it stays as-is.

---

## Update 1: CLAUDE.md — Add State-Transition Model to Domain Terminology

Find the `### Rate Conversions` section. Add this new section immediately AFTER it:

```markdown
### Cash Flow Model Types

**Simple (Flat CDR/CPR)**: The existing model. A constant CDR and CPR are applied uniformly to the total pool balance each month. Every dollar in the pool is treated identically regardless of loan status or age. Fast, conventional, easy to stress test. Uses `project_cashflows()`.

**Advanced (State-Transition)**: Tracks pool balance across 7 states and applies age-specific empirical transition probabilities each month. More realistic — an all-Current pool has zero defaults until month ~5 because loans must flow through the full delinquency pipeline before reaching Default.

**7 model states**:

| State | Generates Cash? | Can Prepay? | Can Cure to Current? | Can Roll to Default? | Terminal? |
|-------|----------------|-------------|---------------------|---------------------|-----------|
| Current | Yes — interest + scheduled principal | Yes | N/A | No | No |
| Delinquent (0-30) | No — missed payment | No | Yes | No | No |
| Late_1 (days ~31-60) | No | No | Yes | No | No |
| Late_2 (days ~61-90) | No | No | Yes | No | No |
| Late_3 (days ~91-120) | No | No | Yes | Yes — only state that can | No |
| Default / Charged Off | Recoveries only | No | No | N/A | Yes (absorbing) |
| Fully Paid | Lump sum at prepayment | N/A | No | N/A | Yes (absorbing) |

**Transition rules**:

```
Current          → Current, Delinquent (0-30), Fully Paid
Delinquent (0-30)→ Current (cure), Late_1
Late_1           → Current (cure), Late_2
Late_2           → Current (cure), Late_3
Late_3           → Current (cure), Default
Default          → Default                    (absorbing)
Fully Paid       → Fully Paid                 (absorbing)
```

**Key rules**:
- Only **Current** loans generate interest and principal cash flows
- Only **Current** loans can prepay (all other non-terminal states are by definition not paying)
- **Loans cannot skip states**: Current → Default is impossible. Current → Late_1 is impossible. Must follow the pipeline.
- **Late sub-states represent months in Late (31-120)**: Late_1 is the first month at 31+ days, Late_2 is the second month, Late_3 is the third month. Only Late_3 can roll to Default.
- **All non-terminal states can cure to Current**: Delinquent, Late_1, Late_2, and Late_3 can all cure back to Current. The cure probability generally decreases the deeper the delinquency (empirical from data).
- When a loan cures back to Current, it resumes generating cash flows the next month
- Default and Fully Paid are **absorbing states** — once entered, loans stay permanently
- Transition probabilities are **age-specific** (from `compute_age_transition_probabilities()`)

**Minimum pipeline from Current to Default: 5 months**:
```
Month 1: Current → Delinquent (0-30)   [missed payment]
Month 2: Delinquent → Late_1           [31+ days past due]
Month 3: Late_1 → Late_2              [61+ days]
Month 4: Late_2 → Late_3              [91+ days]
Month 5: Late_3 → Default             [120+ days, terminal]
```

**Monthly loop**:
```
For each month t:
  For each age bucket:
    1. Look up age-specific transition probabilities for this bucket
    2. Apply transitions to UPB in each state:
       Current_UPB      → stays Current, goes Delinquent, prepays to Fully Paid
       Delinquent_UPB   → cures to Current, rolls to Late_1
       Late_1_UPB       → cures to Current, rolls to Late_2
       Late_2_UPB       → cures to Current, rolls to Late_3
       Late_3_UPB       → cures to Current, rolls to Default
       Default_UPB      → stays Default (absorbing)
       Fully_Paid_UPB   → stays Fully Paid (absorbing)
    3. Cash flows from Current loans only:
       interest     = Current_UPB × WAC / 12
       sched_princ  = standard amortization principal for this age/WAC/WAM
       prepayments  = Current_UPB × (Current → Fully Paid transition rate)
    4. Losses from new defaults:
       new_defaults = Late_3_UPB × (Late_3 → Default transition rate)
       losses       = new_defaults × loss_severity
       recoveries   = new_defaults × recovery_rate
    5. Age all loans by 1 month
  Sum across all age buckets for pool-level cash flow at month t
```
```

---

## Update 2: CLAUDE.md — Update Critical Rules

Find this line:
```markdown
- **Cash flows and scenarios operate on Current loans with March 2019 last payment date ONLY.**
```

Replace with:
```markdown
- **Simple model**: Cash flows operate on Current loans with March 2019 last payment date only (unchanged).
- **Advanced (state-transition) model**: Cash flows operate on user-selected pool composition — either Current loans only OR all active loans (Current + Grace Period + Late 16-30 + Late 31-120). User selects via dashboard toggle.
- **`compute_pool_assumptions()`** always uses ALL loans in the cohort for CDR/loss severity (measuring historical rates, not the purchase pool). CPR uses Current March 2019 loans only.
- **`compute_pool_characteristics()`** computes WAC/WAM/WALA on whichever pool the user has selected.
```

---

## Update 3: CLAUDE.md — Update Backsolve Reconstruction for 7 States

Find the backsolve timeline section in `### Data Context — Snapshot with Backsolve Capability` (if it exists from update_prompt_v5). If it references 5 states, update the state table to show 7 states. If it doesn't exist yet, skip this — update_prompt_v5 will add it and you can manually adjust.

If the state table exists, replace it with:
```markdown
**Transition matrix states** (7 states at monthly granularity):

| State | Can transition to | Notes |
|-------|------------------|-------|
| Current | Current, Fully Paid, Delinquent (0-30) | Performing loan |
| Delinquent (0-30) | Current (cure), Late_1 | Grace + Late 16-30 combined (same month) |
| Late_1 (month 1 of 31-120) | Current (cure), Late_2 | Days ~31-60 |
| Late_2 (month 2 of 31-120) | Current (cure), Late_3 | Days ~61-90 |
| Late_3 (month 3 of 31-120) | Current (cure), Default | Days ~91-120. Only state that can roll to Default |
| Default / Charged Off | Default | Terminal / absorbing state |
| Fully Paid | Fully Paid | Terminal / absorbing state |

Note: Tab 1 display combines Late_1/Late_2/Late_3 into "Late (31-120)" for readability. The 7-state granularity is used internally by the state-transition cash flow engine only.
```

---

## Update 4: CLAUDE.md — Update reconstruct_loan_timeline for Late Sub-States

Find the `reconstruct_loan_timeline` function spec. Add these columns to the logic (after the existing `late_31_120_month` calculation):

```markdown
Add Late sub-state month assignments for loans in Late (31-120) or Charged Off:

```python
# Late sub-states: Late_1 starts at late_31_120_month, each subsequent month advances
# Late_1 = first month at 31+ days (same as late_31_120_month)
# Late_2 = late_31_120_month + 1
# Late_3 = late_31_120_month + 2
# Default can only happen after Late_3 (at late_31_120_month + 3 = delinquent_month + 4)

late_mask = df['late_31_120_month'].notna()
df.loc[late_mask, 'late_1_month'] = df.loc[late_mask, 'late_31_120_month']
df.loc[late_mask, 'late_2_month'] = df.loc[late_mask, 'late_31_120_month'] + pd.DateOffset(months=1)
df.loc[late_mask, 'late_3_month'] = df.loc[late_mask, 'late_31_120_month'] + pd.DateOffset(months=2)

# Compute loan age at each sub-state
for col in ['late_1_month', 'late_2_month', 'late_3_month']:
    age_col = col.replace('_month', '_age')
    mask = df[col].notna()
    df.loc[mask, age_col] = (
        (df.loc[mask, col] - df.loc[mask, 'issue_d']).dt.days / 30.44
    ).round().astype(int)
```

**Updated Returns**: Add to the list of added columns:
`late_1_month`, `late_2_month`, `late_3_month`, `late_1_age`, `late_2_age`, `late_3_age`
```

---

## Update 5: CLAUDE.md — Update get_loan_status_at_age for 7 States

Find the `get_loan_status_at_age` function spec. Replace its logic with:

```markdown
**Logic** (evaluate in this order — first match wins):
```python
# Convert loan age to reference month
ref_month = df['issue_d'] + pd.DateOffset(months=age)

# Terminal states (absorbing)
if ref_month >= default_month:           'Default'
elif ref_month >= payoff_month:          'Fully Paid'
# Late sub-states (check in reverse order: Late_3 first)
elif ref_month >= late_3_month:          'Late_3'
elif ref_month >= late_2_month:          'Late_2'
elif ref_month >= late_1_month:          'Late_1'
# Delinquent
elif ref_month == delinquent_month:      'Delinquent (0-30)'
# Not yet originated or still performing
elif age > loan_age_months:              None
else:                                    'Current'
```

Handle NaT: if a transition column is NaT, that transition never happened.

Note: For Tab 1 display purposes (`compute_transition_matrix` in update_prompt_v5), Late_1/Late_2/Late_3 are combined into 'Late (31-120)'. For the state-transition cash flow engine, the 7-state version is used.
```

---

## Update 6: CLAUDE.md — Update compute_age_transition_probabilities for 7 States

Find the `compute_age_transition_probabilities` function spec. Add this note:

```markdown
**7-state mode** (for state-transition cash flow engine):

When called with `states='7state'` parameter, compute transition probabilities across all 7 states instead of 5. The additional columns in the output are:

- For `from_status = 'Delinquent (0-30)'`: `to_current_pct` (cure), `to_late_1_pct` (roll)
- For `from_status = 'Late_1'`: `to_current_pct` (cure), `to_late_2_pct` (roll)
- For `from_status = 'Late_2'`: `to_current_pct` (cure), `to_late_3_pct` (roll)
- For `from_status = 'Late_3'`: `to_current_pct` (cure), `to_default_pct` (roll to default)

Each row still sums to 100%. The `states='5state'` mode (default) combines Late sub-states for Tab 1 display.

**Expected validation patterns** (7-state):
- Delinquent can only go to Current or Late_1 (nothing else)
- Late_1 can only go to Current or Late_2
- Late_2 can only go to Current or Late_3
- Late_3 can only go to Current or Default
- Current → Late_1/Late_2/Late_3/Default = 0% (cannot skip states)
- Cure rates should decrease with depth: Delinquent cure > Late_1 cure > Late_2 cure > Late_3 cure (general expectation, verify empirically)
```

---

## Update 7: CLAUDE.md — Add New Function Specs for State-Transition Engine

Find the `## NEW Functions to Build` section. Add these function specs AFTER the existing `project_cashflows` spec:

```markdown
#### `build_pool_state(df, include_statuses=None) → dict`

Construct the initial pool state for the state-transition model from the loan DataFrame.

**Parameters**:
- `df`: Loan DataFrame with `reconstruct_loan_timeline()` and `calc_amort()` already applied
- `include_statuses`: list of loan statuses to include. Options:
  - `['Current']` — only Current loans (default)
  - `['Current', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']` — all active loans

**Logic**:
```python
pool_df = df[df['loan_status'].isin(include_statuses)].copy()

# Map LC statuses to 7 model states
# For Late (31-120 days) loans, determine which sub-state based on months since entering Late
snapshot = pd.Timestamp('2019-03-01')

status_map = {
    'Current': 'Current',
    'In Grace Period': 'Delinquent (0-30)',
    'Late (16-30 days)': 'Delinquent (0-30)',
}

pool_df['model_state'] = pool_df['loan_status'].map(status_map)

# For Late (31-120) loans: determine sub-state from months since late_31_120_month
late_mask = pool_df['loan_status'] == 'Late (31-120 days)'
if late_mask.any():
    months_in_late = (
        (snapshot - pool_df.loc[late_mask, 'late_31_120_month']).dt.days / 30.44
    ).round().astype(int)
    # Month 0 in late = Late_1, month 1 = Late_2, month 2+ = Late_3
    pool_df.loc[late_mask & (months_in_late == 0), 'model_state'] = 'Late_1'
    pool_df.loc[late_mask & (months_in_late == 1), 'model_state'] = 'Late_2'
    pool_df.loc[late_mask & (months_in_late >= 2), 'model_state'] = 'Late_3'

# Group by model_state and age_bucket_label
pool_state = {}
all_states = ['Current', 'Delinquent (0-30)', 'Late_1', 'Late_2', 'Late_3']
for state in all_states:
    state_df = pool_df[pool_df['model_state'] == state]
    if len(state_df) == 0:
        pool_state[state] = pd.DataFrame(
            columns=['age_bucket', 'upb', 'loan_count', 'wac', 'wam']
        )
        continue

    grouped = state_df.groupby('age_bucket_label').apply(lambda g: pd.Series({
        'age_bucket': g['age_bucket_label'].iloc[0],
        'upb': g['out_prncp'].sum(),
        'loan_count': len(g),
        'wac': np.average(g['int_rate'], weights=g['out_prncp']) if g['out_prncp'].sum() > 0 else 0,
        'wam': np.average(g['updated_remaining_term'], weights=g['out_prncp']) if g['out_prncp'].sum() > 0 else 0,
    })).reset_index(drop=True)

    pool_state[state] = grouped

return pool_state
```

**Returns**: dict keyed by 7 model states → DataFrame with `age_bucket`, `upb`, `loan_count`, `wac`, `wam` per bucket. Default and Fully Paid start empty (cumulative states).

---

#### `project_cashflows_transition(pool_state, age_transition_probs, loss_severity, recovery_rate, num_months) → pd.DataFrame`

State-transition cash flow engine. Tracks pool balance across 7 states with age-specific transition probabilities each month.

**Parameters**:
- `pool_state`: dict from `build_pool_state()` — initial UPB in each (state, age_bucket)
- `age_transition_probs`: DataFrame from `compute_age_transition_probabilities(states='7state')` — empirical transition rates by age bucket and from_status
- `loss_severity`: float — FIXED across all scenarios
- `recovery_rate`: float — 1 - loss_severity
- `num_months`: int — projection horizon (typically max WAM in pool)

**Logic — monthly loop**:

```python
results = []
cumulative_default_upb = 0
cumulative_paid_upb = 0

for t in range(1, num_months + 1):
    month_cf = {
        'month': t,
        'interest': 0, 'scheduled_principal': 0, 'prepayments': 0,
        'new_defaults': 0, 'losses': 0, 'recoveries': 0,
    }

    new_pool_state = {state: [] for state in pool_state}

    for age_bucket_label in all_age_buckets:
        # --- Get UPB in each state for this age bucket ---
        current_upb = get_bucket_upb(pool_state, 'Current', age_bucket_label)
        delinq_upb  = get_bucket_upb(pool_state, 'Delinquent (0-30)', age_bucket_label)
        late1_upb   = get_bucket_upb(pool_state, 'Late_1', age_bucket_label)
        late2_upb   = get_bucket_upb(pool_state, 'Late_2', age_bucket_label)
        late3_upb   = get_bucket_upb(pool_state, 'Late_3', age_bucket_label)
        wac = get_bucket_wac(...)
        wam = get_bucket_wam(...)

        # --- Look up transition probabilities for this age bucket ---
        probs = lookup_probs(age_transition_probs, age_bucket_label)

        # Current transitions
        curr_to_curr    = probs[('Current', 'to_current_pct')]
        curr_to_delinq  = probs[('Current', 'to_delinquent_0_30_pct')]
        curr_to_paid    = probs[('Current', 'to_fully_paid_pct')]

        # Delinquent transitions
        delinq_to_curr  = probs[('Delinquent (0-30)', 'to_current_pct')]
        delinq_to_late1 = probs[('Delinquent (0-30)', 'to_late_1_pct')]

        # Late_1 transitions
        late1_to_curr   = probs[('Late_1', 'to_current_pct')]
        late1_to_late2  = probs[('Late_1', 'to_late_2_pct')]

        # Late_2 transitions
        late2_to_curr   = probs[('Late_2', 'to_current_pct')]
        late2_to_late3  = probs[('Late_2', 'to_late_3_pct')]

        # Late_3 transitions
        late3_to_curr   = probs[('Late_3', 'to_current_pct')]
        late3_to_default= probs[('Late_3', 'to_default_pct')]

        # === Cash flows (Current loans only) ===
        monthly_rate = wac / 12
        interest = current_upb * monthly_rate
        sched_principal = calc_scheduled_principal(current_upb, wac, wam)
        prepay_amount = current_upb * curr_to_paid

        month_cf['interest'] += interest
        month_cf['scheduled_principal'] += sched_principal
        month_cf['prepayments'] += prepay_amount

        # === UPB transitions ===
        # From Current
        new_current_from_current = (current_upb * curr_to_curr) - sched_principal - prepay_amount
        new_delinq_from_current  = current_upb * curr_to_delinq

        # From Delinquent
        new_current_from_delinq_cure = delinq_upb * delinq_to_curr
        new_late1_from_delinq        = delinq_upb * delinq_to_late1

        # From Late_1
        new_current_from_late1_cure  = late1_upb * late1_to_curr
        new_late2_from_late1         = late1_upb * late1_to_late2

        # From Late_2
        new_current_from_late2_cure  = late2_upb * late2_to_curr
        new_late3_from_late2         = late2_upb * late2_to_late3

        # From Late_3
        new_current_from_late3_cure  = late3_upb * late3_to_curr
        new_default_from_late3       = late3_upb * late3_to_default

        # === Losses ===
        month_cf['new_defaults'] += new_default_from_late3
        month_cf['losses'] += new_default_from_late3 * loss_severity
        month_cf['recoveries'] += new_default_from_late3 * recovery_rate

        # === Aggregate new state balances for this age bucket ===
        new_current = max(new_current_from_current + new_current_from_delinq_cure
                        + new_current_from_late1_cure + new_current_from_late2_cure
                        + new_current_from_late3_cure, 0)
        new_delinq = new_delinq_from_current
        new_late1 = new_late1_from_delinq
        new_late2 = new_late2_from_late1
        new_late3 = new_late3_from_late2  # Late_3 comes from Late_2, NOT Late_3 staying

        # Update new_pool_state for this bucket (advance age when crossing boundary)

    # === Cumulative tracking ===
    cumulative_default_upb += month_cf['new_defaults']
    cumulative_paid_upb += month_cf['scheduled_principal'] + month_cf['prepayments']

    month_cf['current_upb'] = sum all Current buckets
    month_cf['delinquent_upb'] = sum all Delinquent buckets
    month_cf['late_1_upb'] = sum all Late_1 buckets
    month_cf['late_2_upb'] = sum all Late_2 buckets
    month_cf['late_3_upb'] = sum all Late_3 buckets
    month_cf['default_upb'] = cumulative_default_upb
    month_cf['fully_paid_upb'] = cumulative_paid_upb
    month_cf['total_cf'] = interest + sched_principal + prepay_amount + recoveries - losses
    month_cf['net_cf'] = month_cf['total_cf']

    results.append(month_cf)
    pool_state = new_pool_state

return pd.DataFrame(results)
```

**Returns**: DataFrame with one row per month, columns:
- `month`: projection month (1, 2, 3, ...)
- `interest`, `scheduled_principal`, `prepayments`: cash from Current loans only
- `new_defaults`: UPB entering Default this month (from Late_3 only)
- `losses`, `recoveries`: from new defaults × loss_severity / recovery_rate
- `current_upb`, `delinquent_upb`, `late_1_upb`, `late_2_upb`, `late_3_upb`: end-of-month active state balances
- `default_upb`, `fully_paid_upb`: cumulative terminal state balances
- `total_cf`, `net_cf`: for IRR calculation (compatible with `calculate_irr()` and `solve_price()`)

---

#### `build_scenarios_transition(base_probs, stress_pct, upside_pct) → dict`

Generate stressed/upside transition probability tables for the state-transition model.

**Parameters**:
- `base_probs`: DataFrame from `compute_age_transition_probabilities(states='7state')`
- `stress_pct`: float (e.g., 0.15 for 15% stress)
- `upside_pct`: float (e.g., 0.15 for 15% upside)

**Stress logic** — increase delinquency entry, decrease cure rates:
- `Current → Delinquent (0-30)` × `(1 + stress_pct)` — more loans go delinquent
- `Current → Fully Paid` × `(1 - stress_pct)` — fewer prepayments
- `Delinquent → Current` (cure) × `(1 - stress_pct)` — fewer cures
- `Late_1 → Current` (cure) × `(1 - stress_pct)` — fewer cures
- `Late_2 → Current` (cure) × `(1 - stress_pct)` — fewer cures
- `Late_3 → Current` (cure) × `(1 - stress_pct)` — fewer cures
- `Late_3 → Default` is **NOT stressed directly** — it increases mechanically because fewer loans cure from Late_3, so more roll to Default after row re-normalization

**Upside logic** — opposite direction:
- `Current → Delinquent (0-30)` × `(1 - upside_pct)`
- `Current → Fully Paid` × `(1 + upside_pct)`
- All cure rates × `(1 + upside_pct)`

**After applying multipliers**: Clamp all probabilities to [0, 1], then re-normalize each row to sum to 100% by adjusting the "stay/roll" probability. For example, if Delinquent cure rate increases, Delinquent→Late_1 decreases to compensate.

**Returns**: `{'base': base_probs, 'stress': stressed_probs, 'upside': upside_probs}`
```

---

## Update 8: CLAUDE.md — Update Dashboard Specification

Find the `### Sidebar Controls` section. Add these new controls:

```markdown
- **Model Type** (`st.radio`): "Simple (Flat CDR/CPR)" or "Advanced (State-Transition)". Default: Simple.
- **Pool Composition** (`st.radio`): "Current Loans Only" or "All Active Loans". Only visible when Model Type = Advanced. When Simple, pool is always Current loans with March 2019 last payment only.
```

Find the Tab 2 and Tab 3 sections. Add this note to both:

```markdown
**Model-dependent behavior**:
- When **Simple** is selected: use `project_cashflows()` (existing flat CDR/CPR engine). Cash flow chart shows smooth exponential decay.
- When **Advanced** is selected: use `project_cashflows_transition()`. Cash flow chart will show zero defaults in early months (~5 months for an all-Current pool), then ramping defaults as loans flow through the pipeline. Display a stacked area chart of UPB by state over time (Current, Delinquent, Late_1, Late_2, Late_3, Default, Fully Paid).
- Both models feed into the same `calculate_irr()` and `solve_price()` functions via the `net_cf` column.
- Tab 3 scenario comparison works with both models — Simple uses CDR/CPR shifts, Advanced uses transition probability shifts via `build_scenarios_transition()`.
```

---

## Update 9: CLAUDE.md — Update Testing Requirements

Find the testing section. Add this block:

```markdown
**cashflow_engine (state-transition, 7-state)**:

- `build_pool_state`:
  - All-Current pool: verify Delinquent, Late_1, Late_2, Late_3 DataFrames are all empty
  - Pool with Late (31-120 days) loans: verify they are correctly assigned to Late_1/Late_2/Late_3 based on months since entering Late
  - Verify Grace Period and Late (16-30 days) both map to Delinquent (0-30)
  - Verify UPB sums match input data
  - Verify WAC and WAM are correctly balance-weighted

- `project_cashflows_transition`:
  - All-Current pool: verify months 1-4 have zero defaults and zero losses (minimum 5-month pipeline)
  - Verify first defaults appear at month ~5
  - Verify delinquent_upb > 0 starting month 1 (some Current loans go delinquent)
  - Verify late_1_upb > 0 starting month 2
  - Verify late_2_upb > 0 starting month 3
  - Verify late_3_upb > 0 starting month 4
  - Verify new_defaults > 0 starting month ~5
  - **Conservation test**: each month, `current_upb + delinquent_upb + late_1_upb + late_2_upb + late_3_upb + default_upb + fully_paid_upb ≈ initial_pool_upb` (within tolerance for losses)
  - Verify interest is only generated by Current UPB
  - Verify prepayments come only from Current → Fully Paid
  - Zero-default case: set all Current→Delinquent rates to 0 → no loans ever go delinquent → losses = 0 for all months
  - Verify output `net_cf` column works with `calculate_irr()`

- `build_scenarios_transition`:
  - Verify base probabilities unchanged
  - Verify stress increases Current→Delinquent, decreases ALL cure rates (Delinquent, Late_1, Late_2, Late_3)
  - Verify upside decreases Current→Delinquent, increases ALL cure rates
  - Verify Late_3→Default is NOT directly stressed (changes only because cure rate changed and row re-normalized)
  - Verify all rows sum to 100% after shifts
  - Verify loss severity identical across all three scenarios
```

---

## Update 10: .claude/agents/financial-validator.md — Add 7-State Tests

Find the cash flow validation section. Add:

```markdown
### Step 4b: State-Transition Cash Flow Checks (7-State)

1. **Pipeline timing test**: All-Current pool at $100K, 10% WAC, 36-month WAM.
   - Month 1: new_defaults = $0, delinquent_upb > $0
   - Month 2: new_defaults = $0, late_1_upb > $0
   - Month 3: new_defaults = $0, late_2_upb > $0
   - Month 4: new_defaults = $0, late_3_upb > $0
   - Month 5: new_defaults > $0 (first defaults arrive)

2. **Conservation test**: For each month:
   `current + delinquent + late_1 + late_2 + late_3 + cumulative_default + cumulative_paid ≈ initial_upb`

3. **Cure flow test**: Set Late_1→Current cure rate to 50%. Verify that ~50% of Late_1 UPB flows back to Current the next month and generates cash flows again.

4. **No-skip test**: Verify these are always zero:
   - Current → Late_1, Late_2, Late_3, Default (must go through Delinquent first)
   - Delinquent → Late_2, Late_3, Default (must go through Late_1 first)
   - Late_1 → Late_3, Default (must go through Late_2 first)
   - Late_2 → Default (must go through Late_3 first)

5. **Cash flow source test**:
   - interest = current_upb_start × WAC/12 (NOT delinquent or late UPB)
   - prepayments from Current only
   - defaults from Late_3 only

6. **Scenario ordering**: Stress IRR < Base IRR < Upside IRR
```

---

## Update 11: .claude/agents/dashboard-reviewer.md — Add Model Toggle Tests

Find Tab 2 checks. Add:

```markdown
7. **Model Type toggle**:
   - Switch to Advanced → verify cash flow chart changes (no instant defaults, shows pipeline ramp-up)
   - Verify a UPB-by-state stacked area chart appears (Current/Delinquent/Late_1/Late_2/Late_3/Default/Fully Paid)
   - Switch back to Simple → verify chart reverts
   - Verify IRR values differ between Simple and Advanced

8. **Pool Composition** (Advanced mode only):
   - Select "Current Loans Only" → verify pool UPB matches Current loans total
   - Select "All Active Loans" → verify pool UPB increases (includes delinquent/late)
   - Verify "All Active" shows non-zero Delinquent and/or Late UPB at month 0
   - Verify Pool Composition control is hidden when Simple is selected
```

Find Tab 3 checks. Add:

```markdown
5. **Scenario comparison with Advanced model**:
   - Toggle to Advanced → run scenarios
   - Verify defaults ramp up slower in Upside (more curing) and faster in Stress (less curing)
   - Verify loss severity identical across scenarios
   - Verify IRR ordering: Stress < Base < Upside
```

---

## Update 12: CLAUDE.md — Update Workflow Phases

Find Phase 3. Replace with:

```markdown
3. **Phase 3**: Build cash flow engines in `src/cashflow_engine.py`:
   - `compute_pool_assumptions()` — CDR (conditional, 12-month trailing), CPR (single-month from calc_amort), loss severity
   - `compute_pool_characteristics()` — WAC, WAM, WALA, total UPB (accepts any pool composition)
   - `project_cashflows()` — existing flat CDR/CPR model (UNCHANGED)
   - `build_pool_state()` — construct 7-state initial vector from loan data
   - `project_cashflows_transition()` — NEW 7-state transition model with age-specific probabilities
   - `calculate_irr()` — works with both models (uses net_cf column)
   - `solve_price()` — works with both models
```

Find Phase 4. Replace with:

```markdown
4. **Phase 4**: Build scenario analysis in `src/scenario_analysis.py`:
   - `build_scenarios()` — multiplicative CDR/CPR shifts for flat model (UNCHANGED)
   - `build_scenarios_transition()` — NEW transition probability shifts for 7-state model
   - `compare_scenarios()` — works with both models
```

---

That's it. Twelve edits across three files. The existing flat model is completely untouched. This adds a parallel 7-state engine with proper delinquency pipeline timing, Late sub-state tracking, curing at every stage, and a minimum 5-month path from Current to Default.
