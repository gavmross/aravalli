# Refactor: Remove "Current Only" Option — Use All Active Loans Everywhere

## Goal

Remove the dual-population option (Current Loans Only vs All Active Loans) from the dashboard. **The only investable pool is now ALL active loans** — defined as `loan_status` in `['Current', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']`. There should be no toggle, no radio button, and no code path that filters to Current-only for Tabs 2 & 3.

The state-transition model already handles delinquent loans correctly (mapping them to Delinquent (0-30), Late_1, Late_2, Late_3), so there is no analytical reason to exclude them.

---

## Scope of Changes

### Files to Modify

| File | What Changes |
|------|-------------|
| `app.py` | Remove Pool Composition radio; change `df_current_march` filter to include all active statuses; rename variable; update warning messages and display labels |
| `src/cashflow_engine.py` | Rename `df_current` param → `df_active` in `compute_pool_assumptions()` and `compute_pool_characteristics()`; update docstrings and comments; update CPR comment from "pool_cpr_current" to "pool_cpr_active" |
| `src/scenario_analysis.py` | Rename `df_current` param → `df_active` in `compute_base_assumptions()`; update docstrings |
| `src/portfolio_analytics.py` | In `calculate_performance_metrics()`: remove the separate `pool_cpr_current` calculation, keep only `pool_cpr_active` (renamed to `pool_cpr`); remove `pct_prepaid_current`, keep only a single `pct_prepaid` computed from all active loans; update column order and docstring. **NOTE**: This function has a "DO NOT MODIFY LOGIC" header — the logic is not changing, only the redundant Current-only CPR block is being removed and columns renamed. The active-loans CPR block stays as-is. |
| `CLAUDE.md` | Remove Pool Composition radio from sidebar controls; update Data Flow section; update Critical Rules; update function signatures in Function Reference |
| `docs/user_guide.md` | Remove Pool Composition docs; update population descriptions for Tabs 2 & 3; update performance metrics table; update troubleshooting |
| `docs/calculations.md` | Update CPR population description from "Current loans only" to "all active loans"; update pool characteristics population |
| `docs/data_cleaning.md` | Update "Cash Flow Projection Population" section at the bottom — should reference all active loans with March 2019 last payment, not just Current |
| `.claude/skills/data-schema/SKILL.md` | Update "Data Population Splits" section — CPR line should say all active loans |
| `.claude/skills/streamlit-patterns/SKILL.md` | Update `df_current_march` filter pattern and Tab 2 layout docs |
| `tests/test_cashflow_engine.py` | Update any tests that pass `df_current` → `df_active`; update parameter names |
| `tests/test_scenario_analysis.py` | Update parameter names |
| `tests/test_portfolio_analytics.py` | Update assertions that reference `pool_cpr_current` or `pct_prepaid_current` → `pool_cpr` / `pct_prepaid` |

---

## Detailed Changes Per File

### 1. `app.py`

**Remove the Pool Composition radio button from the sidebar.** Delete the `st.radio` for "Pool Composition" and all conditional logic around it (`if pool_composition == "Current Loans Only"` / `else`).

**Change the investable pool filter.** Currently there is something like:
```python
df_current_march = df_filtered[
    (df_filtered['loan_status'] == 'Current') &
    (df_filtered['last_pymnt_d'] == '2019-03-01')
]
```
Replace with:
```python
active_statuses = ['Current', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
df_active_march = df_filtered[
    (df_filtered['loan_status'].isin(active_statuses)) &
    (df_filtered['last_pymnt_d'] == '2019-03-01')
]
```

**Rename all references** from `df_current_march` → `df_active_march` throughout the file (used in Tabs 2 and 3).

**Update warning messages** from "No Current loans with March 2019 payment date" → "No active loans with March 2019 payment date".

**Update the sidebar summary** if it mentions "Current" loans — it should say "Active" loans.

**In Tab 1 (Performance Metrics display):**
- Remove the line that drops `pool_cpr_active` column: `pm_display = pm_display.drop(columns=["pool_cpr_active"])`
- Instead, drop `pool_cpr_current` if it still exists (or just keep the single `pool_cpr` column)
- Rename the display column from `"CPR (Current Only)"` → `"CPR (Active Loans)"`
- Rename `"pct_prepaid_current"` → `"pct_prepaid"` or `"UPB Prepaid of OUPB"` (the display name)

**In Tab 2**, update the call:
```python
pool_assumptions = compute_pool_assumptions(df_filtered, df_active_march)
pool_chars = compute_pool_characteristics(df_active_march)
```
And for `build_pool_state`, always pass all active statuses:
```python
pool_state = build_pool_state(
    df_active_march,
    include_statuses=['Current', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
)
```

**In Tab 3**, same changes — use `df_active_march` everywhere.

### 2. `src/cashflow_engine.py`

**`compute_pool_assumptions(df_all, df_current)`** → rename second param to `df_active`:
- Update function signature: `def compute_pool_assumptions(df_all: pd.DataFrame, df_active: pd.DataFrame) -> dict:`
- Update docstring: change "Current loans only with last_pymnt_d == 2019-03-01" → "Active loans (Current + In Grace + Late) with last_pymnt_d == 2019-03-01"
- Update CPR comment: change `# Same logic as pool_cpr_current` → `# Same logic as pool_cpr in calculate_performance_metrics`
- Rename internal variable `paid_current` → `paid_active` and change `df_current` → `df_active`

**`compute_pool_characteristics(df_current)`** → rename param to `df_active`:
- Update function signature: `def compute_pool_characteristics(df_active: pd.DataFrame) -> dict:`
- Update docstring: "Extract pool-level aggregates from active loans (March 2019 last payment)."
- Rename `df_current` → `df_active` throughout the function body

**`build_pool_state()`** → update the default `include_statuses`:
```python
if include_statuses is None:
    include_statuses = ['Current', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
```
This makes all active loans the default instead of Current-only.

### 3. `src/scenario_analysis.py`

**`compute_base_assumptions(df_all, df_current)`** → rename to `df_active`:
- Update signature and docstring
- Update the internal call: `compute_pool_assumptions(df_all, df_active)`

### 4. `src/portfolio_analytics.py`

In **`calculate_performance_metrics()`**:

**Remove the entire `pool_cpr_current` calculation block** (the "CPR - CURRENT LOANS ONLY" section). Keep only the `pool_cpr_active` block and rename it to `pool_cpr`.

**Remove the `pct_prepaid_current` calculation** that uses Current loans only. Replace it with a `pct_prepaid` that uses all active loans (the same logic, just on `active_loans` instead of `current_loans`). OR — if there is already a `pct_prepaid_active`, just rename it to `pct_prepaid` and remove the current-only version.

**Update the column order list** at the bottom of the function:
```python
col_order = [
    'vintage', 'orig_loan_count', 'orig_upb_mm',
    'pct_active', 'pct_current', 'pct_fully_paid', 'pct_charged_off',
    'pct_defaulted_count', 'pct_defaulted_upb',
    'pool_cpr', 'pct_prepaid',
    'loss_severity', 'recovery_rate'
]
```

**Update the docstring** to remove references to `pool_cpr_current` and `pct_prepaid_current`.

**Update the verbose print formatting** if it references the old column names.

### 5. `CLAUDE.md`

**Sidebar Controls section** (line ~247-248): Remove the Pool Composition radio bullet entirely. It should just list Strata Type and Strata Value.

**Data Flow section** (lines ~254-259): Change step 3 from:
> Tab 1 uses ALL loans; Tabs 2 & 3 use ONLY Current loans with `last_pymnt_d == 2019-03-01` (or all active loans if Pool Composition is set to "All Active Loans")

To:
> Tab 1 uses ALL loans; Tabs 2 & 3 use all active loans (Current + In Grace + Late) with `last_pymnt_d == 2019-03-01`

Change step 4 from:
> CDR from ALL loans; CPR from Current March 2019 only; Loss severity from Charged Off subset

To:
> CDR from ALL loans; CPR from active March 2019 loans; Loss severity from Charged Off subset

**Critical Rules section** (line ~317): Change:
> Cash flows and scenarios operate on Current loans with March 2019 last payment date ONLY.

To:
> Cash flows and scenarios operate on all active loans (Current + In Grace + Late) with March 2019 last payment date.

Change:
> CPR from Current March 2019 only.

To:
> CPR from all active March 2019 loans.

**Function Reference section**: Update parameter names from `df_current` → `df_active` for:
- `compute_pool_assumptions(df_all, df_active)`
- `compute_pool_characteristics(df_active)`
- `compute_base_assumptions(df_all, df_active)`
- `build_pool_state()` default `include_statuses` description

**Performance metrics docstring** in the function reference: Remove `pool_cpr_current` and `pct_prepaid_current`, replace with `pool_cpr` and `pct_prepaid`.

### 6. `docs/user_guide.md`

**Sidebar Controls section**: Remove the Pool Composition radio documentation entirely.

**Performance Metrics Table**: Remove the `pool_cpr_current` row. Remove `pool_cpr_active` row. Replace with a single `pool_cpr` row described as "CPR for all active loans". Same for prepaid columns.

**Tab 2 population note** (around line 268): Change from:
> Only Current loans with March 2019 last payment date are used for projections (unless "All Active Loans" is selected via the Pool Composition control).

To:
> All active loans (Current + In Grace Period + Late) with March 2019 last payment date are used for projections.

**Tab 2 assumptions table**: Change CPR source from "From Current March 2019 loans" → "From active March 2019 loans (pool-level SMM annualized)".

**Pool characteristics table**: Change UPB source from "Current March 2019 loans" → "active March 2019 loans".

**Data Population Splits section** (if present): Update CPR line from "Current loans with last_pymnt_d == '2019-03-01' ONLY" → "Active loans (Current + In Grace + Late) with last_pymnt_d == '2019-03-01'"

**Troubleshooting section**: Update the "No Current loans" FAQ → "No active loans with March 2019 payment date".

### 7. `docs/calculations.md`

**CPR section** (around line 176): Change "Computed from Current loans with March 2019 last payment date only" → "Computed from all active loans (Current + In Grace + Late) with March 2019 last payment date".

**Pool characteristics section** (around line 220): Change "All computed from Current loans with March 2019 last payment date only" → "All computed from active loans with March 2019 last payment date".

### 8. `docs/data_cleaning.md`

**Cash Flow Projection Population section** at the bottom: Change "Current loans with `last_pymnt_d = 2019-03-01`" → "Active loans (Current + In Grace + Late) with `last_pymnt_d = 2019-03-01`". Update the count and UPB numbers to reflect all active loans, not just Current. (Run a query to get the actual numbers if the database is available.)

### 9. `.claude/skills/data-schema/SKILL.md`

**Data Population Splits section**: Update CPR line and Pool Characteristics line from "Current loans" → "Active loans". Update the SQL helper comment.

### 10. `.claude/skills/streamlit-patterns/SKILL.md`

**Tab 2 Key Patterns section**: Update the `df_current_march` filter code example:
```python
active_statuses = ['Current', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
df_active_march = df_filtered[
    (df_filtered['loan_status'].isin(active_statuses)) &
    (df_filtered['last_pymnt_d'] == '2019-03-01')
]
```

### 11. Tests

**`tests/test_cashflow_engine.py`**:
- Rename `df_current` → `df_active` in all test function arguments and fixture references
- Update any test that constructs a DataFrame with only `loan_status == 'Current'` for pool assumptions — it should still work, but verify test names/comments are updated
- If any test explicitly checks that delinquent loans are excluded, remove that assertion

**`tests/test_scenario_analysis.py`**:
- Rename `df_current` → `df_active` in `compute_base_assumptions` calls

**`tests/test_portfolio_analytics.py`**:
- Update assertions for `pool_cpr_current` → `pool_cpr`
- Update assertions for `pct_prepaid_current` → `pct_prepaid`
- Remove any assertions about `pool_cpr_active` as a separate column (it's now just `pool_cpr`)

---

## What NOT to Change

- **`calc_amort()`** — no changes needed, it operates on individual loans regardless of status
- **`calculate_credit_metrics()`** — already uses all active statuses for its active metrics section
- **`calculate_transition_matrix()`** — no changes needed
- **`reconstruct_loan_timeline()`** and related functions — no changes needed
- **`project_cashflows_transition()`** — no changes needed, it already processes whatever `build_pool_state` gives it
- **`project_cashflows()` / `solve_price()` / `build_scenarios()` / `compare_scenarios()`** — these are the flat CDR/CPR model functions that are already not used by the dashboard; leave them as-is
- **`export_to_sqlite.py`** — no changes needed to the data pipeline
- **`transition_viz.py`** — no changes needed

---

## Verification Steps

After making all changes:

1. **Run `pytest tests/ -v`** — all tests should pass
2. **Launch the dashboard** with `streamlit run app.py` and verify:
   - No Pool Composition radio button in sidebar
   - Tab 1 performance metrics table shows a single `CPR` column (not two)
   - Tab 2 loads and displays pool characteristics, cash flows, IRR for the full active pool
   - Tab 3 scenario comparison works with the full active pool
   - Warning messages say "active loans" not "Current loans"
3. **Spot-check numbers**: The total UPB in Tab 2 should be slightly higher than before (includes delinquent loan balances), and the pool should include loans in Grace/Late statuses in the initial state distribution
4. **Verify docs are consistent**: `docs/calculations.md`, `docs/user_guide.md`, `docs/data_cleaning.md`, and `CLAUDE.md` should all consistently say "active loans" instead of "Current loans" for the investable pool

---

## Summary

This is a simplification refactor. We are removing the complexity of two code paths (Current-only vs All Active) and standardizing on a single population: **all active loans**. The state-transition model already correctly handles delinquent loans, so there is no reason to exclude them from the investable pool.
