Compare documentation against actual code implementations and flag any discrepancies.

## Steps

### 1. Formula Audit: `docs/calculations.md` vs Source Code

For each formula documented in `calculations.md`, locate the corresponding code and verify they match:

- Read `docs/calculations.md`
- For each formula listed, find the implementation in `src/cashflow_engine.py`, `src/scenario_analysis.py`, `src/portfolio_analytics.py`, or `src/amortization.py`
- Use sequential-thinking to confirm the documented formula matches the code
- Pay special attention to: CPR/SMM conversions, MDR conversion, loss severity (capped recoveries), IRR annualization, cash flow loop order of operations

### 2. Cleaning Audit: `docs/data_cleaning.md` vs `scripts/export_to_sqlite.py`

- Read `docs/data_cleaning.md`
- Read `scripts/export_to_sqlite.py`
- Verify every documented cleaning step has a corresponding code implementation
- Verify no cleaning steps in the code are undocumented
- Check that row counts and statistics mentioned in the doc are still accurate

### 3. User Guide Audit: `docs/user_guide.md` vs Reality

- Verify the setup instructions actually work (correct venv name, correct pip install command)
- Verify the file structure in the user guide matches the actual repo
- Verify the module descriptions match what the modules actually contain
- Verify the "how to run" instructions are correct

### 4. CLAUDE.md Cross-Check

- Verify the Domain Terminology & Formulas section in `CLAUDE.md` matches `docs/calculations.md`
- Verify the File Structure section matches the actual directory tree
- Verify the Critical Rules still apply (no contradictions with code behavior)

### 5. Cross-Reference Consistency

Use grep to find all occurrences of key terms across all files and verify consistency:

```bash
grep -rn "loss_severity" docs/ src/ CLAUDE.md
grep -rn "CDR" docs/ src/ CLAUDE.md
grep -rn "CPR" docs/ src/ CLAUDE.md
grep -rn "SMM" docs/ src/ CLAUDE.md
grep -rn "recovery_rate" docs/ src/ CLAUDE.md
```

Verify that the same formula/definition is used everywhere it appears.

## Output Format

```
DOCUMENTATION SYNC REPORT
===========================

calculations.md:
  Formulas checked: N
  ✓ Matching: [list]
  ✗ Drifted: [list with details of mismatch]

data_cleaning.md:
  Steps checked: N
  ✓ Matching: [list]
  ✗ Drifted: [list with details]

user_guide.md:
  Sections checked: N
  ✓ Accurate: [list]
  ✗ Outdated: [list with details]

CLAUDE.md:
  ✓ Consistent / ✗ Drifted (details)

Cross-references:
  ✓ All consistent / ✗ Mismatches found (details)
```

## Important

- The CODE is the source of truth, not the documentation. If they disagree, the documentation is wrong.
- Exception: if the code appears to have a bug, flag both the doc AND the code for review.
- Never silently approve. List every formula checked, even if all match.
