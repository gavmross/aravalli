Validate all financial formulas in the codebase against the lending-math skill reference.

## Steps

1. Read `.claude/skills/lending-math/SKILL.md` for the authoritative formula definitions
2. Use sequential-thinking MCP to verify each formula below. For EACH one:
   a. State the formula as implemented in code (quote the exact line)
   b. State the correct formula from the skill file
   c. Confirm they match OR flag the specific discrepancy

### Formulas to Check

**In `src/cashflow_engine.py`:**
- [ ] MDR = 1 - (1 - CDR)^(1/12)
- [ ] SMM = 1 - (1 - CPR)^(1/12)
- [ ] monthly_rate = WAC / 12
- [ ] defaults = beginning_balance × MDR
- [ ] loss = defaults × loss_severity
- [ ] recovery = defaults × (1 - loss_severity)
- [ ] interest = performing_balance × monthly_rate (NOT beginning_balance)
- [ ] scheduled_principal capped at performing_balance
- [ ] prepayments = (performing_balance - scheduled_principal) × SMM
- [ ] ending_balance floored at 0
- [ ] total_cashflow = interest + total_principal + recovery
- [ ] IRR annualization: (1 + monthly_irr)^12 - 1 (NOT monthly × 12)
- [ ] Month 0 cash flow = -(purchase_price × total_upb)
- [ ] Price solver uses brentq with round-trip verification

**In `src/scenario_analysis.py`:**
- [ ] Stress CDR = base_cdr × stress_multiplier
- [ ] Stress CPR = base_cpr × stress_multiplier
- [ ] Loss severity is FIXED across all scenarios (same value for base/stress/upside)

**In `src/portfolio_analytics.py` and `src/amortization.py`:**
- [ ] CPR = 1 - (1 - SMM)^12
- [ ] Recovery rate capped: min(recoveries, exposure)
- [ ] loss_severity + recovery_rate == 1.0

3. Run `pytest tests/test_cashflow_engine.py -v` to confirm numerical results
4. Run these specific sanity checks in Python:
   ```python
   # Test 1: Simple amortization
   # $10,000 at 10% for 36 months → PMT = $322.67
   
   # Test 2: IRR round-trip
   # cash flows [-100, 60, 60] → IRR ≈ 13.07%
   
   # Test 3: Price solver round-trip
   # Solve for price at 12% target, then verify IRR at that price = 12%
   ```

## Output Format

```
FORMULA VALIDATION REPORT
==========================
✓ MDR conversion — MATCH (line 45 of cashflow_engine.py)
✗ Interest calculation — MISMATCH
  Code:     interest = beginning_balance * monthly_rate
  Expected: interest = performing_balance * monthly_rate
  Fix:      Change beginning_balance to (beginning_balance - defaults)

Tests: 12/12 passed
Sanity checks: 3/3 passed
```
