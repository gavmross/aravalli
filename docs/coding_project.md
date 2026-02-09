════════════════════════════════════════════════════════════════
ARAVALLI CAPITAL - ASSOCIATE ASSESSMENT
Time: 1-2 weeks
Tools: Claude Code, Codex, any AI (encouraged)
════════════════════════════════════════════════════════════════

BUILD: A Loan Portfolio Investment Analysis Tool

You're evaluating a portfolio of consumer loans for purchase.
Build a tool that helps an investor understand the portfolio
and model returns under different scenarios.

────────────────────────────────────────────────────────────────
DATA SOURCE
────────────────────────────────────────────────────────────────

Use Lending Club's public loan data:
https://www.kaggle.com/datasets/wordsforthewise/lending-club

This contains ~2.2M loans with origination and performance data.
 
────────────────────────────────────────────────────────────────
REQUIREMENTS
────────────────────────────────────────────────────────────────

PART 1: PORTFOLIO ANALYTICS

Ingest and analyze the loan data. Provide:

- Pool stratifications (grade, term, purpose, geography, vintage)
- Credit metrics (WAC, WAM, WALA, avg FICO, avg DTI)
- Performance metrics by vintage:
  - Cumulative default rate (CDR)
  - Cumulative prepayment rate (CPR)
  - Loss severity / recovery rate
- Delinquency transition matrix (roll rates) 
- Key insights: what stands out about this portfolio?

────────────────────────────────────────────────────────────────

PART 2: CASH FLOW PROJECTION & IRR

Build a cash flow model for a subset of the portfolio.

Given:
- A purchase price (as % of UPB)
- Assumptions for future defaults and prepayments

Calculate:
- Monthly projected cash flows (principal, interest, losses)
- IRR at the given price
- Price to achieve a target IRR (solve the inverse)

Example usage:
  `project --vintage 2018Q1 --price 0.95 --cdr 0.10 --cpr 0.15`
  → Returns: projected cash flows, IRR

  `solve-price --vintage 2018Q1 --target-irr 0.12 --cdr 0.10 --cpr 0.15`
  → Returns: price that achieves 12% IRR

────────────────────────────────────────────────────────────────

PART 3: SCENARIO ANALYSIS

Allow the user to run multiple scenarios and compare outcomes.

- Base case (your best estimate from historical data)
- Stress case (elevated defaults, slower prepays)
- Upside case (lower defaults, faster prepays)

Output should show IRR sensitivity across scenarios.

Example:
  `scenarios --vintage 2018Q1 --price 0.95`
 
  | Scenario | CDR   | CPR   | IRR    |
  |----------|-------|-------|--------|
  | Base     | 8%    | 12%   | 10.2%  |
  | Stress   | 15%   | 8%    | 4.1%   |
  | Upside   | 5%    | 18%   | 13.7%  |

────────────────────────────────────────────────────────────────
DELIVERABLES
────────────────────────────────────────────────────────────────

A. Working code repository
   - Runnable with clear instructions
   - Modular and readable

B. Writeup (2-3 pages):
   - Data issues you found and how you handled them
   - Your metric and cash flow definitions
   - Key insights about the portfolio
   - What scenarios would you recommend for stress testing and why?

C. Presentation (30 min + 15 min Q&A)
   - Demo the tool
   - Present your portfolio insights
   - Walk through the cash flow logic
   - Defend your scenario assumptions

────────────────────────────────────────────────────────────────
WHAT WE'RE LOOKING FOR
────────────────────────────────────────────────────────────────

- Work that's correct, not just work that runs
- Code that someone else could pick up and extend
- Judgment about when to trust AI output and when to verify
- Clear communication about your choices and findings

We'll ask you to explain, modify, and defend your work live.
────────────────────────────────────────────────────────────────