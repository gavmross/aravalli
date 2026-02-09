# Lending Club Loan Portfolio Investment Analysis Tool

An interactive Streamlit dashboard for analyzing a portfolio of ~2.2M Lending Club consumer loans. The tool lets an investor filter by pool strata, view credit and performance metrics, project cash flows under purchase price assumptions, and compare base/stress/upside scenarios with IRR calculations.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .env

# 2. Activate
source .env/bin/activate    # macOS/Linux
# .env\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the database (requires data/accepted_2007_to_2018Q4.csv)
python scripts/export_to_sqlite.py

# 5. Launch the dashboard
streamlit run app.py
```

## Project Structure

```
├── app.py                          # Streamlit dashboard (sidebar + 3 tabs)
├── src/
│   ├── amortization.py             # Loan amortization calculations
│   ├── portfolio_analytics.py      # Credit metrics, performance, transitions
│   ├── cashflow_engine.py          # Cash flow projections, IRR, price solver
│   └── scenario_analysis.py        # Base/stress/upside scenario builder
├── tests/                          # pytest test suite (101 tests)
├── data/
│   ├── accepted_2007_to_2018Q4.csv # Raw Lending Club data (~2.2M loans)
│   └── loans.db                    # Cleaned SQLite database
├── scripts/
│   ├── export_to_sqlite.py         # Data cleaning pipeline
│   └── analysis.ipynb              # Original analysis notebook
└── docs/
    ├── user_guide.md               # Setup, usage, tab descriptions
    ├── calculations.md             # Every formula with worked examples
    └── data_cleaning.md            # Cleaning steps with rationale
```

## Dashboard Tabs

**Tab 1 — Portfolio Analytics**: Pool stratifications by grade/term/purpose/state/vintage, credit metrics (WAC, WAM, FICO, DTI), performance metrics by vintage (CDR, CPR, loss severity), and delinquency transition matrix.

**Tab 2 — Cash Flow Projection & IRR**: Monthly projected cash flows with defaults, prepayments, and recoveries. Displays IRR at the selected purchase price and a price solver for target IRR.

**Tab 3 — Scenario Analysis**: Base/stress/upside comparison with adjustable multiplicative shifts. Shows IRR sensitivity, balance runoff, and aggregate metrics across scenarios.

## Documentation

| Document | Contents |
|----------|----------|
| [User Guide](docs/user_guide.md) | Setup instructions, how to run, detailed tab descriptions with screenshots |
| [Calculations](docs/calculations.md) | Every financial formula with variable definitions and worked examples |
| [Data Cleaning](docs/data_cleaning.md) | All 20 cleaning steps with rationale, row counts, and statistics |

## Testing

```bash
pytest tests/ -v
```

101 tests covering amortization, pool assumptions, cash flow projections, IRR calculations, price solver, and scenario analysis.

## Tech Stack

- **Python 3.11+** with pandas, numpy, scipy, numpy-financial
- **Streamlit** for the interactive dashboard
- **Plotly** for interactive charts
- **SQLite** for data storage
- **pytest** for testing
