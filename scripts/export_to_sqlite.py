"""
export_to_sqlite.py — Data cleaning pipeline for Lending Club loan data.

Reads the raw CSV (accepted_2007_to_2018Q4.csv), applies all cleaning steps
documented in CLAUDE.md and docs/data_cleaning.md, and writes the result
to data/loans.db (SQLite).

Usage:
    source .env/bin/activate   # or .env\\Scripts\\activate on Windows
    python scripts/export_to_sqlite.py
"""

import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "accepted_2007_to_2018Q4.csv"
DB_PATH = PROJECT_ROOT / "data" / "loans.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: print a separator with step info
# ---------------------------------------------------------------------------
def _step(num: int, desc: str, df: pd.DataFrame) -> None:
    log.info(f"Step {num:>2d}: {desc}  —  rows: {len(df):,}")


# ===================================================================
# STEP 0 — Load raw data
# ===================================================================
def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Read the raw Lending Club CSV."""
    log.info(f"Loading {filepath.name} ...")
    df = pd.read_csv(filepath, low_memory=False)
    log.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


# ===================================================================
# STEP 1 — Drop non-loan rows (non-numeric id)
# ===================================================================
def drop_non_loan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where id is non-numeric (summary/policy rows in LC exports)."""
    before = len(df)
    df = df.copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.loc[df["id"].notna()].copy()
    df["id"] = df["id"].astype("int64")
    dropped = before - len(df)
    log.info(f"  Dropped {dropped:,} non-numeric id rows")
    _step(1, "Drop non-loan rows", df)
    return df


# ===================================================================
# STEP 2 — Keep only columns used by calculations, filters, or cleaning
# ===================================================================
# Whitelist of raw columns needed by any src/ function, dashboard filter, or cleaning step.
# All other columns are dropped to keep the DB under ~200MB instead of 1GB+.
KEEP_COLUMNS = [
    # Identifiers & terms
    "id", "loan_status", "funded_amnt", "int_rate", "term", "installment",
    "grade", "sub_grade", "purpose", "addr_state", "home_ownership",
    "issue_d", "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d",
    # Balances & payments
    "out_prncp", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "last_pymnt_amnt", "recoveries", "collection_recovery_fee",
    # Credit & income
    "fico_range_high", "fico_range_low", "last_fico_range_high",
    "last_fico_range_low", "dti", "dti_joint", "annual_inc", "annual_inc_joint",
    "application_type",
]


def keep_only_used_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the whitelisted columns; drop everything else."""
    before_cols = len(df.columns)

    present = [c for c in KEEP_COLUMNS if c in df.columns]
    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        log.warning(f"  Expected columns not found in raw data: {missing}")

    dropped_cols = [c for c in df.columns if c not in present]
    df = df[present].copy()

    log.info(f"  Kept {len(present)} columns, dropped {len(dropped_cols)}")
    log.info(f"  Columns: {before_cols} → {len(df.columns)}")
    _step(2, "Keep only used columns (whitelist)", df)
    return df


# ===================================================================
# STEP 3 — Convert int_rate from percentage string to decimal
# ===================================================================
def convert_interest_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Convert int_rate from percentage (e.g. 10.78) to decimal (0.1078)."""
    df = df.copy()
    df["int_rate"] = pd.to_numeric(df["int_rate"], errors="coerce") / 100.0
    log.info(f"  int_rate range: {df['int_rate'].min():.4f} – {df['int_rate'].max():.4f}")
    _step(3, "Convert int_rate to decimal", df)
    return df


# ===================================================================
# STEP 4 — Parse date columns
# ===================================================================
def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse issue_d, last_pymnt_d, next_pymnt_d, last_credit_pull_d as datetime."""
    df = df.copy()
    date_cols = ["issue_d", "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            non_null = df[col].notna().sum()
            log.info(f"  Parsed {col}: {non_null:,} non-null dates")
    _step(4, "Parse date columns", df)
    return df


# ===================================================================
# STEP 5 — Extract term_months as integer
# ===================================================================
def extract_term_months(df: pd.DataFrame) -> pd.DataFrame:
    """Extract term_months as integer from term string (e.g. ' 36 months' → 36)."""
    df = df.copy()
    df["term_months"] = (
        df["term"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype("Int64")
    )
    log.info(f"  term_months values: {sorted(df['term_months'].dropna().unique().tolist())}")
    _step(5, "Extract term_months", df)
    return df


# ===================================================================
# STEP 6 — Create maturity_month
# ===================================================================
def create_maturity_month(df: pd.DataFrame) -> pd.DataFrame:
    """Create maturity_month = issue_d + term_months."""
    df = df.copy()
    df["maturity_month"] = (
        df["issue_d"].dt.to_period("M") + df["term_months"]
    )
    _step(6, "Create maturity_month", df)
    return df


# ===================================================================
# STEP 7 — Drop rows with missing last_pymnt_d
# ===================================================================
def drop_missing_last_pymnt(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where last_pymnt_d is null (~2,415 loans)."""
    before = len(df)
    df = df.loc[df["last_pymnt_d"].notna()].copy()
    dropped = before - len(df)
    log.info(f"  Dropped {dropped:,} rows with null last_pymnt_d")
    _step(7, "Drop missing last_pymnt_d", df)
    return df


# ===================================================================
# STEP 8 — Drop "Does not meet the credit policy" loans
# ===================================================================
def drop_policy_loans(df: pd.DataFrame) -> pd.DataFrame:
    """Drop loans whose status starts with 'Does not meet the credit policy'."""
    before = len(df)
    policy_mask = df["loan_status"].str.startswith("Does not meet the credit policy")
    df = df.loc[~policy_mask].copy()
    dropped = before - len(df)
    log.info(f"  Dropped {dropped:,} credit-policy loans ({dropped/before*100:.2f}%)")
    _step(8, "Drop policy loans", df)
    return df


# ===================================================================
# STEP 9 — Reclassify Current loans with out_prncp == 0 as Fully Paid
# ===================================================================
def reclassify_current_zero_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Current loans with $0 outstanding principal are effectively Fully Paid."""
    df = df.copy()
    mask = (df["loan_status"] == "Current") & (df["out_prncp"] == 0)
    count = mask.sum()
    df.loc[mask, "loan_status"] = "Fully Paid"
    log.info(f"  Reclassified {count:,} Current → Fully Paid (out_prncp == 0)")
    _step(9, "Reclassify zero-balance Current → Fully Paid", df)
    return df


# ===================================================================
# STEP 10 — Drop Current loans where last_pymnt_d not Feb/March 2019
# ===================================================================
def drop_stale_current(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Current loans whose last payment isn't in Feb or March 2019."""
    before = len(df)
    feb_2019 = pd.Timestamp("2019-02-01")
    mar_2019 = pd.Timestamp("2019-03-01")
    stale_mask = (
        (df["loan_status"] == "Current")
        & (df["last_pymnt_d"] != feb_2019)
        & (df["last_pymnt_d"] != mar_2019)
    )
    dropped = stale_mask.sum()
    df = df.loc[~stale_mask].copy()
    log.info(f"  Dropped {dropped:,} Current loans with stale last_pymnt_d")
    _step(10, "Drop stale Current loans", df)
    return df


# ===================================================================
# STEP 11 — Set negative total_rec_late_fee to 0
# ===================================================================
def clean_negative_late_fees(df: pd.DataFrame) -> pd.DataFrame:
    """Set negative total_rec_late_fee values to 0."""
    df = df.copy()
    neg_mask = df["total_rec_late_fee"] < 0
    count = neg_mask.sum()
    df.loc[neg_mask, "total_rec_late_fee"] = 0.0
    log.info(f"  Zeroed {count:,} negative total_rec_late_fee values")
    _step(11, "Clean negative late fees", df)
    return df


# ===================================================================
# STEP 12 — Set total_rec_late_fee < $15 to 0 (globally)
# ===================================================================
def clean_small_late_fees(df: pd.DataFrame) -> pd.DataFrame:
    """Set total_rec_late_fee to 0 where < $15 (LC min late fee is $15)."""
    df = df.copy()
    df["total_rec_late_fee"] = np.round(df["total_rec_late_fee"], 2)
    small_mask = (df["total_rec_late_fee"] > 0) & (df["total_rec_late_fee"] < 15)
    count = small_mask.sum()
    df.loc[small_mask, "total_rec_late_fee"] = 0.0
    log.info(f"  Zeroed {count:,} late fees between $0–$15")
    _step(12, "Clean small late fees (< $15 → 0)", df)
    return df


# ===================================================================
# STEP 13 — Create current_late_fee_flag
# ===================================================================
def create_current_late_fee_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag Current loans that have late fees > $15 (after cleaning)."""
    df = df.copy()
    mask = (df["loan_status"] == "Current") & (df["total_rec_late_fee"] > 0)
    df["current_late_fee_flag"] = np.where(mask, 1, 0)
    count = df["current_late_fee_flag"].sum()
    log.info(f"  current_late_fee_flag = 1 for {count:,} loans")
    _step(13, "Create current_late_fee_flag", df)
    return df


# ===================================================================
# STEP 14 — Reclassify Grace/Late with out_prncp == 0; create flags
# ===================================================================
def reclassify_delinquent_zero_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Reclassify In Grace/Late loans with $0 balance as Fully Paid; create flags."""
    df = df.copy()

    mask_grace = (df["loan_status"] == "In Grace Period") & (df["out_prncp"] == 0)
    mask_late1 = (df["loan_status"] == "Late (16-30 days)") & (df["out_prncp"] == 0)
    mask_late2 = (df["loan_status"] == "Late (31-120 days)") & (df["out_prncp"] == 0)

    df["grace_to_paid_flag"] = np.where(mask_grace, 1, 0)
    df["late1_to_paid_flag"] = np.where(mask_late1, 1, 0)
    df["late2_to_paid_flag"] = np.where(mask_late2, 1, 0)

    total_reclassified = mask_grace.sum() + mask_late1.sum() + mask_late2.sum()
    log.info(f"  grace_to_paid: {mask_grace.sum():,}, late1_to_paid: {mask_late1.sum():,}, late2_to_paid: {mask_late2.sum():,}")

    # Reclassify to Fully Paid
    df.loc[mask_grace | mask_late1 | mask_late2, "loan_status"] = "Fully Paid"
    log.info(f"  Reclassified {total_reclassified:,} delinquent → Fully Paid")
    _step(14, "Reclassify zero-balance delinquent → Fully Paid", df)
    return df


# ===================================================================
# STEP 15 — Create upb_lost for Charged Off loans
# ===================================================================
def create_upb_lost(df: pd.DataFrame) -> pd.DataFrame:
    """Create upb_lost = -(funded_amnt - total_rec_prncp - recoveries) for Charged Off."""
    df = df.copy()
    mask = df["loan_status"] == "Charged Off"
    df["upb_lost"] = np.where(
        mask,
        -(df["funded_amnt"] - df["total_rec_prncp"] - df["recoveries"]),
        0.0,
    )
    co_count = mask.sum()
    avg_loss = df.loc[mask, "upb_lost"].mean() if co_count > 0 else 0
    log.info(f"  Charged Off loans: {co_count:,}, avg upb_lost: ${avg_loss:,.2f}")
    _step(15, "Create upb_lost", df)
    return df


# ===================================================================
# STEP 16 — Create joint_app_flag, dti_clean, annual_inc_clean
# ===================================================================
def create_joint_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create joint_app_flag, dti_clean, annual_inc_clean."""
    df = df.copy()

    # Joint application flag: 1 if not Individual
    df["joint_app_flag"] = np.where(df["application_type"] != "Individual", 1, 0)
    joint_count = df["joint_app_flag"].sum()
    log.info(f"  joint_app_flag = 1 for {joint_count:,} loans")

    # dti_clean: use dti_joint for joint apps, dti for individual
    df["dti_clean"] = np.where(
        df["joint_app_flag"] == 0,
        df["dti"],
        df["dti_joint"],
    )

    # annual_inc_clean: same logic
    df["annual_inc_clean"] = np.where(
        df["joint_app_flag"] == 0,
        df["annual_inc"],
        df["annual_inc_joint"],
    )

    _step(16, "Create joint_app_flag, dti_clean, annual_inc_clean", df)
    return df


# ===================================================================
# STEP 17 — Create original_fico, latest_fico
# ===================================================================
def create_fico_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create averaged FICO columns."""
    df = df.copy()
    df["original_fico"] = (df["fico_range_high"] + df["fico_range_low"]) / 2
    df["latest_fico"] = (df["last_fico_range_high"] + df["last_fico_range_low"]) / 2
    log.info(f"  original_fico range: {df['original_fico'].min():.0f} – {df['original_fico'].max():.0f}")
    log.info(f"  latest_fico range:   {df['latest_fico'].min():.0f} – {df['latest_fico'].max():.0f}")
    _step(17, "Create original_fico, latest_fico", df)
    return df


# ===================================================================
# STEP 18 — Clean dti_clean: set negative to 0, drop nulls
# ===================================================================
def clean_dti(df: pd.DataFrame) -> pd.DataFrame:
    """Set negative dti_clean to 0; drop rows with null dti_clean."""
    df = df.copy()

    # Set negative values to 0
    neg_mask = df["dti_clean"] < 0
    neg_count = neg_mask.sum()
    df.loc[neg_mask, "dti_clean"] = 0.0
    log.info(f"  Set {neg_count:,} negative dti_clean values to 0")

    # Drop null dti_clean
    before = len(df)
    df = df.loc[df["dti_clean"].notna()].copy()
    dropped = before - len(df)
    log.info(f"  Dropped {dropped:,} rows with null dti_clean")
    _step(18, "Clean dti_clean", df)
    return df


# ===================================================================
# STEP 19 — Create issue_quarter, issue_month_year
# ===================================================================
def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create issue_quarter and issue_month_year from issue_d."""
    df = df.copy()
    df["issue_quarter"] = df["issue_d"].dt.to_period("Q").astype(str)
    df["issue_month_year"] = df["issue_d"].dt.strftime("%b-%Y")
    log.info(f"  issue_quarter range: {df['issue_quarter'].min()} – {df['issue_quarter'].max()}")
    _step(19, "Create issue_quarter, issue_month_year", df)
    return df


# ===================================================================
# STEP 20 — Create curr_paid_late1_flag (for transition matrix)
# ===================================================================
def create_curr_paid_late1_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag Current/Fully Paid loans that had late fees > $15 (reached Late 16-30)."""
    df = df.copy()
    mask = (
        df["loan_status"].isin(["Fully Paid", "Current"])
        & (df["total_rec_late_fee"] > 0)
    )
    df["curr_paid_late1_flag"] = np.where(mask, 1, 0)
    count = df["curr_paid_late1_flag"].sum()
    log.info(f"  curr_paid_late1_flag = 1 for {count:,} loans")
    _step(20, "Create curr_paid_late1_flag", df)
    return df


# ===================================================================
# EXPORT — Write to SQLite
# ===================================================================
def export_to_sqlite(df: pd.DataFrame, db_path: Path) -> None:
    """Write cleaned DataFrame to SQLite database."""
    # Convert Period columns to string for SQLite compatibility
    if "maturity_month" in df.columns:
        df["maturity_month"] = df["maturity_month"].astype(str)

    # Convert datetime columns to string for SQLite
    date_cols = ["issue_d", "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d"]
    for col in date_cols:
        if col in df.columns:
            df[col] = df[col].dt.strftime("%Y-%m-%d")

    # Remove the database if it exists (fresh write)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    df.to_sql("loans", conn, index=False, if_exists="replace")
    conn.close()

    size_mb = db_path.stat().st_size / (1024 * 1024)
    log.info(f"Wrote {len(df):,} rows to {db_path.name} ({size_mb:.1f} MB)")


# ===================================================================
# SUMMARY — Print final statistics
# ===================================================================
def print_summary(df: pd.DataFrame, initial_rows: int) -> None:
    """Print final summary statistics."""
    log.info("=" * 70)
    log.info("FINAL SUMMARY")
    log.info("=" * 70)
    log.info(f"Initial rows:  {initial_rows:,}")
    log.info(f"Final rows:    {len(df):,}")
    log.info(f"Rows dropped:  {initial_rows - len(df):,} ({(initial_rows - len(df))/initial_rows*100:.2f}%)")
    log.info(f"Columns:       {len(df.columns)}")
    log.info("")

    # Loan status distribution
    status_counts = df["loan_status"].value_counts()
    log.info("Loan status distribution:")
    for status, count in status_counts.items():
        pct = count / len(df) * 100
        log.info(f"  {status:<35s} {count:>10,}  ({pct:5.2f}%)")

    # March 2019 Current loans (cash flow population)
    mar_current = df[
        (df["loan_status"] == "Current")
        & (df["last_pymnt_d"] == "2019-03-01")
    ]
    log.info(f"\nMarch 2019 Current loans (CF population): {len(mar_current):,}")
    log.info(f"  Total UPB: ${mar_current['out_prncp'].sum():,.0f}")

    # Grade distribution
    log.info("\nGrade distribution:")
    for grade in sorted(df["grade"].dropna().unique()):
        count = (df["grade"] == grade).sum()
        log.info(f"  {grade}: {count:,}")


# ===================================================================
# MAIN
# ===================================================================
def main() -> None:
    """Run the full cleaning pipeline."""
    if not RAW_CSV.exists():
        log.error(f"Raw CSV not found: {RAW_CSV}")
        sys.exit(1)

    # Load
    df = load_raw_data(RAW_CSV)
    initial_rows = len(df)

    # Clean (steps 1–20, matching CLAUDE.md)
    df = drop_non_loan_rows(df)           # Step 1
    df = keep_only_used_columns(df)       # Step 2
    df = convert_interest_rate(df)        # Step 3
    df = parse_dates(df)                  # Step 4
    df = extract_term_months(df)          # Step 5
    df = create_maturity_month(df)        # Step 6
    df = drop_missing_last_pymnt(df)      # Step 7
    df = drop_policy_loans(df)            # Step 8
    df = reclassify_current_zero_balance(df)  # Step 9
    df = drop_stale_current(df)           # Step 10
    df = clean_negative_late_fees(df)     # Step 11
    df = clean_small_late_fees(df)        # Step 12
    df = create_current_late_fee_flag(df) # Step 13
    df = reclassify_delinquent_zero_balance(df)  # Step 14
    df = create_upb_lost(df)              # Step 15
    df = create_joint_columns(df)         # Step 16
    df = create_fico_columns(df)          # Step 17
    df = clean_dti(df)                    # Step 18
    df = create_date_features(df)         # Step 19
    df = create_curr_paid_late1_flag(df)  # Step 20

    # Summary and export
    print_summary(df, initial_rows)
    export_to_sqlite(df, DB_PATH)
    log.info("Done.")


if __name__ == "__main__":
    main()
