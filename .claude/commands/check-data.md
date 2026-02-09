Run quick data integrity checks against the SQLite database.

## Steps

Use the sqlite MCP to run these queries against `data/loans.db`:

### 1. Row Count
```sql
SELECT COUNT(*) as total_loans FROM loans;
```
Expected: approximately 2.2M rows (exact count depends on cleaning drops).

### 2. Null Counts for Critical Columns
```sql
SELECT
  SUM(CASE WHEN loan_status IS NULL THEN 1 ELSE 0 END) as null_loan_status,
  SUM(CASE WHEN funded_amnt IS NULL THEN 1 ELSE 0 END) as null_funded_amnt,
  SUM(CASE WHEN int_rate IS NULL THEN 1 ELSE 0 END) as null_int_rate,
  SUM(CASE WHEN out_prncp IS NULL THEN 1 ELSE 0 END) as null_out_prncp,
  SUM(CASE WHEN grade IS NULL THEN 1 ELSE 0 END) as null_grade,
  SUM(CASE WHEN term_months IS NULL THEN 1 ELSE 0 END) as null_term_months
FROM loans;
```
Expected: All zeros for these critical columns.

### 3. Loan Status Distribution
```sql
SELECT loan_status, COUNT(*) as cnt,
       ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM loans), 2) as pct
FROM loans
GROUP BY loan_status
ORDER BY cnt DESC;
```
Report the distribution. Current and Fully Paid should dominate.

### 4. Interest Rate Format Check
```sql
SELECT MIN(int_rate), MAX(int_rate), AVG(int_rate) FROM loans;
```
Expected: All values should be decimals (0.05 to 0.31 range). If max > 1.0, the rate was not properly converted from percentage.

### 5. Vintage Distribution
```sql
SELECT issue_quarter, COUNT(*) as cnt
FROM loans
WHERE issue_quarter IS NOT NULL
GROUP BY issue_quarter
ORDER BY issue_quarter;
```
Confirm `issue_quarter` column exists and shows vintages from 2007Q2 through 2018Q4.

### 6. Current Loans with March 2019 Payment
```sql
SELECT COUNT(*) as march_current_count,
       SUM(out_prncp) as march_current_upb
FROM loans
WHERE loan_status = 'Current'
  AND last_pymnt_d = '2019-03-01';
```
This is the population for cash flow projections. Report the count and total UPB.

### 7. Engineered Columns Exist
```sql
SELECT name FROM pragma_table_info('loans')
WHERE name IN ('issue_quarter', 'term_months', 'original_fico', 'latest_fico', 'dti_clean');
```
Confirm these cleaning-derived columns are present.

### 8. Total Column Count
```sql
SELECT COUNT(*) as col_count FROM pragma_table_info('loans');
```
Should be approximately 30-40 columns (29 raw kept + ~10-15 engineered). If it's over 100, the column whitelist was not applied during export.

## Output Format

```
DATA INTEGRITY CHECK
=====================
1. Total rows: 2,260,XXX ✓
2. Null counts: All critical columns clean ✓ / ✗ [details]
3. Status distribution: [summary table]
4. Interest rate range: 0.0525 - 0.3089 ✓ (decimal format)
5. Vintages: 2007Q2 - 2018Q4 ✓ (XX quarters)
6. March 2019 Current: XX,XXX loans, $X.XB UPB
7. Engineered columns: [present/missing list]
8. Column count: XX columns (expected 30-40)

ANOMALIES: [list anything unexpected, or "None found"]
```
