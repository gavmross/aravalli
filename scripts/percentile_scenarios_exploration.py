# %% [markdown]
# # Percentile-Based Scenario Assumptions
#
# Explores replacing fixed multiplicative stress/upside shifts with
# empirical percentiles computed from vintage-level or loan-level distributions.
#
# **CDR**: Per-vintage trailing 12-month CDR → use percentiles for stress/upside.
# **CPR**: Two approaches compared — per-loan implied CPR from cumulative
# prepayments, and per-vintage pool CPR from calc_amort columns.

# %%
import sys
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.insert(0, '.')

from src.amortization import calc_amort, calc_monthly_payment, calc_balance
from src.portfolio_analytics import reconstruct_loan_timeline
from src.cashflow_engine import compute_pool_assumptions

pd.set_option('display.float_format', '{:.6f}'.format)

# %% [markdown]
# ## 1. Load and Prepare Data

# %%
conn = sqlite3.connect('data/loans.db')
df = pd.read_sql('SELECT * FROM loans', conn)
conn.close()
print(f"Loaded {len(df):,} loans")

# Parse dates
for col in ['issue_d', 'last_pymnt_d', 'maturity_month']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# %%
# Run calc_amort
df = calc_amort(df)
print("calc_amort columns added")

# Reconstruct timeline (needed for CDR)
df = reconstruct_loan_timeline(df)
print("Timeline reconstructed")

# %%
# Create vintage columns
df['vintage_year'] = df['issue_d'].dt.year
df['vintage_quarter'] = df['issue_d'].dt.to_period('Q').astype(str)

# Active loans definition (matches compute_pool_assumptions)
ACTIVE_STATUSES = ['Current', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)']
SNAPSHOT = pd.Timestamp('2019-03-01')

df_active = df[df['loan_status'].isin(ACTIVE_STATUSES)].copy()
df_current = df_active[
    (df_active['loan_status'] == 'Current') &
    (df_active['last_pymnt_d'] == SNAPSHOT)
].copy()

print(f"Total loans:   {len(df):,}")
print(f"Active loans:  {len(df_active):,}")
print(f"Current (Mar): {len(df_current):,}")

# %% [markdown]
# ---
# ## 2. Per-Vintage CDR
#
# For each vintage group, compute the CDR using the same trailing 12-month
# methodology as `compute_pool_assumptions()` — but restricted to loans
# from that vintage only.

# %%
def compute_cdr_for_subset(df_subset: pd.DataFrame) -> dict:
    """
    Compute trailing 12-month CDR for a subset of loans.
    Same methodology as compute_pool_assumptions() CDR path.
    Returns dict with 'cdr', 'avg_mdr', 'monthly_mdrs', 'n_loans'.
    """
    snapshot = pd.Timestamp('2019-03-01')
    issue_dates = pd.to_datetime(df_subset['issue_d'])
    default_month = pd.to_datetime(df_subset['default_month'], errors='coerce')
    payoff_month = pd.to_datetime(df_subset['payoff_month'], errors='coerce')

    monthly_mdrs = []
    for m in range(1, 13):
        month_start = snapshot - pd.DateOffset(months=m)
        month_end = month_start + pd.DateOffset(months=1)

        defaults_mask = (default_month >= month_start) & (default_month < month_end)
        default_upb = (
            df_subset.loc[defaults_mask, 'funded_amnt'] -
            df_subset.loc[defaults_mask, 'total_rec_prncp']
        ).clip(lower=0).sum()

        originated = issue_dates <= month_start
        not_defaulted = default_month.isna() | (default_month > month_start)
        not_paid_off = payoff_month.isna() | (payoff_month > month_start)
        performing_mask = originated & not_defaulted & not_paid_off

        perf_idx = performing_mask.values.nonzero()[0]

        if len(perf_idx) > 0:
            funded = df_subset['funded_amnt'].values[perf_idx].astype(np.float64)
            rates = df_subset['int_rate'].values[perf_idx].astype(np.float64)
            terms = df_subset['term_months'].values[perf_idx].astype(np.float64)

            perf_issue = issue_dates.values[perf_idx]
            td = np.datetime64(month_start) - perf_issue
            age_at_month = np.round(
                td.astype('timedelta64[D]').astype(np.float64) / 30.44
            ).astype(int).clip(min=0)

            pmt = calc_monthly_payment(funded, rates, terms)
            est_balance, _, _ = calc_balance(funded, rates, pmt,
                                             age_at_month.astype(np.float64))
            performing_balance = np.clip(est_balance, 0, None).sum()
        else:
            performing_balance = 0.0

        mdr = default_upb / performing_balance if performing_balance > 0 else 0.0
        monthly_mdrs.append(mdr)

    avg_mdr = float(np.mean(monthly_mdrs))
    cdr = 1 - (1 - avg_mdr) ** 12

    return {
        'cdr': cdr,
        'avg_mdr': avg_mdr,
        'monthly_mdrs': monthly_mdrs,
        'n_loans': len(df_subset),
    }

# %% [markdown]
# ### 2a. Annual Vintage CDR

# %%
vintage_years = sorted(df['vintage_year'].dropna().unique())
print(f"Vintage years: {vintage_years}")

annual_cdrs = {}
for yr in vintage_years:
    subset = df[df['vintage_year'] == yr]
    result = compute_cdr_for_subset(subset)
    annual_cdrs[yr] = result
    print(f"  {yr}: CDR = {result['cdr']:.4%}  ({result['n_loans']:,} loans)")

annual_cdr_series = pd.Series({yr: v['cdr'] for yr, v in annual_cdrs.items()})

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
ax = axes[0]
colors = ['#2196F3' if cdr > 0 else '#ccc' for cdr in annual_cdr_series.values]
ax.bar(annual_cdr_series.index.astype(str), annual_cdr_series.values, color=colors)
ax.set_title('CDR by Annual Vintage')
ax.set_ylabel('CDR')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.tick_params(axis='x', rotation=45)

# Show percentile lines
for pctl, color, ls in [(25, 'green', '--'), (50, 'orange', '-'), (75, 'red', '--')]:
    val = np.percentile(annual_cdr_series.values, pctl)
    ax.axhline(val, color=color, ls=ls, lw=1.5, label=f'P{pctl} = {val:.2%}')
ax.legend(fontsize=9)

# Distribution
ax = axes[1]
ax.hist(annual_cdr_series.values, bins=max(len(annual_cdr_series) // 2, 5),
        edgecolor='white', color='#2196F3', alpha=0.7)
ax.set_title(f'CDR Distribution (n={len(annual_cdr_series)} vintages)')
ax.set_xlabel('CDR')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.tight_layout()
plt.show()

# %%
# Percentile summary
print("Annual Vintage CDR Percentiles:")
for p in [10, 25, 50, 75, 90]:
    print(f"  P{p:2d}: {np.percentile(annual_cdr_series.values, p):.4%}")

# %% [markdown]
# ### 2b. Quarterly Vintage CDR
#
# More data points but noisier — fewer loans per cell.

# %%
quarters = sorted(df['vintage_quarter'].dropna().unique())
print(f"Quarterly vintages: {len(quarters)} quarters")

quarterly_cdrs = {}
for q in quarters:
    subset = df[df['vintage_quarter'] == q]
    result = compute_cdr_for_subset(subset)
    quarterly_cdrs[q] = result

quarterly_cdr_series = pd.Series({q: v['cdr'] for q, v in quarterly_cdrs.items()})
quarterly_n_series = pd.Series({q: v['n_loans'] for q, v in quarterly_cdrs.items()})

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.bar(range(len(quarterly_cdr_series)), quarterly_cdr_series.values,
       color='#2196F3', alpha=0.7)
ax.set_title(f'CDR by Quarterly Vintage (n={len(quarterly_cdr_series)})')
ax.set_ylabel('CDR')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
# Label every 4th quarter
tick_idx = list(range(0, len(quarterly_cdr_series), 4))
ax.set_xticks(tick_idx)
ax.set_xticklabels([quarterly_cdr_series.index[i] for i in tick_idx], rotation=45, fontsize=8)

for pctl, color, ls in [(25, 'green', '--'), (50, 'orange', '-'), (75, 'red', '--')]:
    val = np.percentile(quarterly_cdr_series.values, pctl)
    ax.axhline(val, color=color, ls=ls, lw=1.5, label=f'P{pctl} = {val:.2%}')
ax.legend(fontsize=9)

ax = axes[1]
ax.hist(quarterly_cdr_series.values, bins=20, edgecolor='white',
        color='#2196F3', alpha=0.7)
ax.set_title(f'CDR Distribution (n={len(quarterly_cdr_series)} quarters)')
ax.set_xlabel('CDR')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.tight_layout()
plt.show()

# %%
# Percentile summary
print("Quarterly Vintage CDR Percentiles:")
for p in [10, 25, 50, 75, 90]:
    print(f"  P{p:2d}: {np.percentile(quarterly_cdr_series.values, p):.4%}")

# %% [markdown]
# ### 2c. UPB-Weighted Quarterly CDR
#
# Vintages with more outstanding balance should arguably count more.
# Weight each vintage's CDR by its performing UPB at snapshot.

# %%
quarterly_upb = {}
for q in quarters:
    subset = df[df['vintage_quarter'] == q]
    active_subset = subset[subset['loan_status'].isin(ACTIVE_STATUSES)]
    quarterly_upb[q] = active_subset['out_prncp'].sum()

quarterly_upb_series = pd.Series(quarterly_upb)

# Only vintages with meaningful active balance
mask = quarterly_upb_series > 0
weighted_cdrs = quarterly_cdr_series[mask]
weighted_upbs = quarterly_upb_series[mask]

print(f"Vintages with active UPB: {mask.sum()} / {len(mask)}")
print(f"\nUPB-weighted percentiles:")
# Sort by CDR for percentile computation
sorted_idx = weighted_cdrs.argsort()
sorted_cdrs = weighted_cdrs.iloc[sorted_idx].values
sorted_weights = weighted_upbs.iloc[sorted_idx].values
cum_weights = np.cumsum(sorted_weights) / sorted_weights.sum()

for p in [25, 50, 75]:
    idx = np.searchsorted(cum_weights, p / 100.0)
    idx = min(idx, len(sorted_cdrs) - 1)
    print(f"  P{p}: {sorted_cdrs[idx]:.4%}")

# %% [markdown]
# ---
# ## 3. Per-Loan Implied CPR (Cumulative Prepayment Approach)
#
# For Current loans only. Uses the gap between scheduled balance and
# actual balance to back into an implied constant SMM.
#
# ```
# actual_balance ≈ sched_balance × (1 - SMM)^age
# implied_smm = 1 - (actual / scheduled)^(1/age)
# implied_cpr = 1 - (1 - smm)^12
# ```

# %%
def compute_implied_cpr_per_loan(df_curr: pd.DataFrame) -> pd.Series:
    """
    For each Current loan, compute implied CPR from cumulative prepayment.
    Returns a Series of per-loan implied CPR values.
    """
    funded = df_curr['funded_amnt'].values.astype(np.float64)
    rates = df_curr['int_rate'].values.astype(np.float64)
    terms = df_curr['term_months'].values.astype(np.float64)
    actual = df_curr['out_prncp'].values.astype(np.float64)

    # Age at snapshot
    issue_dates = pd.to_datetime(df_curr['issue_d'])
    td = np.datetime64(SNAPSHOT) - issue_dates.values
    age = np.round(
        td.astype('timedelta64[D]').astype(np.float64) / 30.44
    ).astype(int).clip(min=0)

    # Scheduled balance at current age (zero-prepayment assumption)
    pmt = calc_monthly_payment(funded, rates, terms)
    sched, _, _ = calc_balance(funded, rates, pmt, age.astype(np.float64))
    sched = np.clip(sched, 0, None)

    # Ratio = actual / scheduled
    # Only valid where sched > 0 and age > 0
    valid = (sched > 0) & (age > 0) & (actual >= 0)
    ratio = np.ones(len(df_curr))
    ratio[valid] = actual[valid] / sched[valid]

    # Clamp ratio to (0, 1] — ratio > 1 means underpayment, treat as no prepay
    ratio = np.clip(ratio, 1e-10, 1.0)

    # implied_smm = 1 - ratio^(1/age)
    implied_smm = np.zeros(len(df_curr))
    implied_smm[valid] = 1.0 - np.power(ratio[valid], 1.0 / age[valid])
    implied_smm = np.clip(implied_smm, 0, 1)  # no negative SMM

    # implied_cpr = 1 - (1 - smm)^12
    implied_cpr = 1.0 - (1.0 - implied_smm) ** 12

    return pd.Series(implied_cpr, index=df_curr.index, name='implied_cpr')

# %%
loan_cpr = compute_implied_cpr_per_loan(df_current)
print(f"Per-loan implied CPR computed for {len(loan_cpr):,} Current loans")
print(f"\nBasic stats:")
print(loan_cpr.describe())

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Full distribution (lots of zeros expected)
ax = axes[0]
ax.hist(loan_cpr.values, bins=100, edgecolor='none', color='#4CAF50', alpha=0.7)
ax.set_title('Per-Loan Implied CPR Distribution (all)')
ax.set_xlabel('Implied CPR')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylabel('Count')

# Non-zero only
ax = axes[1]
nonzero = loan_cpr[loan_cpr > 0.001]  # threshold at 0.1% to exclude noise
ax.hist(nonzero.values, bins=80, edgecolor='none', color='#4CAF50', alpha=0.7)
ax.set_title(f'Implied CPR > 0.1% (n={len(nonzero):,} of {len(loan_cpr):,})')
ax.set_xlabel('Implied CPR')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Capped at 50% for readability
ax = axes[2]
capped = loan_cpr[loan_cpr.between(0.001, 0.50)]
ax.hist(capped.values, bins=60, edgecolor='none', color='#4CAF50', alpha=0.7)
ax.set_title(f'Implied CPR 0.1%–50% (n={len(capped):,})')
ax.set_xlabel('Implied CPR')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.tight_layout()
plt.show()

# %%
print("Per-Loan Implied CPR Percentiles (all Current loans):")
for p in [10, 25, 50, 75, 90]:
    print(f"  P{p:2d}: {np.percentile(loan_cpr.values, p):.4%}")

print(f"\nFraction with zero/negligible CPR (<0.1%): {(loan_cpr < 0.001).mean():.1%}")

# %% [markdown]
# ### 3a. UPB-Weighted Per-Loan CPR Percentiles
#
# Larger loans should count more — a $25K loan prepaying at 20% CPR
# matters more than a $1K loan at 20% CPR.

# %%
# Weighted percentiles
upb = df_current['out_prncp'].values
sorted_idx = loan_cpr.values.argsort()
sorted_cpr = loan_cpr.values[sorted_idx]
sorted_upb = upb[sorted_idx]
cum_upb = np.cumsum(sorted_upb) / sorted_upb.sum()

print("UPB-Weighted Per-Loan CPR Percentiles:")
for p in [10, 25, 50, 75, 90]:
    idx = np.searchsorted(cum_upb, p / 100.0)
    idx = min(idx, len(sorted_cpr) - 1)
    print(f"  P{p:2d}: {sorted_cpr[idx]:.4%}")

# %% [markdown]
# ---
# ## 4. Per-Vintage Pool CPR
#
# For each vintage's Current loans, aggregate the calc_amort columns
# (beginning_balance, scheduled_principal, unscheduled_principal)
# and compute pool-level SMM → CPR. Symmetric with CDR approach.

# %%
def compute_pool_cpr_for_subset(df_curr_subset: pd.DataFrame) -> dict:
    """
    Compute pool-level CPR from Current loans in a subset.
    Same methodology as compute_pool_assumptions() CPR path.
    """
    paid = df_curr_subset[df_curr_subset['last_pmt_beginning_balance'] > 0]
    if len(paid) == 0:
        return {'cpr': 0.0, 'smm': 0.0, 'n_loans': 0}

    total_beg_bal = paid['last_pmt_beginning_balance'].sum()
    total_sched_princ = paid['last_pmt_scheduled_principal'].sum()
    total_unsched_princ = paid['last_pmt_unscheduled_principal'].sum()

    denom = total_beg_bal - total_sched_princ
    if denom > 0:
        smm = total_unsched_princ / denom
        cpr = 1 - (1 - smm) ** 12
    else:
        smm = 0.0
        cpr = 0.0

    return {'cpr': cpr, 'smm': smm, 'n_loans': len(paid)}

# %%
# Annual vintage CPR
annual_cprs = {}
for yr in vintage_years:
    subset = df_current[df_current['vintage_year'] == yr]
    result = compute_pool_cpr_for_subset(subset)
    annual_cprs[yr] = result
    if result['n_loans'] > 0:
        print(f"  {yr}: CPR = {result['cpr']:.4%}  ({result['n_loans']:,} Current loans)")

annual_cpr_series = pd.Series({yr: v['cpr'] for yr, v in annual_cprs.items()
                                if v['n_loans'] > 0})

# %%
# Quarterly vintage CPR
quarterly_cprs = {}
for q in quarters:
    subset = df_current[df_current['vintage_quarter'] == q]
    result = compute_pool_cpr_for_subset(subset)
    quarterly_cprs[q] = result

quarterly_cpr_series = pd.Series({q: v['cpr'] for q, v in quarterly_cprs.items()
                                   if v['n_loans'] > 0})

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Annual vintage CPR
ax = axes[0, 0]
ax.bar(annual_cpr_series.index.astype(str), annual_cpr_series.values,
       color='#4CAF50')
ax.set_title('Pool CPR by Annual Vintage')
ax.set_ylabel('CPR')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
for pctl, color, ls in [(25, 'red', '--'), (50, 'orange', '-'), (75, 'green', '--')]:
    val = np.percentile(annual_cpr_series.values, pctl)
    ax.axhline(val, color=color, ls=ls, lw=1.5, label=f'P{pctl} = {val:.2%}')
ax.legend(fontsize=9)

# Annual vintage CPR distribution
ax = axes[0, 1]
ax.hist(annual_cpr_series.values, bins=max(len(annual_cpr_series) // 2, 5),
        edgecolor='white', color='#4CAF50', alpha=0.7)
ax.set_title(f'Pool CPR Distribution (n={len(annual_cpr_series)} vintages)')
ax.set_xlabel('CPR')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Quarterly vintage CPR
ax = axes[1, 0]
ax.bar(range(len(quarterly_cpr_series)), quarterly_cpr_series.values,
       color='#4CAF50', alpha=0.7)
ax.set_title(f'Pool CPR by Quarterly Vintage (n={len(quarterly_cpr_series)})')
ax.set_ylabel('CPR')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
tick_idx = list(range(0, len(quarterly_cpr_series), 4))
ax.set_xticks(tick_idx)
ax.set_xticklabels([quarterly_cpr_series.index[i] for i in tick_idx],
                    rotation=45, fontsize=8)

# Quarterly distribution
ax = axes[1, 1]
ax.hist(quarterly_cpr_series.values, bins=20, edgecolor='white',
        color='#4CAF50', alpha=0.7)
ax.set_title(f'Pool CPR Distribution (n={len(quarterly_cpr_series)} quarters)')
ax.set_xlabel('CPR')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.tight_layout()
plt.show()

# %%
print("Annual Vintage Pool CPR Percentiles:")
for p in [10, 25, 50, 75, 90]:
    print(f"  P{p:2d}: {np.percentile(annual_cpr_series.values, p):.4%}")

print(f"\nQuarterly Vintage Pool CPR Percentiles:")
for p in [10, 25, 50, 75, 90]:
    print(f"  P{p:2d}: {np.percentile(quarterly_cpr_series.values, p):.4%}")

# %% [markdown]
# ---
# ## 5. Head-to-Head Comparison
#
# Compare all approaches side by side: current multiplicative, per-vintage
# percentiles (annual & quarterly), and per-loan implied CPR percentiles.

# %%
# Get pool-level base assumptions for reference
pool_assumptions = compute_pool_assumptions(df, df_active)
base_cdr = pool_assumptions['cdr']
base_cpr = pool_assumptions['cpr']
base_ls = pool_assumptions['loss_severity']

print(f"Pool-level base assumptions:")
print(f"  CDR: {base_cdr:.4%}")
print(f"  CPR: {base_cpr:.4%}")
print(f"  Loss Severity: {base_ls:.4%}")

# %%
# Build comparison table
stress_pct = 0.15

approaches = {
    'Current (±15% mult)': {
        'stress_cdr': base_cdr * (1 + stress_pct),
        'base_cdr': base_cdr,
        'upside_cdr': base_cdr * (1 - stress_pct),
        'stress_cpr': base_cpr * (1 - stress_pct),
        'base_cpr': base_cpr,
        'upside_cpr': base_cpr * (1 + stress_pct),
    },
    'Annual Vintage P25/P75': {
        'stress_cdr': np.percentile(annual_cdr_series.values, 75),
        'base_cdr': base_cdr,  # keep pool-level as base
        'upside_cdr': np.percentile(annual_cdr_series.values, 25),
        'stress_cpr': np.percentile(annual_cpr_series.values, 25),
        'base_cpr': base_cpr,
        'upside_cpr': np.percentile(annual_cpr_series.values, 75),
    },
    'Quarterly Vintage P25/P75': {
        'stress_cdr': np.percentile(quarterly_cdr_series.values, 75),
        'base_cdr': base_cdr,
        'upside_cdr': np.percentile(quarterly_cdr_series.values, 25),
        'stress_cpr': np.percentile(quarterly_cpr_series.values, 25),
        'base_cpr': base_cpr,
        'upside_cpr': np.percentile(quarterly_cpr_series.values, 75),
    },
    'Per-Loan Implied CPR P25/P75': {
        'stress_cdr': np.percentile(annual_cdr_series.values, 75),  # same CDR
        'base_cdr': base_cdr,
        'upside_cdr': np.percentile(annual_cdr_series.values, 25),
        'stress_cpr': np.percentile(loan_cpr.values, 25),
        'base_cpr': base_cpr,
        'upside_cpr': np.percentile(loan_cpr.values, 75),
    },
    'Per-Loan CPR (UPB-Weighted)': {
        'stress_cdr': np.percentile(annual_cdr_series.values, 75),
        'base_cdr': base_cdr,
        'upside_cdr': np.percentile(annual_cdr_series.values, 25),
        'stress_cpr': sorted_cpr[np.searchsorted(cum_upb, 0.25)],
        'base_cpr': base_cpr,
        'upside_cpr': sorted_cpr[min(np.searchsorted(cum_upb, 0.75), len(sorted_cpr) - 1)],
    },
}

# %%
# Display as formatted table
rows = []
for name, vals in approaches.items():
    rows.append({
        'Approach': name,
        'Stress CDR': f"{vals['stress_cdr']:.2%}",
        'Base CDR': f"{vals['base_cdr']:.2%}",
        'Upside CDR': f"{vals['upside_cdr']:.2%}",
        'CDR Range': f"{vals['stress_cdr'] - vals['upside_cdr']:.2%}",
        'Stress CPR': f"{vals['stress_cpr']:.2%}",
        'Base CPR': f"{vals['base_cpr']:.2%}",
        'Upside CPR': f"{vals['upside_cpr']:.2%}",
        'CPR Range': f"{vals['stress_cpr'] - vals['upside_cpr']:.2%}",  # note: negative = wider spread
    })

comparison_df = pd.DataFrame(rows).set_index('Approach')
print(comparison_df.to_string())

# %%
# Visual comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

approach_names = list(approaches.keys())
y_pos = np.arange(len(approach_names))

# CDR comparison
ax = axes[0]
for i, (name, vals) in enumerate(approaches.items()):
    ax.barh(i, vals['stress_cdr'] - vals['upside_cdr'],
            left=vals['upside_cdr'], color='#ffcdd2', height=0.5)
    ax.plot(vals['base_cdr'], i, 'ko', ms=8)
    ax.plot(vals['stress_cdr'], i, 'rv', ms=8)
    ax.plot(vals['upside_cdr'], i, 'g^', ms=8)
ax.set_yticks(y_pos)
ax.set_yticklabels(approach_names, fontsize=9)
ax.set_title('CDR: Stress / Base / Upside')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_xlabel('CDR')

# CPR comparison
ax = axes[1]
for i, (name, vals) in enumerate(approaches.items()):
    low = min(vals['stress_cpr'], vals['upside_cpr'])
    high = max(vals['stress_cpr'], vals['upside_cpr'])
    ax.barh(i, high - low, left=low, color='#c8e6c9', height=0.5)
    ax.plot(vals['base_cpr'], i, 'ko', ms=8)
    ax.plot(vals['stress_cpr'], i, 'rv', ms=8)
    ax.plot(vals['upside_cpr'], i, 'g^', ms=8)
ax.set_yticks(y_pos)
ax.set_yticklabels(approach_names, fontsize=9)
ax.set_title('CPR: Stress / Base / Upside')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_xlabel('CPR')

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 6. CDR/CPR Percentiles by Grade
#
# Show how the approach works within a strata filter (e.g., by grade).
# This is what the dashboard would compute when a user filters by grade.

# %%
grades = sorted(df['grade'].dropna().unique())

grade_results = {}
for g in grades:
    df_grade = df[df['grade'] == g]
    df_grade_current = df_current[df_current['grade'] == g]

    # Pool-level
    if len(df_grade_current) > 0:
        pool = compute_pool_assumptions(df_grade, df_grade[df_grade['loan_status'].isin(ACTIVE_STATUSES)])
    else:
        pool = {'cdr': 0, 'cpr': 0}

    # Per-vintage CDR (quarterly)
    vintage_cdrs = []
    vintage_cprs = []
    for q in quarters:
        sub_all = df_grade[df_grade['vintage_quarter'] == q]
        sub_curr = df_grade_current[df_grade_current['vintage_quarter'] == q]
        if len(sub_all) > 50:  # minimum loan count for meaningful CDR
            cdr_result = compute_cdr_for_subset(sub_all)
            vintage_cdrs.append(cdr_result['cdr'])
        if len(sub_curr) > 20:
            cpr_result = compute_pool_cpr_for_subset(sub_curr)
            if cpr_result['n_loans'] > 0:
                vintage_cprs.append(cpr_result['cpr'])

    grade_results[g] = {
        'pool_cdr': pool['cdr'],
        'pool_cpr': pool['cpr'],
        'vintage_cdrs': vintage_cdrs,
        'vintage_cprs': vintage_cprs,
    }

# %%
# Show CDR percentiles by grade
print(f"{'Grade':<6} {'Pool CDR':>10} {'P25 CDR':>10} {'P50 CDR':>10} {'P75 CDR':>10} {'N Vintages':>12}")
print("-" * 60)
for g in grades:
    r = grade_results[g]
    cdrs = r['vintage_cdrs']
    if len(cdrs) >= 3:
        print(f"{g:<6} {r['pool_cdr']:>10.2%} {np.percentile(cdrs, 25):>10.2%} "
              f"{np.percentile(cdrs, 50):>10.2%} {np.percentile(cdrs, 75):>10.2%} {len(cdrs):>12}")
    else:
        print(f"{g:<6} {r['pool_cdr']:>10.2%} {'(too few)':>10} {'':>10} {'':>10} {len(cdrs):>12}")

print(f"\n{'Grade':<6} {'Pool CPR':>10} {'P25 CPR':>10} {'P50 CPR':>10} {'P75 CPR':>10} {'N Vintages':>12}")
print("-" * 60)
for g in grades:
    r = grade_results[g]
    cprs = r['vintage_cprs']
    if len(cprs) >= 3:
        print(f"{g:<6} {r['pool_cpr']:>10.2%} {np.percentile(cprs, 25):>10.2%} "
              f"{np.percentile(cprs, 50):>10.2%} {np.percentile(cprs, 75):>10.2%} {len(cprs):>12}")
    else:
        print(f"{g:<6} {r['pool_cpr']:>10.2%} {'(too few)':>10} {'':>10} {'':>10} {len(cprs):>12}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# CDR by grade — pool vs percentiles
ax = axes[0]
x = np.arange(len(grades))
pool_cdrs = [grade_results[g]['pool_cdr'] for g in grades]
p25_cdrs = [np.percentile(grade_results[g]['vintage_cdrs'], 25)
            if len(grade_results[g]['vintage_cdrs']) >= 3 else np.nan for g in grades]
p75_cdrs = [np.percentile(grade_results[g]['vintage_cdrs'], 75)
            if len(grade_results[g]['vintage_cdrs']) >= 3 else np.nan for g in grades]

ax.bar(x, pool_cdrs, width=0.4, color='#2196F3', alpha=0.7, label='Pool CDR')
ax.scatter(x, p25_cdrs, color='green', marker='^', s=60, zorder=5, label='P25 (upside)')
ax.scatter(x, p75_cdrs, color='red', marker='v', s=60, zorder=5, label='P75 (stress)')
ax.set_xticks(x)
ax.set_xticklabels(grades)
ax.set_title('CDR by Grade: Pool vs Vintage Percentiles')
ax.set_ylabel('CDR')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend()

# CPR by grade
ax = axes[1]
pool_cprs = [grade_results[g]['pool_cpr'] for g in grades]
p25_cprs = [np.percentile(grade_results[g]['vintage_cprs'], 25)
            if len(grade_results[g]['vintage_cprs']) >= 3 else np.nan for g in grades]
p75_cprs = [np.percentile(grade_results[g]['vintage_cprs'], 75)
            if len(grade_results[g]['vintage_cprs']) >= 3 else np.nan for g in grades]

ax.bar(x, pool_cprs, width=0.4, color='#4CAF50', alpha=0.7, label='Pool CPR')
ax.scatter(x, p25_cprs, color='red', marker='v', s=60, zorder=5, label='P25 (stress)')
ax.scatter(x, p75_cprs, color='green', marker='^', s=60, zorder=5, label='P75 (upside)')
ax.set_xticks(x)
ax.set_xticklabels(grades)
ax.set_title('CPR by Grade: Pool vs Vintage Percentiles')
ax.set_ylabel('CPR')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 7. Summary & Recommendations
#
# Run this cell after viewing all the charts above.

# %%
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
CDR APPROACHES:
  Annual vintage:    {len(annual_cdr_series)} data points — stable but thin
  Quarterly vintage: {len(quarterly_cdr_series)} data points — more granular, noisier

CPR APPROACHES:
  Per-loan implied:     {len(loan_cpr):,} data points — rich distribution but
                        many zeros ({(loan_cpr < 0.001).mean():.0%} near zero)
  Annual vintage pool:  {len(annual_cpr_series)} data points — symmetric with CDR, same N
  Quarterly vintage:    {len(quarterly_cpr_series)} data points — richer, noisier

CURRENT MULTIPLICATIVE (±15%):
  Stress CDR: {base_cdr * 1.15:.2%}   Upside CDR: {base_cdr * 0.85:.2%}   Range: {base_cdr * 0.30:.2%}
  Stress CPR: {base_cpr * 0.85:.2%}   Upside CPR: {base_cpr * 1.15:.2%}   Range: {base_cpr * 0.30:.2%}

QUARTERLY VINTAGE PERCENTILES:
  Stress CDR (P75): {np.percentile(quarterly_cdr_series.values, 75):.2%}   Upside CDR (P25): {np.percentile(quarterly_cdr_series.values, 25):.2%}   Range: {np.percentile(quarterly_cdr_series.values, 75) - np.percentile(quarterly_cdr_series.values, 25):.2%}
  Stress CPR (P25): {np.percentile(quarterly_cpr_series.values, 25):.2%}   Upside CPR (P75): {np.percentile(quarterly_cpr_series.values, 75):.2%}   Range: {np.percentile(quarterly_cpr_series.values, 75) - np.percentile(quarterly_cpr_series.values, 25):.2%}
""")

print("""QUESTIONS TO ANSWER FROM THE CHARTS:
  1. Do the vintage CDR distributions look reasonable? Are there outlier
     vintages (very old or very new) that distort percentiles?
  2. Annual vs quarterly — does the extra granularity help or just add noise?
  3. Per-loan implied CPR — is the zero-heavy distribution a problem, or does
     the UPB weighting fix it?
  4. Per-vintage pool CPR — is it too similar to CDR approach (thin N)?
  5. When filtering by grade, do percentiles shift in the expected direction
     (higher-grade pools have tighter CDR ranges)?
""")
