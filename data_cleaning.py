"""
Data Cleaning Pipeline
======================
Handles: missing values · outliers · duplicates · label inconsistencies
Outputs: cleaned CSV + a text audit report
"""

import pandas as pd
import numpy as np
from scipy import stats

# ── Load ───────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv("/home/claude/raw_sales_data.csv", parse_dates=["order_date"])
df     = df_raw.copy()
report = []

def log(msg):
    print(msg)
    report.append(msg)

log("=" * 60)
log("  DATA CLEANING AUDIT REPORT")
log("=" * 60)
log(f"\n📦 Raw shape : {df.shape[0]} rows × {df.shape[1]} columns")

# ── Step 1 — Duplicates ────────────────────────────────────────────────────────
n_dup = df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
log(f"\n[1] Duplicates removed  : {n_dup}")
log(f"    Shape after step 1  : {df.shape}")

# ── Step 2 — Label standardisation ────────────────────────────────────────────
df["category"] = df["category"].str.strip().str.title()
log(f"\n[2] Category labels standardised → {df['category'].unique()}")

# ── Step 3 — Missing values ────────────────────────────────────────────────────
missing_before = df.isnull().sum()
log(f"\n[3] Missing values before treatment:\n{missing_before[missing_before > 0].to_string()}")

# customer_age → median imputation
age_median = df["customer_age"].median()
df["customer_age"] = df["customer_age"].fillna(age_median)

# rating → group median by category
df["rating"] = df.groupby("category")["rating"].transform(
    lambda x: x.fillna(x.median())
)

# discount_pct → 0 (no discount)
df["discount_pct"] = df["discount_pct"].fillna(0)

missing_after = df.isnull().sum().sum()
log(f"    Missing values after treatment : {missing_after}")

# ── Step 4 — Outlier detection & capping ──────────────────────────────────────
log("\n[4] Outlier treatment (IQR method):")
outlier_cols = ["unit_price", "quantity", "customer_age"]
outlier_summary = {}

for col in outlier_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower=lower, upper=upper)
    outlier_summary[col] = {"lower": round(lower, 2), "upper": round(upper, 2), "capped": int(n_out)}
    log(f"    {col:15s} → bounds [{lower:.2f}, {upper:.2f}]  |  {n_out} values capped")

# ── Step 5 — Recalculate revenue after cleaning ───────────────────────────────
df["revenue"] = np.round(
    df["quantity"] * df["unit_price"] * (1 - df["discount_pct"] / 100), 2
)
log(f"\n[5] Revenue recalculated after price / quantity correction")

# ── Step 6 — Feature engineering ──────────────────────────────────────────────
df["month"]         = df["order_date"].dt.month_name()
df["day_of_week"]   = df["order_date"].dt.day_name()
df["age_group"]     = pd.cut(
    df["customer_age"],
    bins=[17, 25, 35, 45, 60, 150],
    labels=["18-25", "26-35", "36-45", "46-60", "60+"]
)
log(f"\n[6] New columns added : month, day_of_week, age_group")

# ── Final summary ──────────────────────────────────────────────────────────────
log(f"\n{'=' * 60}")
log(f"  CLEANING COMPLETE")
log(f"  Final shape : {df.shape[0]} rows × {df.shape[1]} columns")
log(f"  Missing     : {df.isnull().sum().sum()} cells")
log(f"{'=' * 60}")

# ── Save outputs ───────────────────────────────────────────────────────────────
df.to_csv("/home/claude/cleaned_sales_data.csv", index=False)
with open("/home/claude/cleaning_report.txt", "w") as f:
    f.write("\n".join(report))

print("\n✅  cleaned_sales_data.csv saved")
print("✅  cleaning_report.txt saved")
