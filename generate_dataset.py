"""
Dataset Generator — Simulates a realistic retail sales dataset
with intentional data quality issues for cleaning practice.
"""

import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

# ── Base data ─────────────────────────────────────────────────────────────────
categories   = ["Electronics", "Clothing", "Groceries", "Books", "Sports"]
regions      = ["North", "South", "East", "West"]
payment_methods = ["Credit Card", "Debit Card", "Cash", "UPI"]

data = {
    "order_id":       [f"ORD{str(i).zfill(4)}" for i in range(1, n + 1)],
    "customer_age":   np.random.randint(18, 70, n).astype(float),
    "category":       np.random.choice(categories, n),
    "region":         np.random.choice(regions, n),
    "payment_method": np.random.choice(payment_methods, n),
    "quantity":       np.random.randint(1, 15, n).astype(float),
    "unit_price":     np.round(np.random.uniform(10, 500, n), 2),
    "discount_pct":   np.round(np.random.uniform(0, 30, n), 1),
    "rating":         np.round(np.random.uniform(1, 5, n), 1),
    "order_date":     pd.date_range("2023-01-01", periods=n, freq="17h"),
}

df = pd.DataFrame(data)
df["revenue"] = np.round(
    df["quantity"] * df["unit_price"] * (1 - df["discount_pct"] / 100), 2
)

# ── Inject data quality issues ─────────────────────────────────────────────────
# 1. Missing values (~8%)
for col, frac in [("customer_age", 0.06), ("rating", 0.08), ("discount_pct", 0.05)]:
    idx = df.sample(frac=frac, random_state=42).index
    df.loc[idx, col] = np.nan

# 2. Outliers
df.loc[np.random.choice(df.index, 10), "unit_price"]   = np.random.uniform(5000, 10000, 10)
df.loc[np.random.choice(df.index, 8),  "quantity"]     = np.random.randint(100, 300, 8)
df.loc[np.random.choice(df.index, 5),  "customer_age"] = np.random.choice([-5, 150, 200], 5)

# 3. Duplicates (30 rows)
dup_rows = df.sample(30, random_state=7)
df = pd.concat([df, dup_rows], ignore_index=True)

# 4. Inconsistent category labels
noise_idx = df.sample(15, random_state=9).index
df.loc[noise_idx, "category"] = df.loc[noise_idx, "category"].str.lower()

# 5. Recalculate revenue (now has outlier contamination)
df["revenue"] = np.round(
    df["quantity"] * df["unit_price"] * (1 - df["discount_pct"].fillna(0) / 100), 2
)

df.to_csv("/home/claude/raw_sales_data.csv", index=False)
print(f"Raw dataset saved: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"Duplicates: {df.duplicated().sum()}")
