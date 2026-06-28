"""
Visualization & Dashboard
=========================
Produces 10 publication-quality charts saved as:
  • individual PNGs
  • one combined dashboard (sales_dashboard.png)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

# ── Theme ──────────────────────────────────────────────────────────────────────
PALETTE   = ["#4361EE", "#3A0CA3", "#7209B7", "#F72585", "#4CC9F0",
             "#06D6A0", "#FFB703", "#FB8500"]
BG_DARK   = "#0D1117"
BG_CARD   = "#161B22"
TEXT_CLR  = "#E6EDF3"
GRID_CLR  = "#21262D"
ACCENT    = "#4361EE"

sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "figure.facecolor":  BG_DARK,
    "axes.facecolor":    BG_CARD,
    "axes.edgecolor":    GRID_CLR,
    "axes.labelcolor":   TEXT_CLR,
    "axes.titlecolor":   TEXT_CLR,
    "xtick.color":       TEXT_CLR,
    "ytick.color":       TEXT_CLR,
    "grid.color":        GRID_CLR,
    "text.color":        TEXT_CLR,
    "legend.facecolor":  BG_CARD,
    "legend.edgecolor":  GRID_CLR,
    "font.family":       "DejaVu Sans",
})

# ── Load cleaned data ──────────────────────────────────────────────────────────
df = pd.read_csv("/home/claude/cleaned_sales_data.csv", parse_dates=["order_date"])
df["month_num"] = df["order_date"].dt.month
month_order = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
day_order   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

print("📊 Generating dashboard...")

# ══════════════════════════════════════════════════════════════════════════════
#   DASHBOARD — 3×3 grid + KPI row
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 26), facecolor=BG_DARK)
fig.suptitle("🛒  Retail Sales Analytics Dashboard",
             fontsize=26, fontweight="bold", color=TEXT_CLR, y=0.98)

gs = gridspec.GridSpec(4, 3, figure=fig,
                       hspace=0.52, wspace=0.38,
                       top=0.94, bottom=0.04, left=0.06, right=0.97)

# ── KPI cards row ──────────────────────────────────────────────────────────────
kpi_ax = fig.add_subplot(gs[0, :])
kpi_ax.set_facecolor(BG_DARK)
kpi_ax.axis("off")

kpis = [
    ("Total Revenue",   f"₹{df['revenue'].sum()/1e6:.2f}M",  "#4361EE"),
    ("Total Orders",    f"{len(df):,}",                       "#7209B7"),
    ("Avg Order Value", f"₹{df['revenue'].mean():.0f}",       "#F72585"),
    ("Avg Rating",      f"{df['rating'].mean():.2f} ⭐",       "#06D6A0"),
    ("Top Category",    df.groupby('category')['revenue'].sum().idxmax(), "#FFB703"),
]
for i, (label, value, color) in enumerate(kpis):
    x = 0.02 + i * 0.197
    rect = FancyBboxPatch((x, 0.05), 0.18, 0.85,
                          boxstyle="round,pad=0.02",
                          facecolor=BG_CARD, edgecolor=color,
                          linewidth=2, transform=kpi_ax.transAxes, zorder=2)
    kpi_ax.add_patch(rect)
    kpi_ax.text(x + 0.09, 0.68, label, ha="center", va="center",
                fontsize=10, color="#8B949E", transform=kpi_ax.transAxes)
    kpi_ax.text(x + 0.09, 0.30, value, ha="center", va="center",
                fontsize=15, fontweight="bold", color=color,
                transform=kpi_ax.transAxes)

# ── Chart 1 — Revenue by Category (bar) ───────────────────────────────────────
ax1 = fig.add_subplot(gs[1, 0])
cat_rev = df.groupby("category")["revenue"].sum().sort_values(ascending=True)
bars = ax1.barh(cat_rev.index, cat_rev.values / 1000,
                color=PALETTE[:len(cat_rev)], edgecolor="none", height=0.6)
for bar, val in zip(bars, cat_rev.values):
    ax1.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
             f"₹{val/1000:.0f}K", va="center", fontsize=8, color=TEXT_CLR)
ax1.set_title("Revenue by Category", fontweight="bold", pad=10)
ax1.set_xlabel("Revenue (₹K)")
ax1.tick_params(labelsize=8)

# ── Chart 2 — Monthly Revenue Trend (line) ────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 1])
monthly = df.groupby("month_num")["revenue"].sum()
ax2.plot(monthly.index, monthly.values / 1000, color=PALETTE[0],
         linewidth=2.5, marker="o", markersize=6, markerfacecolor=PALETTE[3])
ax2.fill_between(monthly.index, monthly.values / 1000,
                 alpha=0.15, color=PALETTE[0])
ax2.set_title("Monthly Revenue Trend", fontweight="bold", pad=10)
ax2.set_xlabel("Month")
ax2.set_ylabel("Revenue (₹K)")
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"], fontsize=8)

# ── Chart 3 — Payment Method Distribution (donut) ─────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
pay_counts = df["payment_method"].value_counts()
wedges, texts, autotexts = ax3.pie(
    pay_counts, labels=pay_counts.index, colors=PALETTE[:4],
    autopct="%1.1f%%", startangle=90,
    wedgeprops=dict(width=0.55, edgecolor=BG_DARK, linewidth=2),
    pctdistance=0.75
)
for t in texts:    t.set_color(TEXT_CLR); t.set_fontsize(8)
for t in autotexts: t.set_color(BG_DARK);  t.set_fontsize(8); t.set_fontweight("bold")
ax3.set_title("Payment Methods", fontweight="bold", pad=10)

# ── Chart 4 — Revenue by Region (grouped bar) ─────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
region_cat = df.groupby(["region", "category"])["revenue"].sum().unstack(fill_value=0)
region_cat_k = region_cat / 1000
x = np.arange(len(region_cat_k.index))
width = 0.15
for i, (col, color) in enumerate(zip(region_cat_k.columns, PALETTE)):
    ax4.bar(x + i * width, region_cat_k[col], width,
            label=col, color=color, edgecolor="none")
ax4.set_xticks(x + width * 2)
ax4.set_xticklabels(region_cat_k.index, fontsize=8)
ax4.set_title("Revenue by Region & Category", fontweight="bold", pad=10)
ax4.set_ylabel("Revenue (₹K)")
ax4.legend(fontsize=6, ncol=2, loc="upper right")

# ── Chart 5 — Rating Distribution (violin + box) ──────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
sns.violinplot(data=df, x="category", y="rating",
               palette=PALETTE[:5], inner="box",
               ax=ax5, linewidth=0.8)
ax5.set_title("Rating Distribution by Category", fontweight="bold", pad=10)
ax5.set_xlabel("")
ax5.set_ylabel("Rating")
ax5.tick_params(axis="x", labelsize=7, rotation=15)

# ── Chart 6 — Heatmap: Revenue by Day × Category ─────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
df["dow"] = pd.Categorical(df["day_of_week"],
                           categories=day_order, ordered=True)
pivot = df.groupby(["dow", "category"])["revenue"].mean().unstack(fill_value=0)
pivot = pivot.reindex(day_order)
sns.heatmap(pivot / 1000, ax=ax6, cmap="YlOrRd",
            linewidths=0.5, linecolor=BG_DARK,
            annot=True, fmt=".0f", annot_kws={"size": 7},
            cbar_kws={"label": "Avg Rev (₹K)"})
ax6.set_title("Avg Revenue: Day × Category", fontweight="bold", pad=10)
ax6.set_ylabel("")
ax6.tick_params(axis="x", labelsize=7, rotation=30)
ax6.tick_params(axis="y", labelsize=7)

# ── Chart 7 — Age Group vs Revenue (box) ──────────────────────────────────────
ax7 = fig.add_subplot(gs[3, 0])
age_order = ["18-25", "26-35", "36-45", "46-60", "60+"]
sns.boxplot(data=df, x="age_group", y="revenue",
            order=age_order, palette=PALETTE[:5],
            ax=ax7, linewidth=0.8, fliersize=2)
ax7.set_title("Revenue by Age Group", fontweight="bold", pad=10)
ax7.set_xlabel("Age Group")
ax7.set_ylabel("Revenue (₹)")
ax7.tick_params(labelsize=8)

# ── Chart 8 — Discount vs Revenue (scatter) ───────────────────────────────────
ax8 = fig.add_subplot(gs[3, 1])
sample = df.sample(min(300, len(df)), random_state=42)
scatter = ax8.scatter(sample["discount_pct"], sample["revenue"],
                      c=sample["rating"], cmap="plasma",
                      s=25, alpha=0.7, edgecolors="none")
cbar = plt.colorbar(scatter, ax=ax8)
cbar.set_label("Rating", fontsize=8, color=TEXT_CLR)
cbar.ax.yaxis.set_tick_params(color=TEXT_CLR, labelsize=7)
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT_CLR)
# Trend line
m, b = np.polyfit(sample["discount_pct"], sample["revenue"], 1)
x_line = np.linspace(sample["discount_pct"].min(), sample["discount_pct"].max(), 100)
ax8.plot(x_line, m * x_line + b, color=PALETTE[3], linewidth=1.5, linestyle="--")
ax8.set_title("Discount % vs Revenue (coloured by Rating)", fontweight="bold", pad=10)
ax8.set_xlabel("Discount %")
ax8.set_ylabel("Revenue (₹)")
ax8.tick_params(labelsize=8)

# ── Chart 9 — Top 5 Days Revenue (bar) ────────────────────────────────────────
ax9 = fig.add_subplot(gs[3, 2])
day_rev = df.groupby("dow")["revenue"].sum().reindex(day_order).dropna()
colors9 = [PALETTE[3] if v == day_rev.max() else PALETTE[0] for v in day_rev.values]
bars9 = ax9.bar(day_rev.index, day_rev.values / 1000,
                color=colors9, edgecolor="none")
ax9.set_title("Revenue by Day of Week", fontweight="bold", pad=10)
ax9.set_ylabel("Revenue (₹K)")
ax9.tick_params(axis="x", rotation=35, labelsize=7)
for bar in bars9:
    ax9.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f"₹{bar.get_height():.0f}K",
             ha="center", va="bottom", fontsize=7, color=TEXT_CLR)

# ── Save ───────────────────────────────────────────────────────────────────────
fig.savefig("/mnt/user-data/outputs/sales_dashboard.png",
            dpi=150, bbox_inches="tight", facecolor=BG_DARK)
print("✅  sales_dashboard.png saved")

# ══════════════════════════════════════════════════════════════════════════════
#   BEFORE / AFTER — Missing values comparison
# ══════════════════════════════════════════════════════════════════════════════
df_raw = pd.read_csv("/home/claude/raw_sales_data.csv")

fig2, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG_DARK)
fig2.suptitle("Data Quality: Before vs After Cleaning",
              fontsize=16, fontweight="bold", color=TEXT_CLR, y=1.02)

for ax, data, title, color in zip(
        axes,
        [df_raw, df],
        ["Raw Dataset", "Cleaned Dataset"],
        [PALETTE[3], PALETTE[5]]):
    miss = data.isnull().sum()
    miss = miss[miss > 0] if miss[miss > 0].any() else pd.Series({"No missing": 0})
    ax.bar(miss.index, miss.values, color=color, edgecolor="none")
    ax.set_title(title, fontweight="bold", color=TEXT_CLR, pad=10)
    ax.set_ylabel("Missing Count")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    for i, v in enumerate(miss.values):
        ax.text(i, v + 0.5, str(v), ha="center", fontsize=9, color=TEXT_CLR)

plt.tight_layout()
fig2.savefig("/mnt/user-data/outputs/before_after_cleaning.png",
             dpi=150, bbox_inches="tight", facecolor=BG_DARK)
print("✅  before_after_cleaning.png saved")

# ══════════════════════════════════════════════════════════════════════════════
#   CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
fig3, ax = plt.subplots(figsize=(8, 6), facecolor=BG_DARK)
corr_cols = ["customer_age", "quantity", "unit_price", "discount_pct", "rating", "revenue"]
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, cmap="coolwarm", center=0,
            annot=True, fmt=".2f", linewidths=0.5, linecolor=BG_DARK,
            annot_kws={"size": 10, "weight": "bold"})
ax.set_title("Feature Correlation Matrix", fontsize=14,
             fontweight="bold", color=TEXT_CLR, pad=12)
ax.tick_params(labelsize=9)
fig3.savefig("/mnt/user-data/outputs/correlation_heatmap.png",
             dpi=150, bbox_inches="tight", facecolor=BG_DARK)
print("✅  correlation_heatmap.png saved")

print("\n🎉  All visualizations complete!")
