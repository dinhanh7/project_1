import pandas as pd
import matplotlib.pyplot as plt
import os

# ====== Paths ======
CSV_PATH = "master_survey_results_FULL.csv"  # CSV cùng folder với script
OUT_DIR = "plots"                            # lưu hình vào ./plots

# ====== Config ======
AGG = "mean"  # "min" / "mean" / "median" nếu trùng (Architecture, NUM_PE)
TITLE = "Speedup_vs_NUM_PE_baseline"
BASELINE_PREFERRED = "TL"   # ưu tiên baseline là "TL" nếu có đủ dữ liệu
BASELINE_FALLBACK = "auto"  # "auto" => chọn architecture có nhiều NUM_PE nhất

# ====== Ensure output folder exists ======
os.makedirs(OUT_DIR, exist_ok=True)

# ====== Load & clean ======
df = pd.read_csv(CSV_PATH)

required = {"NUM_PE", "Architecture", "Total_Cycles"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

df["NUM_PE"] = pd.to_numeric(df["NUM_PE"], errors="coerce")
df["Total_Cycles"] = pd.to_numeric(df["Total_Cycles"], errors="coerce")
df["Architecture"] = df["Architecture"].astype(str)

df = df.dropna(subset=["NUM_PE", "Total_Cycles", "Architecture"])
df = df[(df["NUM_PE"] > 0) & (df["Total_Cycles"] > 0)]

# ====== Aggregate duplicates ======
if AGG == "min":
    agg_df = df.groupby(["Architecture", "NUM_PE"], as_index=False)["Total_Cycles"].min()
elif AGG == "mean":
    agg_df = df.groupby(["Architecture", "NUM_PE"], as_index=False)["Total_Cycles"].mean()
elif AGG == "median":
    agg_df = df.groupby(["Architecture", "NUM_PE"], as_index=False)["Total_Cycles"].median()
else:
    raise ValueError("AGG must be one of: 'min', 'mean', 'median'")

# ====== Choose baseline ======
archs = sorted(agg_df["Architecture"].unique().tolist())
num_pe_sets = {a: set(agg_df.loc[agg_df["Architecture"] == a, "NUM_PE"].tolist()) for a in archs}

baseline = None
if BASELINE_PREFERRED in archs:
    baseline = BASELINE_PREFERRED
elif BASELINE_FALLBACK == "auto":
    # chọn architecture có nhiều điểm NUM_PE nhất
    baseline = max(archs, key=lambda a: len(num_pe_sets[a]))
else:
    baseline = archs[0]  # fallback đơn giản

# baseline series: Total_Cycles theo NUM_PE
base_df = (
    agg_df[agg_df["Architecture"] == baseline][["NUM_PE", "Total_Cycles"]]
    .rename(columns={"Total_Cycles": "Baseline_Cycles"})
)

# ====== Compute speedup for other architectures ======
others = agg_df[agg_df["Architecture"] != baseline].copy()
merged = others.merge(base_df, on="NUM_PE", how="inner")  # chỉ giữ NUM_PE có baseline
merged["Speedup"] = merged["Baseline_Cycles"] / merged["Total_Cycles"]

# ====== Plot ======
fig, ax = plt.subplots(figsize=(9, 5))

# sort for nice lines
merged = merged.sort_values(["Architecture", "NUM_PE"])

for arch, sub in merged.groupby("Architecture", sort=False):
    ax.plot(sub["NUM_PE"], sub["Speedup"], marker="o", linewidth=2, label=arch)

ax.set_title(f"{TITLE} (baseline={baseline})")
ax.set_xlabel("NUM_PE")
ax.set_ylabel("Speedup = Total_Cycles(baseline) / Total_Cycles(method)")
ax.axhline(1.0, linestyle="--", linewidth=1)  # speedup=1
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
ax.legend(title="Architecture (excluding baseline)", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()

# ====== Save (always) ======
out_path = os.path.join(OUT_DIR, f"{TITLE}.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print("Baseline:", baseline)
print("Saved:", out_path)

plt.show()
