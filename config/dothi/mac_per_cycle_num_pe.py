import pandas as pd
import matplotlib.pyplot as plt
import os

# ====== Paths ======
CSV_PATH = "master_survey_results_FULL.csv"  # CSV cùng folder với script
OUT_DIR = "plots"                            # lưu hình vào ./plots

# ====== Config ======
AGG = "mean"  # "min" / "mean" / "median" nếu trùng (Architecture, NUM_PE)
TITLE = "Throughput_MAC_per_cycle_vs_NUM_PE"

# ====== Ensure output folder exists ======
os.makedirs(OUT_DIR, exist_ok=True)

# ====== Load & clean ======
df = pd.read_csv(CSV_PATH)

required = {"NUM_PE", "Architecture", "Total_MACs", "Total_Cycles"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

df["NUM_PE"] = pd.to_numeric(df["NUM_PE"], errors="coerce")
df["Total_MACs"] = pd.to_numeric(df["Total_MACs"], errors="coerce")
df["Total_Cycles"] = pd.to_numeric(df["Total_Cycles"], errors="coerce")
df["Architecture"] = df["Architecture"].astype(str)

df = df.dropna(subset=["NUM_PE", "Total_MACs", "Total_Cycles", "Architecture"])
df = df[(df["NUM_PE"] > 0) & (df["Total_MACs"] >= 0) & (df["Total_Cycles"] > 0)]

# ====== Create metric ======
df["Throughput_MAC_per_cycle"] = df["Total_MACs"] / df["Total_Cycles"]

# ====== Aggregate duplicates ======
if AGG == "min":
    plot_df = df.groupby(["Architecture", "NUM_PE"], as_index=False)["Throughput_MAC_per_cycle"].min()
elif AGG == "mean":
    plot_df = df.groupby(["Architecture", "NUM_PE"], as_index=False)["Throughput_MAC_per_cycle"].mean()
elif AGG == "median":
    plot_df = df.groupby(["Architecture", "NUM_PE"], as_index=False)["Throughput_MAC_per_cycle"].median()
else:
    raise ValueError("AGG must be one of: 'min', 'mean', 'median'")

plot_df = plot_df.sort_values(["Architecture", "NUM_PE"])

# ====== Plot ======
fig, ax = plt.subplots(figsize=(9, 5))

for arch, sub in plot_df.groupby("Architecture", sort=False):
    ax.plot(sub["NUM_PE"], sub["Throughput_MAC_per_cycle"], marker="o", linewidth=2, label=arch)

ax.set_title(TITLE)
ax.set_xlabel("NUM_PE")
ax.set_ylabel("Throughput_MAC_per_cycle = Total_MACs / Total_Cycles")
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
ax.legend(title="Architecture", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()

# ====== Save (always) ======
out_path = os.path.join(OUT_DIR, f"{TITLE}.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print("Saved:", out_path)

plt.show()
