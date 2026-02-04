import pandas as pd
import matplotlib.pyplot as plt
import os

# ====== Paths ======
CSV_PATH = "master_survey_results_FULL.csv"  # CSV cùng folder với script
OUT_DIR = "plots"                            # luôn lưu vào ./plots

# ====== Config ======
AGG = "mean"  # "min" / "mean" / "median" nếu trùng (Architecture, NUM_PE)
TITLE = "CacheMissRate_vs_NUM_PE"

# ====== Ensure output folder exists ======
os.makedirs(OUT_DIR, exist_ok=True)

# ====== Load ======
df = pd.read_csv(CSV_PATH)

required = {"NUM_PE", "Architecture", "misses_cpu_core_miss", "references_cpu_core_cache"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

# ====== Clean ======
df["NUM_PE"] = pd.to_numeric(df["NUM_PE"], errors="coerce")
df["misses_cpu_core_miss"] = pd.to_numeric(df["misses_cpu_core_miss"], errors="coerce")
df["references_cpu_core_cache"] = pd.to_numeric(df["references_cpu_core_cache"], errors="coerce")
df["Architecture"] = df["Architecture"].astype(str)

df = df.dropna(subset=["NUM_PE", "Architecture", "misses_cpu_core_miss", "references_cpu_core_cache"])
df = df[(df["NUM_PE"] > 0) & (df["references_cpu_core_cache"] > 0)]
df = df[df["misses_cpu_core_miss"] >= 0]

# ====== Create metric ======
df["MissRate"] = df["misses_cpu_core_miss"] / df["references_cpu_core_cache"]

# (optional) lọc outlier rõ ràng (MissRate thường trong [0,1])
df = df[(df["MissRate"] >= 0) & (df["MissRate"] <= 1)]

# ====== Aggregate duplicates ======
if AGG == "min":
    plot_df = df.groupby(["Architecture", "NUM_PE"], as_index=False)["MissRate"].min()
elif AGG == "mean":
    plot_df = df.groupby(["Architecture", "NUM_PE"], as_index=False)["MissRate"].mean()
elif AGG == "median":
    plot_df = df.groupby(["Architecture", "NUM_PE"], as_index=False)["MissRate"].median()
else:
    raise ValueError("AGG must be one of: 'min', 'mean', 'median'")

plot_df = plot_df.sort_values(["Architecture", "NUM_PE"])

# ====== Plot ======
fig, ax = plt.subplots(figsize=(9, 5))

for arch, sub in plot_df.groupby("Architecture", sort=False):
    ax.plot(sub["NUM_PE"], sub["MissRate"], marker="o", linewidth=2, label=arch)

ax.set_title(TITLE)
ax.set_xlabel("NUM_PE")
ax.set_ylabel("MissRate = misses_cpu_core_miss / references_cpu_core_cache")
ax.set_ylim(0, 1)
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
ax.legend(title="Architecture", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()

# ====== Save (always) ======
out_path = os.path.join(OUT_DIR, f"{TITLE}.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print("Saved:", out_path)

plt.show()
