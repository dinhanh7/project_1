import pandas as pd
import matplotlib.pyplot as plt
import os

# ====== Paths (NO /mnt/data) ======
CSV_PATH = "master_survey_results_FULL.csv"   # file CSV nằm cùng folder với script
OUT_DIR = "plots"                             # lưu tất cả hình vào ./plots

# ====== Config ======
AGG = "min"  # "min" / "mean" / "median"
TITLE = "Breakdown_DMA_Compute_vs_NUM_PE"

# ====== Ensure output folder exists ======
os.makedirs(OUT_DIR, exist_ok=True)

# ====== Load & clean ======
df = pd.read_csv(CSV_PATH)

required = {"NUM_PE", "Architecture", "DMA_Cycles", "Compute_Cycles"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

df["NUM_PE"] = pd.to_numeric(df["NUM_PE"], errors="coerce")
df["DMA_Cycles"] = pd.to_numeric(df["DMA_Cycles"], errors="coerce")
df["Compute_Cycles"] = pd.to_numeric(df["Compute_Cycles"], errors="coerce")
df["Architecture"] = df["Architecture"].astype(str)

df = df.dropna(subset=["NUM_PE", "DMA_Cycles", "Compute_Cycles", "Architecture"])
df = df[(df["NUM_PE"] > 0) & (df["DMA_Cycles"] > 0) & (df["Compute_Cycles"] > 0)]

# ====== Aggregate duplicates ======
if AGG == "min":
    plot_df = df.groupby(["Architecture", "NUM_PE"], as_index=False)[["DMA_Cycles", "Compute_Cycles"]].min()
elif AGG == "mean":
    plot_df = df.groupby(["Architecture", "NUM_PE"], as_index=False)[["DMA_Cycles", "Compute_Cycles"]].mean()
elif AGG == "median":
    plot_df = df.groupby(["Architecture", "NUM_PE"], as_index=False)[["DMA_Cycles", "Compute_Cycles"]].median()
else:
    raise ValueError("AGG must be one of: 'min', 'mean', 'median'")

plot_df = plot_df.sort_values(["Architecture", "NUM_PE"])

# ====== Plot ======
fig, ax = plt.subplots(figsize=(10, 5))

architectures = list(plot_df["Architecture"].unique())
linestyles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
markers = ["o", "s", "^", "D", "v", "P", "X"]

for i, arch in enumerate(architectures):
    sub = plot_df[plot_df["Architecture"] == arch]
    ls = linestyles[i % len(linestyles)]
    mk = markers[i % len(markers)]

    # DMA: đỏ
    ax.plot(sub["NUM_PE"], sub["DMA_Cycles"],
            color="red", linestyle=ls, marker=mk, linewidth=2,
            label=f"{arch} - DMA")
    # Compute: xanh
    ax.plot(sub["NUM_PE"], sub["Compute_Cycles"],
            color="blue", linestyle=ls, marker=mk, linewidth=2,
            label=f"{arch} - Compute")

ax.set_title(TITLE)
ax.set_xlabel("NUM_PE")
ax.set_ylabel("Cycles")
ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Architecture - Type")
plt.tight_layout()

# ====== Save (to one folder) ======
out_path = os.path.join(OUT_DIR, f"{TITLE}.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print("Saved:", out_path)

plt.show()
