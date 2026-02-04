import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# ====== Paths ======
CSV_PATH = "master_survey_results_FULL.csv"  # CSV cùng folder với script
OUT_DIR = "plots"                            # luôn lưu vào ./plots

# ====== Config ======
AGG = "min"   # "min" / "mean" / "median" nếu trùng (Architecture, NUM_PE, BUFFER_SIZE_BYTES)
TITLE = "Total_Cycles_vs_NUM_PE_by_BUFFER_SIZE_BYTES"
AUTO_LOG_Y = True
LOG_TRIGGER_RATIO = 100  # nếu max/min trong 1 subplot >= ratio thì bật log Y

# ====== Ensure output folder exists ======
os.makedirs(OUT_DIR, exist_ok=True)

# ====== Load & clean ======
df = pd.read_csv(CSV_PATH)

required = {"NUM_PE", "Total_Cycles", "Architecture", "BUFFER_SIZE_BYTES"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

df["NUM_PE"] = pd.to_numeric(df["NUM_PE"], errors="coerce")
df["Total_Cycles"] = pd.to_numeric(df["Total_Cycles"], errors="coerce")
df["BUFFER_SIZE_BYTES"] = pd.to_numeric(df["BUFFER_SIZE_BYTES"], errors="coerce")
df["Architecture"] = df["Architecture"].astype(str)

df = df.dropna(subset=["NUM_PE", "Total_Cycles", "Architecture", "BUFFER_SIZE_BYTES"])
df = df[(df["NUM_PE"] > 0) & (df["Total_Cycles"] > 0) & (df["BUFFER_SIZE_BYTES"] > 0)]

# ====== Aggregate duplicates ======
group_cols = ["BUFFER_SIZE_BYTES", "Architecture", "NUM_PE"]
if AGG == "min":
    plot_df = df.groupby(group_cols, as_index=False)["Total_Cycles"].min()
elif AGG == "mean":
    plot_df = df.groupby(group_cols, as_index=False)["Total_Cycles"].mean()
elif AGG == "median":
    plot_df = df.groupby(group_cols, as_index=False)["Total_Cycles"].median()
else:
    raise ValueError("AGG must be one of: 'min', 'mean', 'median'")

plot_df = plot_df.sort_values(["BUFFER_SIZE_BYTES", "Architecture", "NUM_PE"])

# ====== Facet subplots by BUFFER_SIZE_BYTES ======
buffers = sorted(plot_df["BUFFER_SIZE_BYTES"].unique().tolist())
n = len(buffers)

# grid layout
ncols = 3 if n >= 3 else n
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
axes_flat = axes.ravel()

architectures = sorted(plot_df["Architecture"].unique().tolist())

for i, buf in enumerate(buffers):
    ax = axes_flat[i]
    sub_buf = plot_df[plot_df["BUFFER_SIZE_BYTES"] == buf]

    # auto log scale per subplot (optional)
    use_log = False
    if AUTO_LOG_Y:
        y_min = sub_buf["Total_Cycles"].min()
        y_max = sub_buf["Total_Cycles"].max()
        if y_min > 0 and (y_max / y_min) >= LOG_TRIGGER_RATIO:
            use_log = True

    for arch in architectures:
        sub = sub_buf[sub_buf["Architecture"] == arch]
        if sub.empty:
            continue
        ax.plot(sub["NUM_PE"], sub["Total_Cycles"], marker="o", linewidth=2, label=arch)

    ax.set_title(f"BUFFER_SIZE_BYTES = {int(buf)}")
    ax.set_xlabel("NUM_PE")
    ax.set_ylabel("Total_Cycles" + (" (log)" if use_log else ""))
    if use_log:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)

# tắt các ô dư
for j in range(n, len(axes_flat)):
    axes_flat[j].axis("off")

# legend chung
handles, labels = axes_flat[0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, title="Architecture", loc="upper center", ncol=min(len(labels), 6))

fig.suptitle(TITLE, y=1.02)
plt.tight_layout()

# ====== Save (always) ======
out_path = os.path.join(OUT_DIR, f"{TITLE}.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print("Saved:", out_path)

plt.show()
