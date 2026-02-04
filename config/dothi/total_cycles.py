import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_PATH = "master_survey_results_FULL.csv"

# ====== Config ======
AUTO_LOG_Y = True          # tự bật log nếu chênh lệch lớn
FORCE_LOG_Y = None         # None = theo AUTO_LOG_Y, True = luôn log, False = không log
LOG_TRIGGER_RATIO = 100    # nếu max/min Total_Cycles >= ratio này thì bật log
AGG = "min"                # nếu trùng (Architecture, NUM_PE): "min" / "mean" / "median"
TITLE = "Total_Cycles vs NUM_PE by Architecture"

# ====== Load & clean ======
df = pd.read_csv(CSV_PATH)

required = {"NUM_PE", "Total_Cycles", "Architecture"}
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
    plot_df = df.groupby(["Architecture", "NUM_PE"], as_index=False)["Total_Cycles"].min()
elif AGG == "mean":
    plot_df = df.groupby(["Architecture", "NUM_PE"], as_index=False)["Total_Cycles"].mean()
elif AGG == "median":
    plot_df = df.groupby(["Architecture", "NUM_PE"], as_index=False)["Total_Cycles"].median()
else:
    raise ValueError("AGG must be one of: 'min', 'mean', 'median'")

plot_df = plot_df.sort_values(["Architecture", "NUM_PE"])

# ====== Decide log scale ======
use_log_y = False
if FORCE_LOG_Y is not None:
    use_log_y = bool(FORCE_LOG_Y)
elif AUTO_LOG_Y:
    y_min = plot_df["Total_Cycles"].min()
    y_max = plot_df["Total_Cycles"].max()
    if y_min > 0 and (y_max / y_min) >= LOG_TRIGGER_RATIO:
        use_log_y = True

# ====== Plot ======
fig, ax = plt.subplots(figsize=(9, 5))

for arch, sub in plot_df.groupby("Architecture", sort=False):
    ax.plot(sub["NUM_PE"], sub["Total_Cycles"], marker="o", linewidth=2, label=arch)

ax.set_title(TITLE)
ax.set_xlabel("NUM_PE")
ax.set_ylabel("Total_Cycles" + (" (log scale)" if use_log_y else ""))

if use_log_y:
    ax.set_yscale("log")

ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
ax.legend(title="Architecture", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()

# ====== Save image: filename exactly equals TITLE ======
out_path = os.path.join(os.getcwd(), f"{TITLE}.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")

print("Saved:", out_path)
plt.show()
