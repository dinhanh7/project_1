import pandas as pd
import matplotlib.pyplot as plt
import os

# ====== Paths ======
CSV_PATH = "master_survey_results_FULL.csv"  # CSV cùng folder với script
OUT_DIR = "plots"                            # luôn lưu vào ./plots

# ====== Config ======
AGG = "mean"  # "min" / "mean" / "median" nếu trùng (Architecture, NUM_PE)
TITLE = "seconds_vs_NUM_PE"

# ====== Ensure output folder exists ======
os.makedirs(OUT_DIR, exist_ok=True)

# ====== Load ======
df = pd.read_csv(CSV_PATH)

# ====== Find seconds column robustly ======
# Ưu tiên đúng tên "seconds", nếu không có thì tìm theo case-insensitive.
if "seconds" not in df.columns:
    lower_map = {c.lower(): c for c in df.columns}
    if "seconds" in lower_map:
        df = df.rename(columns={lower_map["seconds"]: "seconds"})
    else:
        raise ValueError(
            "Missing column 'seconds'. "
            f"Available columns include: {list(df.columns)}"
        )

required = {"NUM_PE", "Architecture", "seconds"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

# ====== Clean ======
df["NUM_PE"] = pd.to_numeric(df["NUM_PE"], errors="coerce")
df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")
df["Architecture"] = df["Architecture"].astype(str)

df = df.dropna(subset=["NUM_PE", "seconds", "Architecture"])
df = df[(df["NUM_PE"] > 0) & (df["seconds"] > 0)]

# ====== Aggregate duplicates ======
if AGG == "min":
    plot_df = df.groupby(["Architecture", "NUM_PE"], as_index=False)["seconds"].min()
elif AGG == "mean":
    plot_df = df.groupby(["Architecture", "NUM_PE"], as_index=False)["seconds"].mean()
elif AGG == "median":
    plot_df = df.groupby(["Architecture", "NUM_PE"], as_index=False)["seconds"].median()
else:
    raise ValueError("AGG must be one of: 'min', 'mean', 'median'")

plot_df = plot_df.sort_values(["Architecture", "NUM_PE"])

# ====== Plot ======
fig, ax = plt.subplots(figsize=(9, 5))

for arch, sub in plot_df.groupby("Architecture", sort=False):
    ax.plot(sub["NUM_PE"], sub["seconds"], marker="o", linewidth=2, label=arch)

ax.set_title(TITLE)
ax.set_xlabel("NUM_PE")
ax.set_ylabel("seconds")
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
ax.legend(title="Architecture", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()

# ====== Save (always) ======
out_path = os.path.join(OUT_DIR, f"{TITLE}.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print("Saved:", out_path)

plt.show()
