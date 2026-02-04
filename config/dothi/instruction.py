import pandas as pd
import matplotlib.pyplot as plt
import os

# ====== Paths ======
CSV_PATH = "master_survey_results_FULL.csv"  # CSV cùng folder với script
OUT_DIR = "plots"                            # luôn lưu vào ./plots

# ====== Config ======
AGG = "mean"  # "min" / "mean" / "median" nếu trùng (Architecture, NUM_PE)
TITLE = "IPC_vs_NUM_PE"

# ====== Ensure output folder exists ======
os.makedirs(OUT_DIR, exist_ok=True)

# ====== Load ======
df = pd.read_csv(CSV_PATH)

required = {"NUM_PE", "Architecture", "instructions_cpu_core", "cycles_cpu_core"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

# ====== Clean ======
df["NUM_PE"] = pd.to_numeric(df["NUM_PE"], errors="coerce")
df["instructions_cpu_core"] = pd.to_numeric(df["instructions_cpu_core"], errors="coerce")
df["cycles_cpu_core"] = pd.to_numeric(df["cycles_cpu_core"], errors="coerce")
df["Architecture"] = df["Architecture"].astype(str)

df = df.dropna(subset=["NUM_PE", "Architecture", "instructions_cpu_core", "cycles_cpu_core"])
df = df[(df["NUM_PE"] > 0) & (df["cycles_cpu_core"] > 0)]
df = df[df["instructions_cpu_core"] >= 0]

# ====== Create metric ======
df["IPC"] = df["instructions_cpu_core"] / df["cycles_cpu_core"]

# (optional) lọc outlier vô lý (IPC thường không quá lớn; tuỳ CPU, để rộng một chút)
df = df[(df["IPC"] >= 0) & (df["IPC"] <= 10)]

# ====== Aggregate duplicates ======
if AGG == "min":
    plot_df = df.groupby(["Architecture", "NUM_PE"], as_index=False)["IPC"].min()
elif AGG == "mean":
    plot_df = df.groupby(["Architecture", "NUM_PE"], as_index=False)["IPC"].mean()
elif AGG == "median":
    plot_df = df.groupby(["Architecture", "NUM_PE"], as_index=False)["IPC"].median()
else:
    raise ValueError("AGG must be one of: 'min', 'mean', 'median'")

plot_df = plot_df.sort_values(["Architecture", "NUM_PE"])

# ====== Plot ======
fig, ax = plt.subplots(figsize=(9, 5))

for arch, sub in plot_df.groupby("Architecture", sort=False):
    ax.plot(sub["NUM_PE"], sub["IPC"], marker="o", linewidth=2, label=arch)

ax.set_title(TITLE)
ax.set_xlabel("NUM_PE")
ax.set_ylabel("IPC = instructions_cpu_core / cycles_cpu_core")
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
ax.legend(title="Architecture", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()

# ====== Save (always) ======
out_path = os.path.join(OUT_DIR, f"{TITLE}.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print("Saved:", out_path)

plt.show()
