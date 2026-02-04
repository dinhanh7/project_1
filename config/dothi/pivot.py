import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ====== Paths ======
CSV_PATH = "master_survey_results_FULL.csv"
OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ====== Config ======
AGG = "mean"  # "min" / "mean" / "median"
ARCH_ORDER = ["ISC", "TL", "WS", "WSIS"]
NUM_PE_LIST = [3, 6, 12, 24, 48]

METRICS = [
    ("Total_Cycles", "Total_Cycles", "Total_Cycles"),
    ("DMA_ratio", "DMA_ratio", "DMA_ratio = DMA_Cycles / Total_Cycles"),
    ("Throughput_MAC_per_cycle", "Throughput_MAC_per_cycle", "MAC / cycle = Total_MACs / Total_Cycles"),
]

def agg_func_name(agg: str) -> str:
    if agg in ("min", "mean", "median"):
        return agg
    raise ValueError("AGG must be one of: 'min', 'mean', 'median'")

def prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["NUM_PE"] = pd.to_numeric(df["NUM_PE"], errors="coerce")
    df["Architecture"] = df["Architecture"].astype(str)
    df = df.dropna(subset=["NUM_PE", "Architecture"])
    df = df[df["NUM_PE"] > 0]
    return df

def make_metric_value(df: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    df = df.copy()

    if metric_key == "Total_Cycles":
        if "Total_Cycles" not in df.columns:
            raise ValueError("Missing column 'Total_Cycles'")
        df["Total_Cycles"] = pd.to_numeric(df["Total_Cycles"], errors="coerce")
        df = df.dropna(subset=["Total_Cycles"])
        df = df[df["Total_Cycles"] > 0]
        df["metric_value"] = df["Total_Cycles"]

    elif metric_key == "DMA_ratio":
        for c in ["DMA_Cycles", "Total_Cycles"]:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' for DMA_ratio")
        df["DMA_Cycles"] = pd.to_numeric(df["DMA_Cycles"], errors="coerce")
        df["Total_Cycles"] = pd.to_numeric(df["Total_Cycles"], errors="coerce")
        df = df.dropna(subset=["DMA_Cycles", "Total_Cycles"])
        df = df[(df["DMA_Cycles"] >= 0) & (df["Total_Cycles"] > 0)]
        df["metric_value"] = df["DMA_Cycles"] / df["Total_Cycles"]
        # lọc ratio hợp lý (tuỳ bạn bỏ nếu muốn)
        df = df[(df["metric_value"] >= 0) & (df["metric_value"] <= 1)]

    elif metric_key == "Throughput_MAC_per_cycle":
        for c in ["Total_MACs", "Total_Cycles"]:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' for Throughput_MAC_per_cycle")
        df["Total_MACs"] = pd.to_numeric(df["Total_MACs"], errors="coerce")
        df["Total_Cycles"] = pd.to_numeric(df["Total_Cycles"], errors="coerce")
        df = df.dropna(subset=["Total_MACs", "Total_Cycles"])
        df = df[(df["Total_MACs"] >= 0) & (df["Total_Cycles"] > 0)]
        df["metric_value"] = df["Total_MACs"] / df["Total_Cycles"]

    else:
        raise ValueError("Unknown metric_key")

    return df[["NUM_PE", "Architecture", "metric_value"]]

def plot_grouped_bar(df_metric: pd.DataFrame, title: str, ylabel: str):
    # aggregate
    m_agg = (
        df_metric.groupby(["NUM_PE", "Architecture"], as_index=False)["metric_value"]
                 .agg(agg_func_name(AGG))
    )

    # pivot matrix
    pivot = m_agg.pivot_table(index="NUM_PE", columns="Architecture", values="metric_value", aggfunc="first")
    pivot = pivot.reindex(sorted(pivot.index), axis=0)

    # order columns
    cols_all = list(pivot.columns)
    cols = [a for a in ARCH_ORDER if a in cols_all] + [a for a in sorted(cols_all) if a not in ARCH_ORDER]
    pivot = pivot.reindex(cols, axis=1)

    num_pes = pivot.index.tolist()
    archs = pivot.columns.tolist()
    if len(num_pes) == 0 or len(archs) == 0:
        raise ValueError(f"No data to plot for {title}. Check NUM_PE_LIST / Architecture names.")

    # grouped bars
    x = np.arange(len(num_pes))
    n_arch = len(archs)
    width = 0.8 / max(n_arch, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, arch in enumerate(archs):
        y = pivot[arch].values
        ax.bar(x + (i - (n_arch - 1) / 2) * width, y, width=width, label=arch)

    ax.set_title(title)
    ax.set_xlabel("NUM_PE")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(pe)) if float(pe).is_integer() else str(pe) for pe in num_pes])
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(title="Architecture", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    # save always: filename = TITLE
    out_path = os.path.join(OUT_DIR, f"{title}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print("Saved:", out_path)

    plt.show()

# ====== Main ======
df = pd.read_csv(CSV_PATH)
df = prepare_base(df)

# filter NUM_PE
allowed = set(float(x) for x in NUM_PE_LIST)
df = df[df["NUM_PE"].astype(float).isin(allowed)]

# run all metrics
for metric_key, title, ylabel in METRICS:
    TITLE = f"GroupedBar_{title}_vs_NUM_PE"
    df_m = make_metric_value(df, metric_key)
    plot_grouped_bar(df_m, TITLE, ylabel)
