import pandas as pd
import matplotlib.pyplot as plt
import os
import math

# ====== Paths ======
CSV_PATH = "master_survey_results_FULL.csv"
OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ====== Config ======
AGG = "mean"        # "min" / "mean" / "median"
BASELINE = "TL"     # Speedup_vs_TL
TITLE = "Best_per_NUM_PE"  # tên ảnh sẽ là TITLE.png (hoặc TITLE_pX.png nếu nhiều trang)

ROWS_PER_PAGE = 25  # số dòng bảng mỗi ảnh (tùy bạn)
DPI = 200

# ====== Load & clean ======
df = pd.read_csv(CSV_PATH)

required = {"NUM_PE", "Architecture", "Total_Cycles"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

df["NUM_PE"] = pd.to_numeric(df["NUM_PE"], errors="coerce")
df["Total_Cycles"] = pd.to_numeric(df["Total_Cycles"], errors="coerce")
df["Architecture"] = df["Architecture"].astype(str)

df = df.dropna(subset=["NUM_PE", "Architecture", "Total_Cycles"])
df = df[(df["NUM_PE"] > 0) & (df["Total_Cycles"] > 0)]

# ====== Aggregate duplicates ======
if AGG == "min":
    agg_df = df.groupby(["NUM_PE", "Architecture"], as_index=False)["Total_Cycles"].min()
elif AGG == "mean":
    agg_df = df.groupby(["NUM_PE", "Architecture"], as_index=False)["Total_Cycles"].mean()
elif AGG == "median":
    agg_df = df.groupby(["NUM_PE", "Architecture"], as_index=False)["Total_Cycles"].median()
else:
    raise ValueError("AGG must be one of: 'min', 'mean', 'median'")

# ====== Best architecture per NUM_PE ======
best_idx = agg_df.groupby("NUM_PE")["Total_Cycles"].idxmin()
best_df = agg_df.loc[best_idx].copy()

best_df = best_df.rename(columns={
    "Architecture": "Best_Architecture",
    "Total_Cycles": "Total_Cycles"
}).sort_values("NUM_PE")

# ====== Baseline cycles (TL) per NUM_PE ======
tl_df = agg_df[agg_df["Architecture"] == BASELINE][["NUM_PE", "Total_Cycles"]].rename(
    columns={"Total_Cycles": "TL_Total_Cycles"}
)

# ====== Merge + Speedup vs TL ======
out = best_df.merge(tl_df, on="NUM_PE", how="left")
out["Speedup_vs_TL"] = out["TL_Total_Cycles"] / out["Total_Cycles"]

# ====== Final table ======
out = out[["NUM_PE", "Best_Architecture", "Total_Cycles", "Speedup_vs_TL"]].copy()

# format cho dễ đọc
out["NUM_PE"] = out["NUM_PE"].astype(int, errors="ignore")
out["Total_Cycles"] = out["Total_Cycles"].map(lambda x: f"{x:,.0f}")
out["Speedup_vs_TL"] = out["Speedup_vs_TL"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}×")

# ====== Render table to image (multi-page if needed) ======
def render_table_image(df_page: pd.DataFrame, title: str, out_path: str):
    nrows = len(df_page)
    ncols = len(df_page.columns)

    # kích thước figure theo số dòng
    fig_h = 1.2 + 0.35 * (nrows + 1)   # +1 cho header
    fig_w = 10.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    ax.set_title(title, fontsize=14, pad=12)

    table = ax.table(
        cellText=df_page.values,
        colLabels=df_page.columns,
        cellLoc="center",
        colLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    # style header
    for c in range(ncols):
        cell = table[(0, c)]
        cell.set_text_props(weight="bold")
        cell.set_height(cell.get_height() * 1.15)

    # nhấn mạnh Speedup lớn (nếu có)
    # cột Speedup_vs_TL là cột cuối
    speed_col = ncols - 1
    for r in range(1, nrows + 1):  # row 0 là header
        txt = table[(r, speed_col)].get_text().get_text()
        if txt.endswith("×"):
            try:
                val = float(txt[:-1].replace("×", ""))
                if val >= 1.5:
                    table[(r, speed_col)].set_text_props(weight="bold")
            except:
                pass

    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)

# paginate
total_rows = len(out)
pages = math.ceil(total_rows / ROWS_PER_PAGE)

for p in range(pages):
    start = p * ROWS_PER_PAGE
    end = min((p + 1) * ROWS_PER_PAGE, total_rows)
    page_df = out.iloc[start:end]

    if pages == 1:
        out_path = os.path.join(OUT_DIR, f"{TITLE}.png")
        page_title = TITLE
    else:
        out_path = os.path.join(OUT_DIR, f"{TITLE}_p{p+1}.png")
        page_title = f"{TITLE} (page {p+1}/{pages})"

    render_table_image(page_df, page_title, out_path)
