"""Presentation plots for the dairy production hackathon project.

Inputs (written by the statistics export):
    data/stats/summary_by_genetic_group_thi.csv
    data/processed/matched_45cows.csv
Outputs: reports/figures/*.png (300 dpi) and *.pdf (vector), plus the plotted
per-cow tables as CSV.

Run:  python src/make_plots.py          (figures with title, takeaway and footnote)
      python src/make_plots.py --bare   (chart only, for slides -> reports/figures/bare/)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "data" / "stats" / "summary_by_genetic_group_thi.csv"
MATCHED_CSV = ROOT / "data" / "processed" / "matched_45cows.csv"
OUT_DIR = ROOT / "reports" / "figures"
BARE = "--bare" in sys.argv
FIG_DIR = OUT_DIR / "bare" if BARE else OUT_DIR

MILK = "Milk Yield (L/day/cow)"
DMI = "DMI (kg)"
GROUPS = ["Local", "HF50", "HF62.5", "HF75", "HF87.5"]
THIS = ["T0", "T1", "T2"]
THI_LABELS = {"T0": "T0 (THI ≤75)", "T1": "T1 (THI 75–80)", "T2": "T2 (THI ≥80)"}

# one colour per genetic group, identical in all three plots
COLORS = dict(zip(GROUPS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
THI_MARKERS = {"T0": "o", "T1": "s", "T2": "^"}

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "axes.labelsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.frameon": False,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
    "pdf.fonttype": 42,
})


def pct(x, digits=0):
    """Signed percentage with a true minus sign."""
    return f"{x:+.{digits}f}%".replace("-", "−")


def new_figure(title, takeaway, note):
    if BARE:
        fig, ax = plt.subplots(figsize=(10, 5.2))
        fig.subplots_adjust(left=0.09, right=0.76, top=0.96, bottom=0.15)
        ax.tick_params(length=0)
        return fig, ax
    fig, ax = plt.subplots(figsize=(10, 6.4))
    fig.subplots_adjust(left=0.09, right=0.76, top=0.80, bottom=0.25)
    fig.text(0.03, 0.955, title, fontsize=17, fontweight="bold", color=INK, va="top")
    fig.text(0.03, 0.895, takeaway, fontsize=12, color=INK_2, va="top", linespacing=1.35)
    fig.text(0.03, 0.035, note, fontsize=9.5, color=MUTED, va="bottom", linespacing=1.4)
    ax.tick_params(length=0)
    return fig, ax


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300)
    plt.close(fig)
    print(f"saved {stem}.png / .pdf")


def group_handles(marker="o", line=False):
    return [Line2D([], [], color=COLORS[g], marker=marker, markersize=9, markeredgecolor="white",
                   linewidth=2 if line else 0, label=g) for g in GROUPS]


# ---------------------------------------------------------------- load + verify
def load_and_verify():
    summary = pd.read_csv(SUMMARY_CSV)
    matched = pd.read_csv(MATCHED_CSV)

    need_summary = {"Genetic Group", "THI", "variable", "n", "mean"}
    need_matched = {"Animal ID", "Genetic Group", "THI", "Replication No", DMI, MILK}
    assert need_summary <= set(summary.columns), f"summary missing {need_summary - set(summary.columns)}"
    assert need_matched <= set(matched.columns), f"matched missing {need_matched - set(matched.columns)}"
    assert MILK in set(summary["variable"]), f"{MILK!r} not in summary variable column"
    assert set(matched["Genetic Group"]) == set(GROUPS), sorted(set(matched["Genetic Group"]))
    assert set(matched["THI"]) == set(THIS), sorted(set(matched["THI"]))
    assert not matched.duplicated(["Animal ID", "THI", "Replication No"]).any(), "duplicate observation keys"
    assert (matched.groupby("Animal ID")["Genetic Group"].nunique() == 1).all(), "cow in >1 genetic group"

    print("VERIFICATION")
    print(f"  matched rows: {len(matched)}, cows: {matched['Animal ID'].nunique()}")
    print(f"  missing DMI: {int(matched[DMI].isna().sum())}, missing milk yield: {int(matched[MILK].isna().sum())}")
    print("  cows per genetic group:",
          matched.groupby("Genetic Group")["Animal ID"].nunique().reindex(GROUPS).to_dict())
    print("  observations per cow x THI:",
          matched.groupby(["Animal ID", "THI"]).size().value_counts().to_dict(), "(count: n cells)")
    milk_summary = summary[summary["variable"] == MILK]
    print("  summary rows for milk yield:", len(milk_summary),
          "| n per group x THI:", sorted(milk_summary["n"].unique()))

    # the supplied summary must agree with the matched table it was derived from
    recomputed = matched.groupby(["Genetic Group", "THI"])[MILK].mean()
    supplied = milk_summary.set_index(["Genetic Group", "THI"])["mean"]
    assert np.allclose(recomputed.reindex(supplied.index), supplied, atol=1e-3), "summary != matched means"
    print("  supplied group x THI means match the matched table: yes\n")
    return milk_summary, matched


# ---------------------------------------------------------------------- plot 1
def plot_mean_yield_by_thi(milk_summary, n_cows):
    means = milk_summary.pivot(index="Genetic Group", columns="THI", values="mean").reindex(GROUPS)[THIS]
    assert means.notna().all().all(), "missing group x THI mean"
    drop = means["T0"] - means["T2"]
    drop_pct = 100 * drop / means["T0"]
    monotone = bool(((means["T0"] > means["T1"]) & (means["T1"] > means["T2"])).all())
    same_rank = all(list(means[t].sort_values().index) == GROUPS for t in THIS)
    print("PLOT 1  group means (L/cow/day):")
    print(means.round(2).assign(drop_L=drop.round(2), drop_pct=drop_pct.round(1)).to_string(), "\n")

    takeaway = (
        f"Takeaway: mean yield was {'lower at each step from T0 to T2' if monotone else 'lower at T2 than T0'} "
        f"in all five groups ({drop.min():.1f}–{drop.max():.1f} L/cow/day lower at T2);\n"
        + ("groups with a higher HF proportion had higher mean yield at every THI category."
           if same_rank else "group ranking was not the same at every THI category.")
    )
    per_group = n_cows // len(GROUPS)
    note = (
        f"n = {n_cows} cows ({per_group} per genetic group). Each line is one genetic group; each marker is the group mean of "
        f"{int(milk_summary['n'].iloc[0])} observations\n({per_group} cows × 5 repeated observations) in that THI category. "
        "No error bars: repeated observations from the same cow are not independent.\n"
        "Observed association only; this plot does not show that THI caused the difference."
    )
    fig, ax = new_figure("Mean milk yield across THI conditions by genetic group", takeaway, note)
    x = np.arange(len(THIS))
    for g in GROUPS:
        ax.plot(x, means.loc[g], color=COLORS[g], linewidth=2, marker="o", markersize=9,
                markeredgecolor="white", markeredgewidth=1.5, solid_capstyle="round", clip_on=False)
    ax.set_xticks(x, [THI_LABELS[t] for t in THIS])
    ax.set_xlim(-0.2, len(THIS) - 0.8)
    ax.set_ylim(0, np.ceil(means.to_numpy().max()) + 1)
    ax.set_xlabel("THI category")
    ax.set_ylabel("Mean milk yield (L/cow/day)")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=group_handles(line=True), title="Genetic group", loc="upper left",
              bbox_to_anchor=(1.03, 1.0), alignment="left")
    save(fig, "01_mean_milk_yield_by_thi_and_genetic_group")


# ---------------------------------------------------------------------- plot 2
def plot_individual_change(cow_means):
    wide = cow_means.pivot(index=["Animal ID", "Genetic Group"], columns="THI", values=MILK).reset_index()
    invalid = wide["T0"].isna() | (wide["T0"] == 0) | wide["T2"].isna()
    excluded = wide[invalid]
    wide = wide[~invalid].copy()
    wide["change_pct"] = 100 * (wide["T2"] - wide["T0"]) / wide["T0"]
    wide = wide.rename(columns={t: f"mean_milk_{t}" for t in THIS})
    wide.to_csv(OUT_DIR / "02_cow_change_table.csv", index=False, float_format="%.4f")

    counts = wide.groupby("Genetic Group").size().reindex(GROUPS)
    medians = wide.groupby("Genetic Group")["change_pct"].median().reindex(GROUPS)
    n_lower = int((wide["change_pct"] < 0).sum())
    print("PLOT 2  per-cow change T0 -> T2 (%):")
    print(f"  cows excluded (mean_T0 zero or missing, or mean_T2 missing): {len(excluded)}",
          excluded["Animal ID"].tolist())
    print(f"  cows plotted: {len(wide)}; with lower yield at T2: {n_lower}")
    print(wide.groupby("Genetic Group")["change_pct"].agg(["count", "min", "median", "max"])
          .reindex(GROUPS).round(1).to_string(), "\n")

    hf = medians.drop("Local")
    takeaway = (
        f"Takeaway: {n_lower} of {len(wide)} cows had lower mean yield at T2 than at T0. The median change was "
        f"{pct(medians['Local'])} in Local cows,\nwhich had the lowest T0 yields, and between {pct(hf.min())} and "
        f"{pct(hf.max())} in the four HF-cross groups."
    )
    note = (
        f"n = {len(wide)} cows ({', '.join(f'{g}: {counts[g]}' for g in GROUPS)}); {len(excluded)} excluded for a zero or "
        "missing T0 mean. Each point is one cow:\n"
        "100 × (mean T2 − mean T0) / mean T0, each mean taken over that cow's 5 observations. Negative values indicate "
        "lower yield at T2.\nHorizontal jitter is for visibility only. Observed association only, not evidence of causation."
    )
    fig, ax = new_figure("Individual-cow milk yield change from T0 to T2", takeaway, note)
    rng = np.random.default_rng(7)
    ax.axhline(0, color=INK_2, linewidth=1.2, zorder=1)
    for i, g in enumerate(GROUPS):
        y = wide.loc[wide["Genetic Group"] == g, "change_pct"].to_numpy()
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(y)), y, s=85, color=COLORS[g], edgecolor="white",
                   linewidth=1.5, zorder=3)
        ax.hlines(medians[g], i - 0.3, i + 0.3, color=INK, linewidth=3, zorder=4)
        ax.annotate(pct(medians[g]), (i + 0.33, medians[g]), va="center", ha="left", fontsize=11, color=INK)
    ax.set_xticks(range(len(GROUPS)), [f"{g}\n(n = {counts[g]})" for g in GROUPS])
    ax.set_xlim(-0.6, len(GROUPS) - 0.25)
    low = min(wide["change_pct"].min(), 0)
    high = max(wide["change_pct"].max(), 0)
    ax.set_ylim(np.floor(low / 10) * 10 - 5, np.ceil(high / 10) * 10 + 5)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%".replace("-", "−"))
    ax.set_xlabel("Genetic group")
    ax.set_ylabel("Milk yield change, T0 to T2 (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[
        Line2D([], [], color=MUTED, marker="o", markersize=9, markeredgecolor="white", linewidth=0, label="One cow"),
        Line2D([], [], color=INK, linewidth=3, label="Group median"),
        Line2D([], [], color=INK_2, linewidth=1.2, label="No change (0%)"),
    ], loc="upper left", bbox_to_anchor=(1.03, 1.0))
    save(fig, "02_individual_cow_milk_yield_change_t0_to_t2")


# ---------------------------------------------------------------------- plot 3
def plot_dmi_vs_yield(cow_means):
    pts = cow_means.dropna(subset=[DMI, MILK])
    pts.to_csv(OUT_DIR / "03_cow_thi_means_table.csv", index=False, float_format="%.4f")
    n_cows = pts["Animal ID"].nunique()
    group_thi = pts.groupby(["Genetic Group", "THI"])[[DMI, MILK]].mean().unstack("THI").reindex(GROUPS)
    overall = pts.groupby("Genetic Group")[[DMI, MILK]].mean().reindex(GROUPS)
    both_lower = bool(((group_thi[DMI]["T2"] < group_thi[DMI]["T0"])
                       & (group_thi[MILK]["T2"] < group_thi[MILK]["T0"])).all())
    same_order = list(overall[DMI].sort_values().index) == list(overall[MILK].sort_values().index)
    print("PLOT 3  cow x THI means:")
    print(f"  points: {len(pts)} from {n_cows} cows; points per cow:",
          pts.groupby("Animal ID").size().value_counts().to_dict())
    print(group_thi.round(2).to_string(), "\n")

    takeaway = (
        "Takeaway: "
        + ("genetic groups with higher mean DMI also had higher mean milk yield"
           if same_order else "genetic groups differed in both mean DMI and mean milk yield")
        + f" (Local {overall.loc['Local', DMI]:.1f} kg and {overall.loc['Local', MILK]:.1f} L;\n"
        f"HF87.5 {overall.loc['HF87.5', DMI]:.1f} kg and {overall.loc['HF87.5', MILK]:.1f} L)"
        + ("; in every group, mean DMI and mean yield were both lower at T2 than at T0." if both_lower else ".")
    )
    note = (
        f"n = {n_cows} cows, {len(pts)} points. Each point is one cow's mean DMI and mean milk yield over its 5 "
        "observations in one THI category.\n"
        "Each cow contributes up to three points (T0, T1, T2), so points are not independent; no pooled regression "
        "line is drawn.\nObserved association only: DMI and milk yield vary together here, which does not show that "
        "one causes the other."
    )
    fig, ax = new_figure("Dry matter intake versus milk yield", takeaway, note)
    for g in GROUPS:
        for t in THIS:
            sub = pts[(pts["Genetic Group"] == g) & (pts["THI"] == t)]
            ax.scatter(sub[DMI], sub[MILK], s=75, marker=THI_MARKERS[t], color=COLORS[g], edgecolor="white",
                       linewidth=1.2, alpha=0.95, zorder=3)
    ax.set_xlabel("Mean dry matter intake (kg/cow/day)")
    ax.set_ylabel("Mean milk yield (L/cow/day)")
    ax.set_ylim(0, np.ceil(pts[MILK].max()) + 1)
    first = ax.legend(handles=group_handles(), title="Genetic group (colour)", loc="upper left",
                      bbox_to_anchor=(1.03, 1.0), alignment="left")
    ax.add_artist(first)
    ax.legend(handles=[Line2D([], [], color=MUTED, marker=THI_MARKERS[t], markersize=9, markeredgecolor="white",
                              linewidth=0, label=THI_LABELS[t]) for t in THIS],
              title="THI category (shape)", loc="upper left", bbox_to_anchor=(1.03, 0.5), alignment="left")
    save(fig, "03_dmi_vs_milk_yield")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    milk_summary, matched = load_and_verify()
    # one row per cow x THI: mean over that cow's repeated observations
    cow_means = (matched.groupby(["Animal ID", "Genetic Group", "THI"], as_index=False)[[DMI, MILK]].mean())
    plot_mean_yield_by_thi(milk_summary, matched["Animal ID"].nunique())
    plot_individual_change(cow_means)
    plot_dmi_vs_yield(cow_means)


if __name__ == "__main__":
    main()
