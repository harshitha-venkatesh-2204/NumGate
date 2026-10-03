"""Figures for the paper that are not produced by src/analyze.py.

Usage: python paper/make_figures.py   (run paper/rescore_v1.py and src/analyze.py --run-id smoke_opus first)
Writes paper/figures/fig_v1_v2.png, paper/figures/fig_v1_causes.png, and copies the pilot figures.
"""
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
FIG = PAPER / "figures"
BLUE, ORANGE, INK, INK_2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    ax.grid(axis="x" if ax.get_ylabel() == "" else "y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2)


def fig_v1_v2():
    v1 = pd.read_csv(PAPER / "data" / "v1_v2_notes.csv")
    v2 = pd.read_csv(ROOT / "runs" / "smoke_opus" / "notes.csv")
    strategies = ["S1", "S2", "S3", "S4"]
    g1 = v1.groupby("strategy")
    rate_v1 = (g1.v1_claim_err.sum() / g1.v1_ver.sum()).reindex(strategies)
    g2 = v2.groupby("strategy")
    rate_v2 = ((g2.n_Wrong_entity.sum() + g2.n_Wrong_metric.sum() + g2.n_Wrong_value.sum()) / g2.n_claims_verifiable.sum()).reindex(strategies)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(strategies))
    for i, (vals, label, color) in enumerate([(rate_v1, "Gate v1", BLUE), (rate_v2, "Gate v2", ORANGE)]):
        pos = x + (i - 0.5) * 0.36
        ax.bar(pos, vals, width=0.34, color=color, label=label, edgecolor="white", linewidth=2)
        for p, v in zip(pos, vals):
            ax.annotate(f"{v:.1%}", (p, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, ["S1 Direct", "S2 Compute-\nthen-write", "S3 Write-\nthen-verify", "S4 Facts-\ntemplate"])
    ax.set_ylabel("Claim error rate", color=INK_2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, max(rate_v1.max(), rate_v2.max()) * 1.2)
    style(ax)
    ax.legend(frameon=False)
    ax.set_title("Same 48 Claude Opus 5 notes, scored by two gate versions", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_v1_v2.png", dpi=220)
    plt.close(fig)


def fig_v1_causes():
    flags = pd.read_csv(PAPER / "data" / "v1_flags.csv")
    counts = flags["cause"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.barh(counts.index, counts.values, color=BLUE, height=0.6, edgecolor="white")
    for y, v in enumerate(counts.values):
        ax.annotate(str(v), (v, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_xlabel("Claims flagged by gate v1", color=INK_2)
    ax.set_ylabel("")
    style(ax)
    ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(f"Why gate v1 flagged {len(flags)} claims in Opus 5 notes", x=0.02, ha="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_v1_causes.png", dpi=220)
    plt.close(fig)


def copy_pilot_figures():
    for name in ["error_by_strategy.png", "s3_rounds.png"]:
        shutil.copy(ROOT / "results" / "figures" / name, FIG / f"pilot_{name}")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_v1_v2()
    fig_v1_causes()
    copy_pilot_figures()
    print("Wrote", ", ".join(sorted(p.name for p in FIG.glob("*.png"))))
