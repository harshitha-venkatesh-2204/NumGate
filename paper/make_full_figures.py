"""Figures for the full study, from paper/data/full_results.json (written by paper/full_study.py).

Usage: python paper/make_full_figures.py
Writes paper/figures/fig_full_rates.png: per writer, the gate's flag rate and the confirmed error rate for each
strategy, with 95 percent intervals that resample tables. A cell with no confirmed error has no bootstrap interval,
so it shows the exact (Clopper-Pearson) upper bound as a dotted line.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER = Path(__file__).resolve().parent
MODELS = [("claude-opus-5", "Opus 5"), ("claude-sonnet-5", "Sonnet 5"), ("claude-haiku-4-5", "Haiku 4.5")]
STRATS = [("S1", "S1 Direct"), ("S2", "S2 Compute-then-write"), ("S3", "S3 Write-then-verify"), ("S4", "S4 Facts-template")]
# validated categorical slots (blue, orange, aqua, yellow); shapes and labels carry identity too
COLORS = {"S1": "#2a78d6", "S2": "#eb6834", "S3": "#1baf7a", "S4": "#eda100"}
MARKERS = {"S1": "o", "S2": "s", "S3": "^", "S4": "D"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    d = json.loads((PAPER / "data" / "full_results.json").read_text())
    res = d["primary"]
    top = 0
    for model, _ in MODELS:
        for st, _ in STRATS:
            c = res[f"{model}|{st}"]
            for k in ("gate_flag", "confirmed"):
                if c.get(k) and c[k]["hi"] is not None:
                    top = max(top, c[k]["hi"])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True, sharex=True)
    for ax, (model, name) in zip(axes, MODELS):
        for j, (st, label) in enumerate(STRATS):
            y = len(STRATS) - 1 - j
            c = res[f"{model}|{st}"]
            g, cf = c["gate_flag"], c["confirmed"]
            ax.plot([100 * g["lo"], 100 * g["hi"]], [y + 0.14] * 2, color=COLORS[st], lw=1.2, alpha=.45, solid_capstyle="round")
            ax.plot(100 * g["rate"], y + 0.14, marker=MARKERS[st], ms=7, mfc="white", mec=COLORS[st], mew=1.6, ls="none", clip_on=False)
            hi = cf["hi"] if cf["num"] else d.get("exploratory", {}).get("exact_upper", {}).get(f"{model}|{st}", cf["hi"])
            ax.plot([100 * cf["lo"], 100 * hi], [y - 0.14] * 2, color=COLORS[st], lw=2, solid_capstyle="round",
                    ls="-" if cf["num"] else (0, (1, 1.5)))
            ax.plot(100 * cf["rate"], y - 0.14, marker=MARKERS[st], ms=7, color=COLORS[st], mec="white", mew=1.2, ls="none", clip_on=False)
        ax.set_title(name, fontsize=11, color=INK, loc="left")
        ax.set_yticks(range(len(STRATS)))
        ax.set_yticklabels([lab for _, lab in reversed(STRATS)], fontsize=9, color=INK_2)
        ax.set_xlim(-100 * top * 0.03, 100 * top * 1.08)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(axis="both", colors=INK_2, labelsize=9, length=0)
        ax.set_xlabel("percent of checked numbers", fontsize=9, color=INK_2)
    handles = [plt.Line2D([], [], marker="o", ms=7, mfc="white", mec=INK_2, mew=1.6, ls="none", label="gate flags"),
               plt.Line2D([], [], marker="o", ms=7, color=INK_2, ls="none", label="confirmed errors (blind review)")]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.99, 1.04))
    fig.tight_layout()
    out = PAPER / "figures" / "fig_full_rates.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
