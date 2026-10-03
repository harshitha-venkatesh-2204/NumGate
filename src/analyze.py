"""Summaries with bootstrap CIs, figures, the HTML report, and paper-ready sentences.

Usage: python src/analyze.py [--run-id smoke]
Reads runs/<run_id>/{notes,numbers,claims}.csv and writes results/summary.csv, results/figures/*.png,
results/report.html and results/paper_numbers.md. With no --run-id it uses "full" if present, else "smoke".
"""
import argparse
import base64
import html
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import dashboard  # noqa: E402
from config import PROMPTS_DIR, RESULTS_DIR, RUNS_DIR, TABLES_DIR, load_config  # noqa: E402

B = 1000  # bootstrap resamples over notes
SIZES = ["small", "medium", "large"]
STRATEGY_NAMES = {"S1": "Direct", "S2": "Compute-then-write", "S3": "Write-then-verify", "S4": "Facts-template"}
# Validated categorical slots (blue, orange, aqua, yellow); aqua and yellow are low contrast,
# so every figure also uses marker shapes and direct labels.
COLORS = {"S1": "#2a78d6", "S2": "#eb6834", "S3": "#1baf7a", "S4": "#eda100"}
MARKERS = {"S1": "o", "S2": "s", "S3": "^", "S4": "D"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# ---------------------------------------------------------------- statistics


def ratio_ci(num, den, rng):
    """Pooled rate sum(num)/sum(den) with a percentile bootstrap CI over notes."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if den.sum() == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(num), size=(B, len(num)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = num[idx].sum(1) / den[idx].sum(1)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return num.sum() / den.sum(), lo, hi


def add_claim_errors(notes):
    notes = notes.copy()
    notes["n_claim_errors"] = notes["n_Wrong_entity"] + notes["n_Wrong_metric"] + notes["n_Wrong_value"]
    notes["n_binding_errors"] = notes["n_Wrong_entity"] + notes["n_Wrong_metric"]  # RQ2: wrong entity, metric or period
    return notes


def summarize_group(g, rng):
    ok = g[~g["failed"]]
    u, ulo, uhi = ratio_ci(ok["n_Unsupported"], ok["n_numbers_checked"], rng)
    c, clo, chi = ratio_ci(ok["n_claim_errors"], ok["n_claims_verifiable"], rng)
    bnd, blo, bhi = ratio_ci(ok["n_binding_errors"], ok["n_claims_verifiable"], rng)
    return {"n_notes": len(g), "n_failed": int(g["failed"].sum()), "n_numbers": int(ok["n_numbers_checked"].sum()),
            "unsupported_rate": u, "unsupported_ci_lo": ulo, "unsupported_ci_hi": uhi,
            "n_claims": int(ok["n_claims_verifiable"].sum()),
            "binding_error_rate": bnd, "binding_ci_lo": blo, "binding_ci_hi": bhi,
            "claim_error_rate": c, "claim_ci_lo": clo, "claim_ci_hi": chi, "n_wrong_entity": int(ok["n_Wrong_entity"].sum()),
            "n_wrong_metric": int(ok["n_Wrong_metric"].sum()), "n_wrong_value": int(ok["n_Wrong_value"].sum()),
            "n_unverifiable": int(ok["n_Unverifiable"].sum()), "format_ok_rate": ok["format_ok"].mean(),
            "mean_cost_usd": g["cost_usd"].mean(), "mean_latency_s": g["latency_s"].mean()}


def build_summary(notes, rng):
    rows = [{"grouping": "overall", **summarize_group(notes, rng)}]
    for keys in [["strategy"], ["model"], ["size"], ["strategy", "size"], ["strategy", "model"]]:
        for vals, g in notes.groupby(keys):
            vals = vals if isinstance(vals, tuple) else (vals,)
            rows.append({"grouping": " x ".join(keys), **dict(zip(keys, vals)), **summarize_group(g, rng)})
    cols = ["grouping", "strategy", "model", "size"]
    out = pd.DataFrame(rows)
    for c in cols:
        if c not in out:
            out[c] = ""
    return out[cols + [c for c in out.columns if c not in cols]].fillna({"strategy": "", "model": "", "size": ""})


def paired_difference(notes, a, b, num_col, den_col, rng):
    """Rate(b) minus rate(a) on conditions matched by table, model and repeat, with a paired bootstrap CI."""
    key = ["table_id", "model", "repeat"]
    ok = notes[~notes["failed"]]
    m = ok[ok["strategy"] == a].merge(ok[ok["strategy"] == b], on=key, suffixes=("_a", "_b"))
    if m.empty:
        return None
    na, da = m[f"{num_col}_a"].to_numpy(float), m[f"{den_col}_a"].to_numpy(float)
    nb, db = m[f"{num_col}_b"].to_numpy(float), m[f"{den_col}_b"].to_numpy(float)
    idx = rng.integers(0, len(m), size=(B, len(m)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = nb[idx].sum(1) / db[idx].sum(1) - na[idx].sum(1) / da[idx].sum(1)
    diff = nb.sum() / db.sum() - na.sum() / da.sum()
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return diff, lo, hi, len(m)


def s3_round_rates(notes, rng):
    """Unsupported rate per S3 revision round; notes that stopped early carry their last round forward."""
    s3 = notes[(notes["strategy"] == "S3") & notes["s3_round_rates"].notna() & (notes["s3_round_rates"] != "")]
    if s3.empty:
        return pd.DataFrame()
    max_rounds = max(len(json.loads(r)) for r in s3["s3_round_rates"]) - 1  # the rounds this run was made with
    rows = []
    for model, g in s3.groupby("model"):
        rates = [json.loads(r) for r in g["s3_round_rates"]]
        rates = [[x if x is not None else 0.0 for x in r] for r in rates]
        padded = np.array([r + [r[-1]] * (max_rounds + 1 - len(r)) for r in rates], float)
        for rnd in range(max_rounds + 1):
            col = padded[:, rnd]
            boots = col[rng.integers(0, len(col), size=(B, len(col)))].mean(1)
            rows.append({"model": model, "round": rnd, "mean_note_rate": col.mean(),
                         "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5), "n_notes": len(col)})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------- figures


def style(ax, ylabel):
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=None))


def headroom(*series):
    """Top of the y axis: the largest CI upper bound plus room for labels."""
    top = max(float(np.nanmax(s)) for s in series if len(s))
    return max(top, 0.01) * 1.18


def spread_labels(values, min_gap):
    """Nudge end-of-line label positions apart so they never overlap."""
    order = np.argsort(values)
    out = np.array(values, float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


CI_PREFIX = {"unsupported_rate": "unsupported", "binding_error_rate": "binding", "claim_error_rate": "claim"}


def err(row, key):
    rate, p = row[key], CI_PREFIX[key]
    lo, hi = row[f"{p}_ci_lo"], row[f"{p}_ci_hi"]
    return [[max(rate - lo, 0)], [max(hi - rate, 0)]]


def fig_by_strategy(summary, path):
    s = summary[summary["grouping"] == "strategy"].set_index("strategy")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(s))
    series = [("unsupported_rate", "Numbers unsupported (level 1)", COLORS["S1"]),
              ("binding_error_rate", "Claims bound to wrong entity or metric (level 2)", COLORS["S2"]),
              ("claim_error_rate", "Claims with any error (level 2)", COLORS["S3"])]
    for i, (key, label, color) in enumerate(series):
        pos = x + (i - 1) * 0.27
        ax.bar(pos, s[key], width=0.25, color=color, label=label, edgecolor="white", linewidth=2)
        for p, (_, row) in zip(pos, s.iterrows()):
            e = err(row, key)
            ax.errorbar(p, row[key], yerr=e, color=INK_2, capsize=3, linewidth=1)
            ax.annotate(f"{row[key]:.1%}", (p, row[key] + e[1][0]), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [f"{k}\n{STRATEGY_NAMES.get(k, k)}" for k in s.index])
    style(ax, "Error rate (95% bootstrap CI)")
    ax.set_ylim(0, headroom(s["unsupported_ci_hi"], s["claim_ci_hi"], s["binding_ci_hi"]))
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_title("Error rate by generation strategy", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_by_size(summary, path):
    s = summary[summary["grouping"] == "strategy x size"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    strategies = sorted(s["strategy"].unique())
    for ax, key, title in [(axes[0], "unsupported_rate", "Numbers unsupported (level 1)"),
                           (axes[1], "binding_error_rate", "Claims bound to wrong entity or metric (level 2)")]:
        ends = []  # (strategy, last x, last y) for direct labels
        for i, strat in enumerate(strategies):
            g = s[s["strategy"] == strat].set_index("size")
            g = g.reindex([z for z in SIZES if z in g.index])
            dodge = (i - (len(strategies) - 1) / 2) * 0.06  # small x offset so CI bars do not overlap
            xs = [SIZES.index(z) + dodge for z in g.index]
            lo_col = CI_PREFIX[key]
            yerr = [np.clip(g[key] - g[f"{lo_col}_ci_lo"], 0, None), np.clip(g[f"{lo_col}_ci_hi"] - g[key], 0, None)]
            ax.errorbar(xs, g[key], yerr=yerr, color=COLORS.get(strat), marker=MARKERS.get(strat), markersize=7,
                        linewidth=2, capsize=2, elinewidth=1, label=f"{strat} {STRATEGY_NAMES.get(strat, '')}")
            ends.append((strat, xs[-1], g[key].iloc[-1]))
        ymax = max(s[f"{CI_PREFIX[key]}_ci_hi"].max(), 0.01)
        label_y = spread_labels([e[2] for e in ends], ymax * 0.06)
        for (strat, x, _), y in zip(ends, label_y):
            ax.annotate(strat, (x, y), xytext=(10, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.set_xticks(range(len(SIZES)), ["small\n5-10 rows", "medium\n20-40 rows", "large\n80-150 rows"])
        ax.set_xlim(-0.3, len(SIZES) - 0.6)
        style(ax, "Error rate (95% bootstrap CI)" if ax is axes[0] else "")
        ax.set_title(title, loc="left", color=INK)
    axes[0].set_ylim(0, headroom(s["unsupported_ci_hi"], s["binding_ci_hi"]))  # shared by both panels
    axes[1].legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_vs_cost(summary, path):
    s = summary[summary["grouping"] == "strategy x model"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, row in s.iterrows():
        ax.errorbar(row["mean_cost_usd"], row["unsupported_rate"], yerr=err(row, "unsupported_rate"),
                    color=COLORS.get(row["strategy"]), marker=MARKERS.get(row["strategy"]), markersize=8,
                    capsize=3, linewidth=1, markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(f"{row['strategy']} {row['model']}", (row["mean_cost_usd"], row["unsupported_rate"]),
                    xytext=(7, 4), textcoords="offset points", fontsize=8, color=INK)
    if (s["mean_cost_usd"] > 0).all() and s["mean_cost_usd"].max() / s["mean_cost_usd"].min() > 20:
        ax.set_xscale("log")
    elif s["mean_cost_usd"].max() == 0:
        ax.set_xlim(-0.001, 0.01)
        ax.text(0.98, 0.95, "All notes cost $0 (mock model)", transform=ax.transAxes, ha="right", color=INK_2)
    else:
        ax.set_xlim(left=0)
    ax.set_xlabel("Mean cost per note (USD)", color=INK_2)
    style(ax, "Numbers unsupported (95% CI)")
    ax.set_ylim(0, headroom(s["unsupported_ci_hi"]))
    ax.set_title("Error rate versus cost per note", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_s3_rounds(rounds, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    if rounds.empty:
        ax.text(0.5, 0.5, "No S3 notes in this run", ha="center", transform=ax.transAxes)
    for i, (model, g) in enumerate(rounds.groupby("model")):
        color = list(COLORS.values())[i % 4]
        ax.errorbar(g["round"], g["mean_note_rate"], yerr=[g["mean_note_rate"] - g["ci_lo"], g["ci_hi"] - g["mean_note_rate"]],
                    color=color, marker=list(MARKERS.values())[i % 4], markersize=7, linewidth=2, capsize=3, label=model)
        ax.annotate(f"{g['mean_note_rate'].iloc[-1]:.1%}", (g["round"].iloc[-1], g["mean_note_rate"].iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xticks([0, 1, 2], ["draft", "after round 1", "after round 2"])
    style(ax, "Mean unsupported rate per note")
    if not rounds.empty:
        ax.set_ylim(0, headroom(rounds["ci_hi"]))
    ax.set_title("S3 write-then-verify: error rate per revision round", loc="left", color=INK)
    if not rounds.empty:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------- paper sentences


def pct(x):
    return "n/a" if pd.isna(x) else f"{x * 100:.1f} percent"


def ci(lo, hi):
    return "n/a" if pd.isna(lo) else f"{lo * 100:.1f} to {hi * 100:.1f}"


def paper_numbers(summary, notes, rounds, settings, rng):
    by = summary[summary["grouping"] == "strategy"].set_index("strategy")
    lines = ["# Headline numbers", ""]
    if settings.get("models") == ["mock"]:
        lines += ["> These numbers come from the mock model (a template writer with planted errors). "
                  "They test the pipeline and are not research results.", ""]
    lines += [f"Run {settings.get('run_id')}: {len(notes)} notes from {notes['table_id'].nunique()} tables, "
              f"models {', '.join(settings.get('models', []))}, {settings.get('repeats')} repeats per condition. "
              "Confidence intervals are 95 percent percentile bootstrap intervals over notes (1000 resamples).", ""]
    if "S1" in by.index:
        r = by.loc["S1"]
        s = (f"Under direct generation (S1), {pct(r['unsupported_rate'])} of cited numbers were unsupported by the table "
             f"(95 percent CI {ci(r['unsupported_ci_lo'], r['unsupported_ci_hi'])}; {r['n_numbers']} numbers in {r['n_notes']} notes)")
        others = []
        for k in ["S2", "S3", "S4"]:
            if k in by.index:
                o = by.loc[k]
                d = paired_difference(notes, "S1", k, "n_Unsupported", "n_numbers_checked", rng)
                dtxt = f", a change of {d[0] * 100:+.1f} points (paired 95 percent CI {d[1] * 100:+.1f} to {d[2] * 100:+.1f})" if d else ""
                others.append(f"{STRATEGY_NAMES[k].lower()} ({k}) gave {pct(o['unsupported_rate'])} "
                              f"(95 percent CI {ci(o['unsupported_ci_lo'], o['unsupported_ci_hi'])}){dtxt}")
        lines += [s + ("; " + "; ".join(others) if others else "") + ".", ""]
        s = (f"At the claim level, {pct(r['binding_error_rate'])} of verifiable S1 claims attached a number to the wrong "
             f"entity, metric or period (95 percent CI {ci(r['binding_ci_lo'], r['binding_ci_hi'])}; {r['n_claims']} claims)")
        others = [f"{k} {pct(by.loc[k, 'binding_error_rate'])} (CI {ci(by.loc[k, 'binding_ci_lo'], by.loc[k, 'binding_ci_hi'])})"
                  for k in ["S2", "S3", "S4"] if k in by.index]
        lines += [s + (", compared with " + ", ".join(others) if others else "") + ".", ""]
        s = (f"Counting wrong values as well, {pct(r['claim_error_rate'])} of verifiable S1 claims had an error "
             f"(95 percent CI {ci(r['claim_ci_lo'], r['claim_ci_hi'])})")
        others = [f"{k} {pct(by.loc[k, 'claim_error_rate'])} (CI {ci(by.loc[k, 'claim_ci_lo'], by.loc[k, 'claim_ci_hi'])})"
                  for k in ["S2", "S3", "S4"] if k in by.index]
        lines += [s + (", compared with " + ", ".join(others) if others else "") + ".", ""]
        errs = r["n_wrong_entity"] + r["n_wrong_metric"] + r["n_wrong_value"]
        if errs:
            lines += [f"Of the S1 claim errors, {r['n_wrong_entity'] / errs:.0%} were wrong-entity, "
                      f"{r['n_wrong_metric'] / errs:.0%} wrong-metric, and {r['n_wrong_value'] / errs:.0%} wrong-value bindings.", ""]
    size = summary[summary["grouping"] == "strategy x size"]
    for k in ["S1", "S2"]:
        g = size[size["strategy"] == k].set_index("size")
        if len(g):
            parts = [f"{z} tables {pct(g.loc[z, 'unsupported_rate'])} (CI {ci(g.loc[z, 'unsupported_ci_lo'], g.loc[z, 'unsupported_ci_hi'])})"
                     for z in SIZES if z in g.index]
            lines += [f"By table size under {k} ({STRATEGY_NAMES[k].lower()}), the unsupported rate was " + ", ".join(parts) + ".", ""]
    if not rounds.empty:
        pooled = rounds.groupby("round")["mean_note_rate"].mean()
        steps = ", ".join(f"{pct(v)} after round {i}" if i else f"{pct(v)} on the draft" for i, v in pooled.items())
        lines += [f"Write-then-verify (S3) moved the mean per-note unsupported rate from {steps}. "
                  "S3 revises against the same level 1 gate that scores it, so its level 2 claim error rate is the fairer comparison.", ""]
    if "S2" in by.index:
        r = by.loc["S2"]
        lines += [f"Compute-then-write (S2) facts scripts failed on both attempts in {r['n_failed']} of {r['n_notes']} notes "
                  f"({r['n_failed'] / r['n_notes']:.1%}); failed notes are excluded from S2 error rates.", ""]
    model_rows = summary[summary["grouping"] == "model"]
    if len(model_rows) > 1:
        parts = [f"{r['model']} {pct(r['unsupported_rate'])} (CI {ci(r['unsupported_ci_lo'], r['unsupported_ci_hi'])})"
                 for _, r in model_rows.iterrows()]
        lines += ["Pooled over strategies, the unsupported rate by model was " + ", ".join(parts) + ".", ""]
    return "\n".join(lines)

# ---------------------------------------------------------------- HTML report


def img_tag(path):
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(path.stem)}">'


def table_html(df, pct_cols=()):
    out = df.copy()
    for c in out.columns:
        if c in pct_cols:
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
        elif out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
    return out.to_html(index=False, border=0, classes="t", escape=True)


def highlighted_note(note_id, run_dir, numbers):
    rec = json.loads((run_dir / "notes" / f"{note_id}.json").read_text(encoding="utf-8"))
    bad = numbers[(numbers["note_id"] == note_id) & (numbers["label"] == "Unsupported")]
    lines = [l for l in rec["note"].splitlines() if l.strip()]
    out = []
    for i, line in enumerate(lines):
        spans = []  # (start, end) of each flagged number, found by its recorded position
        for _, r in bad[bad["line_idx"] == i].iterrows():
            raw, pos = str(r["raw_text"]), int(r["position"]) if pd.notna(r.get("position")) else -1
            if line[pos:pos + len(raw)] != raw:
                pos = line.find(raw)
            if pos >= 0:
                spans.append((pos, pos + len(raw)))
        text, cur = "", 0
        for a, b in sorted(spans):
            if a < cur:
                continue
            text += html.escape(line[cur:a]) + f'<mark title="Unsupported">{html.escape(line[a:b])}</mark>'
            cur = b
        out.append(text + html.escape(line[cur:]))
    return "<br>".join(out)


def report_html(summary, notes, numbers, settings, cfg, figs, paper_md, run_dir):
    idx = pd.read_csv(TABLES_DIR / "index.csv")
    counts = idx.groupby(["size", "entity_type"]).size().rename("tables").reset_index()
    model_rows = pd.DataFrame([{"name": m, "provider": cfg["models"][m]["provider"], "model_id": cfg["models"][m]["model_id"],
                                "price_in_per_M": cfg["models"][m]["price_in"], "price_out_per_M": cfg["models"][m]["price_out"]}
                               for m in settings.get("models", []) if m in cfg["models"]])
    rate_cols = ["unsupported_rate", "unsupported_ci_lo", "unsupported_ci_hi", "binding_error_rate", "binding_ci_lo",
                 "binding_ci_hi", "claim_error_rate", "claim_ci_lo", "claim_ci_hi", "format_ok_rate"]
    show = ["strategy", "model", "size", "n_notes", "n_failed", "n_numbers", "unsupported_rate", "unsupported_ci_lo",
            "unsupported_ci_hi", "n_claims", "binding_error_rate", "binding_ci_lo", "binding_ci_hi",
            "claim_error_rate", "claim_ci_lo", "claim_ci_hi", "format_ok_rate",
            "mean_cost_usd"]
    sections = []
    for grouping, title in [("strategy", "By strategy"), ("model", "By model"), ("size", "By table size"),
                            ("strategy x size", "Strategy by table size"), ("strategy x model", "Strategy by model")]:
        s = summary[summary["grouping"] == grouping]
        keep = [c for c in show if c not in {"strategy", "model", "size"} or c in grouping.split(" x ")]
        sections.append(f"<h3>{title}</h3>{table_html(s[keep], rate_cols)}")
    examples = []
    for strat in sorted(notes["strategy"].unique()):
        g = notes[(notes["strategy"] == strat) & ~notes["failed"]].sort_values("n_Unsupported", ascending=False)
        if len(g):
            nid = g.iloc[0]["note_id"]
            examples.append(f"<h3>{strat} {STRATEGY_NAMES.get(strat, '')}: <code>{html.escape(nid)}</code></h3>"
                            f"<p class='note'>{highlighted_note(nid, run_dir, numbers)}</p>")
    prompts = "".join(f"<details><summary>{p.name}</summary><pre>{html.escape(p.read_text(encoding='utf-8'))}</pre></details>"
                      for p in sorted(PROMPTS_DIR.glob("*.txt")))
    mock_banner = ("<p class='banner'>This run used the mock model, which writes template notes with planted errors. "
                   "The numbers test the pipeline and are not research results.</p>" if settings.get("models") == ["mock"] else "")
    paper = "".join(f"<p>{html.escape(l)}</p>" for l in paper_md.splitlines() if l.strip() and not l.startswith(("#", ">")))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NumGate Report</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df; --mark:#fde2b3; --panel:#f4f3f0; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --line:#383835; --mark:#7a5200; --panel:#242423; }} }}
:root[data-theme="dark"] {{ --bg:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --line:#383835; --mark:#7a5200; --panel:#242423; }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; margin:0; }}
main {{ max-width:1000px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ margin-top:40px; border-bottom:1px solid var(--line); padding-bottom:6px; }}
.sub, .muted {{ color:var(--ink2); }}
img {{ max-width:100%; height:auto; background:#fcfcfb; border:1px solid var(--line); border-radius:6px; margin:8px 0; }}
.t {{ border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; display:block; overflow-x:auto; }}
.t th, .t td {{ padding:4px 8px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
.t th {{ color:var(--ink2); font-weight:600; }}
.note {{ background:var(--panel); padding:12px; border-radius:6px; font-size:14px; }}
mark {{ background:var(--mark); color:var(--ink); padding:0 2px; border-radius:2px; }}
pre {{ background:var(--panel); padding:12px; border-radius:6px; white-space:pre-wrap; font-size:13px; }}
.banner {{ background:var(--panel); border-left:4px solid #eda100; padding:10px 12px; }}
details summary {{ cursor:pointer; padding:4px 0; }}
</style></head><body><main>
<h1>NumGate report</h1>
<p class="sub">Number faithfulness of LLM-written insight notes. Run <code>{html.escape(str(settings.get('run_id')))}</code>, {len(notes)} notes.</p>
{mock_banner}
<h2>Headline numbers</h2>{paper}
<h2>Figures</h2>
{''.join(img_tag(f) for f in figs)}
<h2>Summary tables</h2>
<p class="muted">Rates pool numbers (or claims) over notes; CIs are 95 percent percentile bootstrap intervals with 1000 resamples over notes. Failed notes (S2 scripts that failed twice) are excluded from rates.</p>
{''.join(sections)}
<h2>Methods</h2>
<p><b>Data.</b> UCI Online Retail II, cleaned (cancelled invoices, non-positive quantity or price, missing country and non-product stock codes removed), aggregated to ISO weeks. Each table lists entities with this week, last week and same-week-last-year revenue, units and orders, plus WoW, YoY, share and rank computed in pandas. {len(idx)} tables were built (seed {cfg['seed']}):</p>
{table_html(counts)}
<p><b>Run settings.</b> Strategies {', '.join(settings.get('strategies', []))}; {settings.get('tables_per_size')} tables per size; {settings.get('repeats')} repeats per condition; temperature 0 where the model accepts it; max {cfg['max_tokens']} output tokens; S3 at most {cfg['s3_max_rounds']} revision rounds; S2 scripts time out after {cfg['s2_timeout_seconds']} seconds with one retry. Budget cap ${settings.get('budget_usd')}.</p>
{table_html(model_rows) if len(model_rows) else ''}
<p><b>Strategies.</b> S1 Direct: the table as markdown, one call. S2 Compute-then-write: the model writes a pandas script that computes up to 25 facts, the script runs in a subprocess, and a second call writes the note from the facts only. S3 Write-then-verify: the S1 draft is gated and flagged numbers are sent back for correction, up to {cfg['s3_max_rounds']} rounds. S4 Facts-template: a deterministic facts packet narrated with every number copied verbatim.</p>
<p><b>Gate.</b> Level 1 labels each number Supported_cell, Supported_derived (fixed operation library), Unsupported, or Skipped (years, week labels, counting words, digits inside entity names), using a rounding-aware match with 0.5 percent tolerance for k, M and B values. Level 2 extracts (entity, metric, period, value, direction) claims ({cfg.get('extractor_model')} for real models, a rule-based extractor for the mock) and labels them Correct, Wrong_entity, Wrong_metric, Wrong_value or Unverifiable. Full rules: docs/method.md.</p>
<h3>Prompts</h3>{prompts}
<h2>Example notes</h2><p class="muted">The note with the most unsupported numbers per strategy; highlighted numbers were flagged by level 1.</p>
{''.join(examples)}
</main></body></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default=None)
    args = p.parse_args()
    run_id = args.run_id or ("full" if (RUNS_DIR / "full" / "notes.csv").exists() else "smoke")
    run_dir = RUNS_DIR / run_id
    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])
    settings = json.loads((run_dir / "run_settings.json").read_text(encoding="utf-8"))
    notes = add_claim_errors(pd.read_csv(run_dir / "notes.csv"))
    numbers = pd.read_csv(run_dir / "numbers.csv")

    summary = build_summary(notes, rng)
    rounds = s3_round_rates(notes, rng)
    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    if not rounds.empty:
        rounds.to_csv(RESULTS_DIR / "s3_rounds.csv", index=False)

    figs = [fig_dir / n for n in ["error_by_strategy.png", "error_by_size.png", "error_vs_cost.png", "s3_rounds.png"]]
    fig_by_strategy(summary, figs[0])
    fig_by_size(summary, figs[1])
    fig_vs_cost(summary, figs[2])
    fig_s3_rounds(rounds, figs[3])

    paper_md = paper_numbers(summary, notes, rounds, settings, rng)
    (RESULTS_DIR / "paper_numbers.md").write_text(paper_md, encoding="utf-8")
    (RESULTS_DIR / "report.html").write_text(report_html(summary, notes, numbers, settings, cfg, figs, paper_md, run_dir),
                                             encoding="utf-8")
    print(f"Analyzed run {run_id}: {len(notes)} notes")
    print(summary[summary["grouping"] == "strategy"][["strategy", "n_notes", "unsupported_rate", "unsupported_ci_lo",
                                                      "unsupported_ci_hi", "binding_error_rate", "claim_error_rate",
                                                      "claim_ci_lo", "claim_ci_hi"]].round(4).to_string(index=False))
    dash_path, _ = dashboard.write_dashboard()
    print("Wrote results/summary.csv, results/figures/*.png, results/report.html, results/paper_numbers.md, "
          f"results/{dash_path.name}")


if __name__ == "__main__":
    main()
