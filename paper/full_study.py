"""Pre-registered analysis of the full study (paper/preregistration.md).

Usage: python paper/full_study.py [--run-id full] [--verdicts paper/adjudication/verdicts_full.csv]

Reads runs/<run_id>/{notes,numbers,claims}.csv, the gate flags per number mention (the same union of Level 1 and
Level 2 flags the dashboard draws), and the blind review verdicts. Writes paper/data/full_results.json,
paper/table_full.tex, paper/table_contrasts.tex, paper/table_gate_rates.tex, paper/table_errtypes.tex and
paper/numbers_full.tex (LaTeX macros).

Unit of analysis: a number mention. A mention is a confirmed error when the gate flagged it and the review
judged it wrong. Intervals resample tables (2,000 resamples, seed 42), because notes on one table are not
independent. The primary set excludes the 6 tables the pilot used, since the gate's rules were revised on them.

Exploratory (not in the pre-registration): if paper/adjudication/error_types_<run_id>.csv exists, each confirmed
error carries a kind (rounding, direction, value, entity), assigned after the review from the reviewers' written
computations. The script then also reports the kinds by writer and strategy, and repeats the strategy contrasts
without rounding slips ("substantive" errors).
"""
import argparse
import json
import sys
from pathlib import Path

import math
import re

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT / "src"))
import dashboard  # noqa: E402

B, SEED = 2000, 42
MODELS = ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"]
MODEL_NAMES = {"claude-opus-5": "Opus 5", "claude-sonnet-5": "Sonnet 5", "claude-haiku-4-5": "Haiku 4.5"}
STRATS = ["S1", "S2", "S3", "S4"]
SIZES = ["small", "medium", "large"]


def pilot_tables():
    return set(pd.read_csv(ROOT / "runs" / "smoke_opus" / "notes.csv").table_id)


def mention_frame(run_dir, verdicts):
    """One row per number mention with gate flag, review verdict and note metadata."""
    R = dashboard.run_payload(run_dir)
    notes = pd.DataFrame(R["notes"])
    rows = []
    for i, x in enumerate(R["numbers"]):
        n = R["notes"][x[0]]
        rows.append({"i": i, "note_id": n["id"], "table_id": n["table"], "model": n["model"], "strategy": n["strategy"],
                     "size": n["size"], "repeat": n["repeat"], "line_idx": x[1], "raw_text": x[2], "position": x[6],
                     "label": x[3], "flag": int(R["flags"][i])})
    m = pd.DataFrame(rows)
    m["checked"] = (m.label != "Skipped").astype(int)
    m["verdict"] = None
    if verdicts is not None:
        v = verdicts.set_index(["note_id", "line_idx", "raw_text", "position"])["final"]
        key = list(zip(m.note_id, m.line_idx, m.raw_text, m.position))
        m["verdict"] = [v.get(k) for k in key]
    m["confirmed"] = ((m.flag == 1) & (m.verdict == "wrong")).astype(int)
    m["etype"] = None
    if verdicts is not None and "error_type" in verdicts:
        e = verdicts.set_index(["note_id", "line_idx", "raw_text", "position"])["error_type"]
        m["etype"] = [e.get(k) for k in key]
    m["substantive"] = ((m.confirmed == 1) & (m.etype != "rounding")).astype(int)
    m["confirmed_or_unclear"] = ((m.flag == 1) & m.verdict.isin(["wrong", "unclear"])).astype(int)
    return m, notes


def per_table(df, num, den):
    g = df.groupby("table_id")[[num, den]].sum()
    return g[num].to_numpy(float), g[den].to_numpy(float), list(g.index)


def boot_rate(df, num, den, rng):
    a, b, _ = per_table(df, num, den)
    if b.sum() == 0:
        return dict(rate=None, lo=None, hi=None, num=int(a.sum()), den=int(b.sum()), tables=len(a))
    idx = rng.integers(0, len(a), size=(B, len(a)))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = a[idx].sum(1) / b[idx].sum(1)
    lo, hi = np.nanpercentile(r, [2.5, 97.5])
    return dict(rate=a.sum() / b.sum(), lo=lo, hi=hi, num=int(a.sum()), den=int(b.sum()), tables=len(a))


def boot_paired(df_a, df_b, num, den, rng):
    """Difference in pooled rates (b minus a) on the same tables, resampling tables."""
    ga = df_a.groupby("table_id")[[num, den]].sum()
    gb = df_b.groupby("table_id")[[num, den]].sum()
    t = sorted(set(ga.index) & set(gb.index))
    if not t or ga.loc[t, den].sum() == 0 or gb.loc[t, den].sum() == 0:
        return dict(diff=None, lo=None, hi=None, p=None, tables=0)
    na, da = ga.loc[t, num].to_numpy(float), ga.loc[t, den].to_numpy(float)
    nb, db = gb.loc[t, num].to_numpy(float), gb.loc[t, den].to_numpy(float)
    idx = rng.integers(0, len(t), size=(B, len(t)))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = nb[idx].sum(1) / db[idx].sum(1) - na[idx].sum(1) / da[idx].sum(1)
    d = d[~np.isnan(d)]
    obs = nb.sum() / db.sum() - na.sum() / da.sum()
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())) if len(d) else None
    lo, hi = np.percentile(d, [2.5, 97.5]) if len(d) else (None, None)
    return dict(diff=obs, lo=lo, hi=hi, p=p, tables=len(t))


def boot_groups(df_a, df_b, num, den, rng):
    """Difference in pooled rates between two disjoint sets of tables (b minus a), resampling each set."""
    a1, a2, _ = per_table(df_a, num, den)
    b1, b2, _ = per_table(df_b, num, den)
    if a2.sum() == 0 or b2.sum() == 0:
        return dict(diff=None, lo=None, hi=None, p=None)
    ia = rng.integers(0, len(a1), size=(B, len(a1)))
    ib = rng.integers(0, len(b1), size=(B, len(b1)))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = b1[ib].sum(1) / b2[ib].sum(1) - a1[ia].sum(1) / a2[ia].sum(1)
    d = d[~np.isnan(d)]
    obs = b1.sum() / b2.sum() - a1.sum() / a2.sum()
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(diff=obs, lo=lo, hi=hi, p=p)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda k: ps[k])
    adj, running = [None] * len(ps), 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[k]))
        adj[k] = running
    return adj


def binom_cdf(k, n, p):
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k + 1))


def clopper(k, n, alpha=0.05):
    """Exact (Clopper-Pearson) interval for k successes in n trials, by bisection on the binomial tail."""
    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if f(mid):
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: 1 - binom_cdf(k - 1, n, p) >= alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p) <= alpha / 2)
    return lower, upper


def kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return None if pe == 1 else (po - pe) / (1 - pe)


# A level followed by a direction word and "from" ("£5,735.62, down from £9,020.33 LW"): gate v3 signs the level
# negative, so a correct cell fails to match. Found by the full-study review; the frozen gate is not changed.
TRAILING_DIRECTION = re.compile(r"^\s*(TW|LW|LY)?\s*,?\s*\(?(down|up|falling|rising|fell|rose|declin\w*|increas\w*|drop\w*|ris\w*|grew|growing)\s+(from|on|versus|vs)", re.I)


def trailing_direction(run_dir, rows):
    """For each flagged item, whether a direction word and 'from' follow the number in its line."""
    lines, out = {}, []
    for x in rows.itertuples():
        if x.note_id not in lines:
            lines[x.note_id] = json.loads((run_dir / "notes" / f"{x.note_id}.json").read_text())["note"].split("\n")
        line = lines[x.note_id][int(x.line_idx)]
        start = int(x.position) + len(str(x.raw_text))
        out.append(bool(TRAILING_DIRECTION.match(line[start:start + 25])))
    return out


def pct(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="full")
    ap.add_argument("--verdicts", default=str(PAPER / "adjudication" / "verdicts_full.csv"))
    ap.add_argument("--out-prefix", default="")
    ap.add_argument("--all-primary", action="store_true", help="testing only: treat every table as primary")
    args = ap.parse_args()
    run_dir = ROOT / "runs" / args.run_id
    vpath = Path(args.verdicts)
    verdicts = pd.read_csv(vpath) if vpath.exists() else None
    tpath = PAPER / "adjudication" / f"error_types_{args.run_id}.csv"
    has_types = verdicts is not None and tpath.exists()
    if has_types:
        verdicts = verdicts.merge(pd.read_csv(tpath), on="item_id", how="left")
    m, notes = mention_frame(run_dir, verdicts)
    raw_notes = pd.read_csv(run_dir / "notes.csv")
    pilot = set() if args.all_primary else pilot_tables()
    m["primary"] = ~m.table_id.isin(pilot)
    rng = np.random.default_rng(SEED)
    out = {"run_id": args.run_id, "adjudicated": verdicts is not None, "pilot_tables": sorted(pilot)}
    out["secondary_note"] = "the 'all' block covers all 60 tables, including the 6 pilot tables"
    ok_notes = raw_notes[~raw_notes.failed]
    out["design"] = {"notes": int(len(raw_notes)), "failed_s2": int(raw_notes.failed.sum()), "tables": int(raw_notes.table_id.nunique()),
                     "primary_tables": int(raw_notes[~raw_notes.table_id.isin(pilot)].table_id.nunique()),
                     "mentions": int(len(m)), "checked": int(m.checked.sum()), "flagged": int(m.flag.sum()),
                     "cost_usd": float(raw_notes.cost_usd.sum() + raw_notes.get("extractor_cost_usd", pd.Series(0)).fillna(0).sum()),
                     "extract_failed": int((raw_notes.extractor.astype(str) == "extract_failed").sum())}
    out["failed_s2_by_model"] = {k: int(v) for k, v in raw_notes[raw_notes.failed].groupby("model").size().items()}
    rn = ok_notes.copy()
    rn["n_binding"] = rn.n_Wrong_entity + rn.n_Wrong_metric
    rn["n_claim_err"] = rn.n_binding + rn.n_Wrong_value
    rn = rn[~rn.table_id.isin(pilot)]
    gate_rates = {}
    for model in MODELS:
        for st in STRATS:
            d = rn[(rn.model == model) & (rn.strategy == st)]
            gate_rates[f"{model}|{st}"] = {"unsupported": boot_rate(d, "n_Unsupported", "n_numbers_checked", rng),
                                           "binding": boot_rate(d, "n_binding", "n_claims_verifiable", rng),
                                           "claim": boot_rate(d, "n_claim_err", "n_claims_verifiable", rng)}
    out["gate_rates_primary"] = gate_rates
    measures = [("gate_flag", "flag"), ("confirmed", "confirmed")] if verdicts is not None else [("gate_flag", "flag")]
    for setname, sub in [("primary", m[m.primary]), ("all", m)]:
        res = {}
        for model in MODELS:
            for st in STRATS:
                d = sub[(sub.model == model) & (sub.strategy == st)]
                cell = {meas: boot_rate(d, col, "checked", rng) for meas, col in measures}
                if verdicts is not None:
                    f = d[d.flag == 1]
                    cell["precision"] = {"confirmed": int(f.confirmed.sum()), "flags": int(len(f)),
                                         "rate": (f.confirmed.sum() / len(f)) if len(f) else None}
                    dn = d.groupby("note_id").confirmed.max()
                    cell["notes_with_error"] = {"k": int(dn.sum()), "n": int(len(dn))}
                res[f"{model}|{st}"] = cell
            dm = sub[sub.model == model]
            res[f"{model}|all"] = {meas: boot_rate(dm, col, "checked", rng) for meas, col in measures}
        out[setname] = res
    if verdicts is not None:
        P = m[m.primary]
        con = {}
        for model in MODELS:
            base = P[(P.model == model) & (P.strategy == "S1")]
            tests = {st: boot_paired(base, P[(P.model == model) & (P.strategy == st)], "confirmed", "checked", rng) for st in ["S2", "S3", "S4"]}
            adj = holm([tests[s]["p"] if tests[s]["p"] is not None else 1.0 for s in ["S2", "S3", "S4"]])
            for s, a in zip(["S2", "S3", "S4"], adj):
                tests[s]["p_holm"] = a
            con[model] = tests
        out["strategy_vs_s1"] = con
        mod = {}
        for st in STRATS:
            base = P[(P.model == "claude-opus-5") & (P.strategy == st)]
            tests = {md: boot_paired(base, P[(P.model == md) & (P.strategy == st)], "confirmed", "checked", rng) for md in MODELS[1:]}
            adj = holm([tests[k]["p"] if tests[k]["p"] is not None else 1.0 for k in MODELS[1:]])
            for k, a in zip(MODELS[1:], adj):
                tests[k]["p_holm"] = a
            mod[st] = tests
        out["model_vs_opus"] = mod
        out["size_s1"] = {md: boot_groups(P[(P.model == md) & (P.strategy == "S1") & (P["size"] == "small")],
                                          P[(P.model == md) & (P.strategy == "S1") & (P["size"] == "large")], "confirmed", "checked", rng) for md in MODELS}
        rel = {}
        for md in MODELS:
            nl = P[P.model == md].groupby(["note_id", "table_id", "strategy", "repeat"]).confirmed.max().reset_index()
            w = nl.pivot_table(index=["table_id", "strategy"], columns="repeat", values="confirmed").dropna()
            rel[md] = {"pairs": int(len(w)), "kappa": kappa(w[1], w[2]) if len(w) and 1 in w and 2 in w else None,
                       "agree": float((w[1] == w[2]).mean()) if len(w) else None,
                       "both": int(((w[1] == 1) & (w[2] == 1)).sum()), "one": int(((w[1] + w[2]) == 1).sum())}
        out["repeat_reliability"] = rel
        key = json.loads((PAPER / "adjudication" / f"key_{args.run_id}.json").read_text())
        kf = pd.DataFrame(key).merge(verdicts[["item_id", "final", "agree"]], on="item_id", how="left")
        out["review"] = {"items": int(len(kf)), "agreement_first_two": float(kf.agree.mean()),
                         "final_counts": kf.final.value_counts().to_dict()}
        miss = {}
        for md in MODELS:
            smp = kf[(kf.kind != "flag") & (kf.model == md) & ~kf.table_id.isin(pilot)]
            k, n = int((smp.final == "wrong").sum()), int(len(smp))
            lo, hi = clopper(k, n) if n else (None, None)
            unflagged = int(P[(P.model == md) & (P.flag == 0) & (P.checked == 1)].shape[0])
            conf = int(P[(P.model == md)].confirmed.sum())
            chk = int(P[(P.model == md)].checked.sum())
            miss[md] = {"sample_wrong": k, "sample_n": n, "miss_rate": k / n if n else None, "lo": lo, "hi": hi,
                        "unflagged_checked": unflagged, "est_missed": (k / n * unflagged) if n else None,
                        "confirmed": conf, "checked": chk,
                        "est_true_rate": ((conf + k / n * unflagged) / chk) if n and chk else None,
                        "est_true_rate_hi": ((conf + hi * unflagged) / chk) if n and chk else None}
            fl = P[(P.model == md) & (P.flag == 1)]
            miss[md]["gate_precision"] = (fl.confirmed.sum() / len(fl)) if len(fl) else None
            miss[md]["gate_precision_ci"] = clopper(int(fl.confirmed.sum()), int(len(fl))) if len(fl) else None
            miss[md]["flags"] = int(len(fl))
            miss[md]["unclear_flags"] = int((fl.verdict == "unclear").sum())
            fa_n = int(((fl.verdict == "correct") | (fl.verdict == "not_a_measurement")).sum())
            miss[md]["false_alarms"] = fa_n
            miss[md]["false_alarm_rate"] = fa_n / (chk - conf - miss[md]["unclear_flags"])
        smp = kf[(kf.kind != "flag") & ~kf.table_id.isin(pilot)]
        k, n = int((smp.final == "wrong").sum()), int(len(smp))
        lo, hi = clopper(k, n)
        out["miss_pooled"] = {"sample_wrong": k, "sample_n": n, "lo": lo, "hi": hi,
                              "not_a_measurement": int((smp.final == "not_a_measurement").sum())}
        fl = P[P.flag == 1]
        out["precision_pooled"] = {"confirmed": int(fl.confirmed.sum()), "flags": int(len(fl)),
                                   "unclear": int((fl.verdict == "unclear").sum()),
                                   "ci": clopper(int(fl.confirmed.sum()), int(len(fl)))}
        out["miss_and_precision"] = miss
        fa = kf[(kf.kind == "flag") & ~kf.table_id.isin(pilot) & (kf.gate_label == "Unsupported")].copy()
        fa["trailing"] = trailing_direction(run_dir, fa)
        out["l1_false_alarms"] = {md: {"false_alarms": int(((fa.model == md) & (fa.final == "correct")).sum()),
                                       "trailing_direction": int(((fa.model == md) & (fa.final == "correct") & fa.trailing).sum()),
                                       "trailing_wrong": int(((fa.model == md) & (fa.final == "wrong") & fa.trailing).sum())}
                                  for md in MODELS}
    if has_types:
        P = m[m.primary]
        kinds = ["rounding", "direction", "value", "entity"]
        et = {}
        for model in MODELS:
            for st in STRATS + ["all"]:
                d = P[(P.model == model) & (P.confirmed == 1) & ((P.strategy == st) if st != "all" else True)]
                et[f"{model}|{st}"] = {k: int((d.etype == k).sum()) for k in kinds}
        out["error_types"] = et
        sub = {}
        for model in MODELS:
            cell = {st: boot_rate(P[(P.model == model) & (P.strategy == st)], "substantive", "checked", rng) for st in STRATS}
            base = P[(P.model == model) & (P.strategy == "S1")]
            tests = {st: boot_paired(base, P[(P.model == model) & (P.strategy == st)], "substantive", "checked", rng) for st in ["S2", "S3", "S4"]}
            adj = holm([tests[k]["p"] if tests[k]["p"] is not None else 1.0 for k in ["S2", "S3", "S4"]])
            for k, a in zip(["S2", "S3", "S4"], adj):
                tests[k]["p_holm"] = a
            sub[model] = {"rates": cell, "vs_s1": tests, "all": boot_rate(P[P.model == model], "substantive", "checked", rng)}
        out["substantive"] = sub
    (PAPER / "data").mkdir(exist_ok=True)
    (PAPER / "data" / f"{args.out_prefix}full_results.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    write_tex(out, args.out_prefix)
    write_macros(out, args.out_prefix)
    print(json.dumps(out["design"], indent=1))


MACRO_MODEL = {"claude-opus-5": "Opus", "claude-sonnet-5": "Sonnet", "claude-haiku-4-5": "Haiku"}
MACRO_STRAT = {"S1": "One", "S2": "Two", "S3": "Three", "S4": "Four"}


def fmt_pct(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{100 * x:.{d}f}\\%"


def fmt_pts(x, d=2):
    """A difference of two rates, in percentage points (no percent sign)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{100 * x:+.{d}f}"


def fmt_p(p):
    """A bootstrap p value; with 2,000 resamples the smallest resolvable value is 0.001."""
    if p is None:
        return "n/a"
    return "$<$0.001" if p < 0.0005 else f"{p:.3f}"


def write_macros(out, prefix):
    """Every full-study figure quoted in the paper, as LaTeX macros."""
    m, d = {}, out["design"]
    m["FullNotes"] = f"{d['notes']:,}"
    m["FullFailed"] = str(d["failed_s2"])
    m["FullTables"] = str(d["tables"])
    m["FullPrimaryTables"] = str(d["primary_tables"])
    m["FullMentions"] = f"{d['mentions']:,}"
    m["FullChecked"] = f"{d['checked']:,}"
    m["FullFlags"] = f"{d['flagged']:,}"
    m["FullCost"] = f"\\${d['cost_usd']:.2f}"
    m["FullExtractFailed"] = str(d["extract_failed"])
    for setname in ["primary"]:
        res = out[setname]
        for model, mm in MACRO_MODEL.items():
            for st, ss in MACRO_STRAT.items():
                c = res[f"{model}|{st}"]
                m[f"Gate{mm}{ss}"] = fmt_pct(c["gate_flag"]["rate"])
                m[f"Checked{mm}{ss}"] = f"{c['gate_flag']['den']:,}"
                if out["adjudicated"]:
                    m[f"Conf{mm}{ss}"] = fmt_pct(c["confirmed"]["rate"])
                    m[f"ConfN{mm}{ss}"] = str(c["confirmed"]["num"])
                    m[f"ConfLo{mm}{ss}"] = fmt_pct(c["confirmed"]["lo"])
                    m[f"ConfHi{mm}{ss}"] = fmt_pct(c["confirmed"]["hi"])
                    m[f"NotesErr{mm}{ss}"] = f"{c['notes_with_error']['k']} of {c['notes_with_error']['n']}"
            a = res[f"{model}|all"]
            m[f"Gate{mm}All"] = fmt_pct(a["gate_flag"]["rate"])
            if out["adjudicated"]:
                m[f"Conf{mm}All"] = fmt_pct(a["confirmed"]["rate"])
                m[f"ConfN{mm}All"] = str(a["confirmed"]["num"])
                m[f"Checked{mm}All"] = f"{a['confirmed']['den']:,}"
    if out["adjudicated"]:
        mp = out["miss_and_precision"]
        for model, mm in MACRO_MODEL.items():
            x = mp[model]
            m[f"Prec{mm}"] = fmt_pct(x["gate_precision"], 1)
            m[f"PrecN{mm}"] = f"{int(round((x['gate_precision'] or 0) * x['flags']))} of {x['flags']}"
            m[f"Miss{mm}"] = f"{x['sample_wrong']} of {x['sample_n']}"
            m[f"MissHi{mm}"] = fmt_pct(x["hi"], 1)
            m[f"TrueRate{mm}"] = fmt_pct(x["est_true_rate"])
            m[f"TrueRateHi{mm}"] = fmt_pct(x["est_true_rate_hi"])
            m[f"Unclear{mm}"] = str(x["unclear_flags"])
            m[f"FARate{mm}"] = fmt_pct(x["false_alarm_rate"], 2)
            r = out["repeat_reliability"][model]
            m[f"Kappa{mm}"] = "n/a" if r["kappa"] is None else f"{r['kappa']:.2f}"
            m[f"Agree{mm}"] = "n/a" if r["agree"] is None else fmt_pct(r["agree"], 0)
            m[f"RecurBoth{mm}"] = str(r["both"])
            m[f"RecurAny{mm}"] = str(r["both"] + r["one"])
            for st, ss in MACRO_STRAT.items():
                if st == "S1":
                    continue
                c = out["strategy_vs_s1"][model][st]
                m[f"Diff{mm}{ss}"] = fmt_pts(c["diff"])
                m[f"DiffLo{mm}{ss}"] = fmt_pts(c["lo"])
                m[f"DiffHi{mm}{ss}"] = fmt_pts(c["hi"])
                m[f"DiffP{mm}{ss}"] = fmt_p(c["p_holm"])
            sz = out["size_s1"][model]
            m[f"Size{mm}"] = fmt_pts(sz["diff"])
            m[f"SizeLo{mm}"] = fmt_pts(sz["lo"])
            m[f"SizeHi{mm}"] = fmt_pts(sz["hi"])
            m[f"SizeP{mm}"] = fmt_p(sz["p"])
        for st, ss in MACRO_STRAT.items():
            for model, mm in list(MACRO_MODEL.items())[1:]:
                c = out["model_vs_opus"][st][model]
                m[f"ModelDiff{mm}{ss}"] = fmt_pts(c["diff"])
                m[f"ModelDiffLo{mm}{ss}"] = fmt_pts(c["lo"])
                m[f"ModelDiffHi{mm}{ss}"] = fmt_pts(c["hi"])
                m[f"ModelDiffP{mm}{ss}"] = fmt_p(c["p_holm"])
        for model, mm in MACRO_MODEL.items():
            a = out["all"][f"{model}|all"]
            m[f"SecConf{mm}"] = fmt_pct(a["confirmed"]["rate"])
            m[f"SecConfN{mm}"] = str(a["confirmed"]["num"])
            m[f"SecChecked{mm}"] = f"{a['confirmed']['den']:,}"
        for model, mm in MACRO_MODEL.items():
            x = out["l1_false_alarms"][model]
            m[f"LoneFA{mm}"] = str(x["false_alarms"])
            m[f"LoneFATrail{mm}"] = str(x["trailing_direction"])
        mpo = out["miss_pooled"]
        m["MissPooled"] = f"{mpo['sample_wrong']} of {mpo['sample_n']}"
        m["MissPooledN"] = str(mpo["sample_n"])
        m["MissPooledHi"] = fmt_pct(mpo["hi"], 1)
        pp = out["precision_pooled"]
        m["PrecPooled"] = fmt_pct(pp["confirmed"] / pp["flags"], 1)
        m["PrecPooledN"] = f"{pp['confirmed']} of {pp['flags']}"
        m["PrecPooledLo"] = fmt_pct(pp["ci"][0], 1)
        m["PrecPooledHi"] = fmt_pct(pp["ci"][1], 1)
        m["UnclearPooled"] = str(pp["unclear"])
        for model, mm in MACRO_MODEL.items():
            lo, hi = out["miss_and_precision"][model]["gate_precision_ci"]
            m[f"PrecLo{mm}"] = fmt_pct(lo, 1)
            m[f"PrecHi{mm}"] = fmt_pct(hi, 1)
        rv = out["review"]
        m["ReviewItems"] = f"{rv['items']:,}"
        m["ReviewAgree"] = fmt_pct(rv["agreement_first_two"], 1)
    if "error_types" in out:
        names = {"rounding": "Round", "direction": "Dir", "value": "Value", "entity": "Entity"}
        for model, mm in MACRO_MODEL.items():
            for st, ss in list(MACRO_STRAT.items()) + [("all", "All")]:
                for k, kk in names.items():
                    m[f"Err{kk}{mm}{ss}"] = str(out["error_types"][f"{model}|{st}"][k])
            sb = out["substantive"][model]
            m[f"Subst{mm}All"] = fmt_pct(sb["all"]["rate"])
            m[f"SubstN{mm}All"] = str(sb["all"]["num"])
            for st, ss in MACRO_STRAT.items():
                m[f"Subst{mm}{ss}"] = fmt_pct(sb["rates"][st]["rate"])
                m[f"SubstN{mm}{ss}"] = str(sb["rates"][st]["num"])
                if st != "S1":
                    c = sb["vs_s1"][st]
                    m[f"SubstDiff{mm}{ss}"] = fmt_pts(c["diff"])
                    m[f"SubstDiffLo{mm}{ss}"] = fmt_pts(c["lo"])
                    m[f"SubstDiffHi{mm}{ss}"] = fmt_pts(c["hi"])
                    m[f"SubstDiffP{mm}{ss}"] = fmt_p(c["p_holm"])
        tot = {k: sum(out["error_types"][f"{md}|all"][k] for md in MODELS) for k in names}
        for k, kk in names.items():
            m[f"Err{kk}Total"] = str(tot[k])
        m["ErrTotal"] = str(sum(tot.values()))
    lines = ["% Generated by paper/full_study.py. Do not edit by hand."] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (PAPER / f"{prefix}numbers_full.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex(out, prefix):
    rows = []
    adj = out["adjudicated"]
    for model in MODELS:
        for st in STRATS:
            c = out["primary"][f"{model}|{st}"]
            g = c["gate_flag"]
            line = [MODEL_NAMES[model] if st == "S1" else "", st, f"{g['den']:,}",
                    f"{pct(g['rate'])} [{pct(g['lo'])}, {pct(g['hi'])}]"]
            if adj:
                cf = c["confirmed"]
                pr = c["precision"]
                line += [f"{cf['num']}", f"{pct(cf['rate'])} [{pct(cf['lo'])}, {pct(cf['hi'])}]",
                         f"{pr['confirmed']}/{pr['flags']}" if pr["flags"] else "0/0",
                         f"{c['notes_with_error']['k']}/{c['notes_with_error']['n']}"]
            rows.append(" & ".join(line) + " \\\\")
        if model != MODELS[-1]:
            rows.append("\\midrule")
    head = ["Writer", "Strategy", "Checked", "Gate flags (\\%)"] + (["Confirmed", "Confirmed (\\%)", "Precision", "Notes with error"] if adj else [])
    tex = ("\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{ll" + "r" * (len(head) - 2) + "}\n\\toprule\n" + " & ".join(head) + " \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n")
    (PAPER / f"{prefix}table_full.tex").write_text(tex, encoding="utf-8")
    if not adj:
        return
    rows = []
    for model in MODELS:
        for st in ["S2", "S3", "S4"]:
            c = out["strategy_vs_s1"][model][st]
            rows.append(" & ".join([MODEL_NAMES[model] if st == "S2" else "", f"{st} minus S1",
                                    f"{pct(c['diff'])} [{pct(c['lo'])}, {pct(c['hi'])}]", fmt_p(c["p_holm"])]) + " \\\\")
        if model != MODELS[-1]:
            rows.append("\\midrule")
    tex = ("\\begin{tabular}{llrr}\n\\toprule\nWriter & Contrast & Difference (points) & $p$ (Holm) \\\\\n\\midrule\n" + "\n".join(rows)
           + "\n\\bottomrule\n\\end{tabular}\n")
    (PAPER / f"{prefix}table_contrasts.tex").write_text(tex, encoding="utf-8")
    rows = []
    for model in MODELS:
        for st in STRATS:
            g = out["gate_rates_primary"][f"{model}|{st}"]
            cell = lambda x: f"{pct(x['rate'])} [{pct(x['lo'])}, {pct(x['hi'])}]"  # noqa: E731
            rows.append(" & ".join([MODEL_NAMES[model] if st == "S1" else "", st,
                                    f"{g['unsupported']['den']:,}", cell(g["unsupported"]), f"{g['binding']['den']:,}",
                                    cell(g["binding"]), cell(g["claim"])]) + " \\\\")
        if model != MODELS[-1]:
            rows.append("\\midrule")
    tex = ("\\begin{tabular}{llrrrrr}\n\\toprule\nWriter & Strategy & Numbers checked & Unsupported (\\%) & Claims verifiable & Binding errors (\\%) & Claim errors (\\%) \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (PAPER / f"{prefix}table_gate_rates.tex").write_text(tex, encoding="utf-8")
    if "error_types" not in out:
        return
    kinds = ["rounding", "direction", "value", "entity"]
    head = ["Writer", "Strategy", "Rounding", "Direction", "Value", "Row or scope", "Total"]
    rows = []
    for model in MODELS:
        for st in STRATS:
            e = out["error_types"][f"{model}|{st}"]
            rows.append(" & ".join([MODEL_NAMES[model] if st == "S1" else "", st]
                                   + [str(e[k]) for k in kinds] + [str(sum(e.values()))]) + " \\\\")
        if model != MODELS[-1]:
            rows.append("\\midrule")
    tex = ("\\begin{tabular}{llrrrrr}\n\\toprule\n" + " & ".join(head) + " \\\\\n\\midrule\n" + "\n".join(rows)
           + "\n\\bottomrule\n\\end{tabular}\n")
    (PAPER / f"{prefix}table_errtypes.tex").write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    main()
