"""Level 2 (claim binding) stress test. Read-only with respect to the project.

Run:  cd <project root> && .venv/bin/python <this file>
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # project root
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
import number_gate as ng  # noqa: E402
from claims_data import CLAIMS, NOTES, PROBES  # noqa: E402

TABLES = {}


def table(tid):
    if tid not in TABLES:
        TABLES[tid] = pd.read_csv(ROOT / "data" / "tables" / f"{tid}.csv")
    return TABLES[tid]

# ------------------------------------------------------------------ independent ground truth (pandas only)


def gt_value(t, ent, col):
    """True value of (entity, virtual column), computed straight from the CSV, not from number_gate."""
    if ent is None:
        if col == "Top10_Revenue":
            return t.sort_values("Revenue_TW", ascending=False).Revenue_TW.iloc[:10].sum()
        if col == "Top10_Share":
            r = t.sort_values("Revenue_TW", ascending=False).Revenue_TW
            return r.iloc[:10].sum() / r.sum() * 100
        return np.nan
    if ent == "TOTAL":
        s = t[["Revenue_TW", "Revenue_LW", "Revenue_LY", "Units_TW", "Units_LW", "Orders_TW", "Orders_LW"]].sum()
        m = {"Revenue_WoW_Pct": (s.Revenue_TW / s.Revenue_LW - 1) * 100,
             "Revenue_YoY_Pct": (s.Revenue_TW / s.Revenue_LY - 1) * 100,
             "Revenue_WoW_Abs": s.Revenue_TW - s.Revenue_LW, "Revenue_YoY_Abs": s.Revenue_TW - s.Revenue_LY,
             "Units_WoW_Pct": (s.Units_TW / s.Units_LW - 1) * 100, "Orders_WoW_Pct": (s.Orders_TW / s.Orders_LW - 1) * 100}
        return m.get(col, s.get(col, np.nan))
    r = t[t.Entity == ent].iloc[0]

    def pct(a, b):
        return (a - b) / b * 100 if b else np.nan
    m = {"Revenue_WoW_Pct": r.WoW_Pct, "Revenue_YoY_Pct": r.YoY_Pct,
         "Revenue_WoW_Abs": r.Revenue_TW - r.Revenue_LW, "Revenue_YoY_Abs": r.Revenue_TW - r.Revenue_LY,
         "Revenue_WoW_Ratio": r.Revenue_TW / r.Revenue_LW if r.Revenue_LW else np.nan,
         "Units_WoW_Pct": pct(r.Units_TW, r.Units_LW), "Units_WoW_Abs": r.Units_TW - r.Units_LW,
         "Orders_WoW_Pct": pct(r.Orders_TW, r.Orders_LW),
         "Share_LW": r.Revenue_LW / t.Revenue_LW.sum() * 100,
         "Share_WoW_pp": r.Share_Pct - r.Revenue_LW / t.Revenue_LW.sum() * 100,
         "Units_YoY_Pct": np.nan}
    if col in m:
        return m[col]
    return r[col] if col in r else np.nan


VT = re.compile(r"^\s*(?P<s1>[+\-−])?\s*(?P<cur>£)?\s*(?P<s2>[+\-−])?(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<ord>st|nd|rd|th)?"
                r"\s*(?P<mult>k|m|bn)?\s*(?P<unit>%|pp|x|percent|units)?\s*$", re.I)


def my_parse(vt):
    m = VT.match(vt)
    if not m:
        return None
    num = m["num"].replace(",", "")
    dec = len(num.split(".")[1]) if "." in num else 0
    scale = {"k": 1e3, "m": 1e6, "bn": 1e9}.get((m["mult"] or "").lower(), 1.0)
    sgn = m["s1"] or m["s2"]
    return {"cited": float(num), "dec": dec, "scale": scale, "sign": -1 if sgn in ("-", "−") else (1 if sgn == "+" else 0)}


def human_correct(vt, direction, v, period, metric):
    """Would a careful human say the cited text is a correct rendering of v? (round or truncate at cited precision)"""
    p = my_parse(vt)
    if p is None or pd.isna(v):
        return False
    level = period in ("TW", "LW", "LY") or metric in ("rank",)
    sign = p["sign"] or (0 if level else {"up": 1, "down": -1}.get(direction, 0))
    if sign and v != 0 and np.sign(v) != sign:
        return False
    x = abs(v) / p["scale"]
    ok = abs(x - p["cited"]) <= 0.5 * 10 ** (-p["dec"]) + 1e-9
    if p["scale"] > 1:  # allow truncation for k/m
        ok |= abs(np.floor(x * 10 ** p["dec"]) / 10 ** p["dec"] - p["cited"]) < 1e-9
    return bool(ok)


HEDGES = {"900%": lambda v: v > 900, "£90k": lambda v: 85e3 <= v < 90e3 or abs(v / 1e3 - 90) <= 0.5,
          "over 12,000%": lambda v: v > 12000}

EXPECTED_TYPE = {"wrong_value": "Wrong_value", "wrong_direction": "Wrong_value",
                 "wrong_entity": "Wrong_entity", "wrong_metric": "Wrong_metric"}
ERR = {"Wrong_value", "Wrong_entity", "Wrong_metric"}


def gt_label(planted):
    if planted == "unverifiable":
        return "Unverifiable"
    if planted in ("correct", "hedge", "ambiguous"):
        return "Correct"
    return "Wrong"


def build(note_key, tup):
    line, ent, metric, period, vt, direction, gt_ent, gt_col, planted, comment = tup
    return {"note": note_key, "line": line, "line_idx": line, "entity": ent, "metric": metric, "period": period,
            "value_text": vt, "direction": direction, "gt_entity": gt_ent, "gt_col": gt_col, "planted": planted,
            "comment": comment}


def check_ground_truth(c, t):
    v = gt_value(t, c["gt_entity"], c["gt_col"])
    c["true_value"] = None if pd.isna(v) else float(v)
    hc = human_correct(c["value_text"], c["direction"], v, c["period"], c["metric"])
    if c["planted"] == "hedge":
        hc = HEDGES[c["value_text"]](abs(v))
        assert hc, c
    elif c["planted"] in ("correct", "ambiguous"):
        assert hc, f"annotated correct but pandas disagrees: {c} true={v}"
    elif c["planted"] == "unverifiable":
        pass
    else:
        assert not hc, f"annotated wrong but pandas says correct: {c} true={v}"
    c["truth"] = gt_label(c["planted"])


def outcome(gt, gate):
    if gt == "Correct":
        return {"Correct": "ok", "Unverifiable": "coverage_loss"}.get(gate, "FALSE_ALARM")
    if gt == "Wrong":
        return {"Correct": "MISS", "Unverifiable": "MISS_hidden"}.get(gate, "detected")
    return "ok" if gate == "Unverifiable" else "MISLABEL_unverifiable"


def run_main():
    rows = []
    for key, (tid, lines) in NOTES.items():
        t = table(tid)
        note = "\n".join(lines)
        claims = [build(key, tup) for tup in CLAIMS[key]]
        for c in claims:
            check_ground_truth(c, t)
        gate_in = [{k: c[k] for k in ("line", "line_idx", "entity", "metric", "period", "value_text", "direction")}
                   for c in claims]
        out = ng.gate_level2(note, t, note_id=key, claims=gate_in)
        ents = t.Entity.astype(str).tolist()
        for c, o in zip(claims, out):
            c["bound_row"] = ng.match_entity(c["entity"], ents)
            c["gate"], c["evidence"] = o["label"], o["evidence"]
            c["outcome"] = outcome(c["truth"], c["gate"])
            c["type_ok"] = (c["gate"] == EXPECTED_TYPE.get(c["planted"])) if c["truth"] == "Wrong" else None
            c["table"] = tid
            rows.append(c)
    return pd.DataFrame(rows)


def run_probes():
    rows = []
    for key, line, *rest in PROBES:
        tid, lines = NOTES[key]
        t = table(tid)
        c = build(key, (line, *rest))
        check_ground_truth(c, t)
        lab, ev = ng.verify_claim(c, t)
        c.update(gate=lab, evidence=ev, outcome=outcome(c["truth"], lab), bound_row=ng.match_entity(c["entity"], t.Entity.astype(str).tolist()))
        rows.append(c)
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ regex extractor on the same notes


def norm_num(vt):
    r = ng.parse_value_text(vt)
    return None if r is None else (round(abs(r["cited"]), 6), r["scale"])


def canon_metric(m):
    return {"sales": "revenue"}.get((m or "").lower(), (m or "").lower())


def run_regex(main):
    recs = []
    spurious = []
    for key, (tid, lines) in NOTES.items():
        t = table(tid)
        ents = t.Entity.astype(str).tolist()
        note = "\n".join(lines)
        rc = ng.regex_claims(note, t)
        rlab = ng.gate_level2(note, t, note_id=key, claims=rc)
        gold = main[main.note == key].to_dict("records")
        used = set()
        for g in gold:
            gk = norm_num(g["value_text"])
            hit = None
            for i, (r, lab) in enumerate(zip(rc, rlab)):
                if i in used or r["line_idx"] != g["line"]:
                    continue
                if norm_num(r["value_text"]) == gk:
                    hit = i
                    break
            if hit is None:
                recs.append({**{k: g[k] for k in ("note", "line", "entity", "metric", "period", "value_text", "truth", "planted")},
                             "found": False})
                continue
            used.add(hit)
            r, lab = rc[hit], rlab[hit]
            g_row = ng.match_entity(g["entity"], ents) if g["gt_entity"] is not None else None
            # what the gold entity really is (use annotated gt_entity, TOTAL for totals)
            gold_ent = g["gt_entity"]
            recs.append({"note": key, "line": g["line"], "value_text": g["value_text"], "truth": g["truth"], "planted": g["planted"],
                         "found": True, "gold_entity": gold_ent, "regex_entity": r["entity"] or "",
                         "ent_ok": (r["entity"] == gold_ent) if gold_ent else (r["entity"] in ("", None)),
                         "gold_metric": g["metric"], "regex_metric": r["metric"], "met_ok": canon_metric(r["metric"]) == g["metric"],
                         "gold_period": g["period"], "regex_period": r["period"],
                         "per_ok": r["period"] == g["period"] or (g["metric"] in ("rank", "share") and g["period"] == "TW" and r["period"] == "TW"),
                         "gold_dir": g["direction"], "regex_dir": r["direction"],
                         "regex_gate": lab["label"], "llm_gate": g["gate"],
                         "regex_outcome": outcome(g["truth"], lab["label"])})
        for i, (r, lab) in enumerate(zip(rc, rlab)):
            if i not in used:
                spurious.append({"note": key, "line": r["line_idx"], "value_text": r["value_text"], "entity": r["entity"],
                                 "metric": r["metric"], "period": r["period"], "gate": lab["label"]})
    return pd.DataFrame(recs), pd.DataFrame(spurious)

# ------------------------------------------------------------------ Monte Carlo: how errors get typed in big tables


def fmt_like(v, kind):
    if kind == "money0":
        return f"£{abs(v):,.0f}"
    if kind == "money2":
        return f"£{abs(v):,.2f}"
    if kind == "pct1":
        return f"{abs(v):.1f}%"
    if kind == "count":
        return f"{abs(v):,.0f}"


def monte_carlo(seed=7, n_per=150):
    rng = np.random.default_rng(seed)
    idx = pd.read_csv(ROOT / "data" / "tables" / "index.csv")
    res = []
    specs = [("revenue", "TW", "Revenue_TW", "money0"), ("revenue", "TW", "Revenue_TW", "money2"),
             ("revenue", "WoW", "WoW_Pct", "pct1"), ("units", "TW", "Units_TW", "count"),
             ("share", "TW", "Share_Pct", "pct1")]
    for size in ["small", "medium", "large"]:
        tids = idx[idx["size"] == size].table_id.tolist()
        for kind in ["wrong_value", "wrong_period"]:
            for metric, period, col, fmt in specs:
                if kind == "wrong_period" and metric == "share":
                    continue
                for _ in range(n_per):
                    t = table(rng.choice(tids))
                    t2 = t.dropna(subset=[col])
                    if kind == "wrong_period":
                        src = {"Revenue_TW": "Revenue_LW", "WoW_Pct": "YoY_Pct", "Units_TW": "Units_LW"}[col]
                        t2 = t2.dropna(subset=[src])
                        t2 = t2[(t2[src] != 0) & (t2[src].round(1) != t2[col].round(1))]
                    if t2.empty:
                        continue
                    r = t2.iloc[rng.integers(len(t2))]
                    true = float(r[col])
                    if kind == "wrong_value":
                        f = rng.uniform(0.03, 0.30) * rng.choice([-1, 1])
                        cited = true * (1 + f)
                    else:
                        cited = float(r[src])
                    vt = fmt_like(cited, fmt)
                    if fmt_like(true, fmt) == vt:
                        continue
                    direction = "none"
                    if period == "WoW":
                        direction = "up" if cited > 0 else "down"
                    lab, ev = ng.verify_claim({"entity": r.Entity.title(), "metric": metric, "period": period,
                                               "value_text": vt, "direction": direction}, t)
                    res.append({"size": size, "kind": kind, "spec": f"{metric}/{period}/{fmt}", "label": lab})
    return pd.DataFrame(res)

# ------------------------------------------------------------------ systematic shortened-name binding


COLOURS = r"\b(RED|WHITE|PINK|BLUE|GREEN|CREAM|IVORY|BLACK|PURPLE|GREY|MINT|YELLOW|CLEAR|NATURAL)\b"
QUAL = r"^(SET OF \d+|SET/\d+|SET \d+|PACK OF \d+|\d+ PIECE)\s+"


def shorten_experiment():
    out = []
    for tid in ["large_product_2011-W06", "large_product_2011-W08", "large_product_2010-W49", "medium_product_2010-W50"]:
        t = table(tid)
        ents = t.Entity.astype(str).tolist()
        toks = {e: set(re.findall(r"[a-z0-9]+", e.lower())) for e in ents}
        for e in ents:
            variants = {}
            v1 = re.sub(QUAL, "", e).strip()
            v1 = re.sub(r"\b(DESIGN|T-LIGHT)\b", "", v1).strip()
            if v1 != e:
                variants["drop qualifier/DESIGN/T-LIGHT"] = v1
            v2 = re.sub(COLOURS, "", e)
            if v2 != e:
                variants["drop colour"] = v2
            for rule, v in variants.items():
                v = re.sub(r"\s+", " ", v).strip().title()
                if len(v) < 5:
                    continue
                vt = set(re.findall(r"[a-z0-9]+", v.lower()))
                owners = [x for x in ents if vt <= toks[x]]
                got = ng.match_entity(v, ents)
                out.append({"table": tid, "rule": rule, "entity": e, "variant": v, "n_owners": len(owners),
                            "unique": len(owners) == 1,
                            "result": "right" if got == e else ("none" if got is None else ("wrong" if len(owners) == 1 else "picked_other_of_ambiguous"))})
    return pd.DataFrame(out)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 80)
    df = run_main()
    print(f"NOTES: {len(NOTES)}   CLAIMS: {len(df)}")
    print("GT:", df["truth"].value_counts().to_dict(), " planted:", df.planted.value_counts().to_dict())
    print("\nGate labels vs GT:\n", pd.crosstab(df["truth"], df.gate, margins=True))
    print("\nOutcomes:", df.outcome.value_counts().to_dict())
    w = df[df["truth"] == "Wrong"]
    print(f"\nWrong claims: {len(w)}; detected {int((w.outcome == 'detected').sum())}; type agrees {int(w.type_ok.sum())}")
    print(pd.crosstab(w.planted, w.gate))
    # error rate as the pipeline would report it
    ver = df[df.gate != "Unverifiable"]
    gate_rate = ver.gate.isin(ERR).mean()
    gt_ver = df[df["truth"] != "Unverifiable"]
    true_rate = (gt_ver["truth"] == "Wrong").mean()
    print(f"\nclaim_error_rate reported by gate: {gate_rate:.3f} ({ver.gate.isin(ERR).sum()}/{len(ver)}); "
          f"true error rate: {true_rate:.3f} ({(gt_ver["truth"] == 'Wrong').sum()}/{len(gt_ver)})")
    print("\nPer note:")
    for k, g in df.groupby("note"):
        v = g[g.gate != "Unverifiable"]
        gv = g[g["truth"] != "Unverifiable"]
        print(f"  {k} {NOTES[k][0]:26} n={len(g):2}  gate_err={v.gate.isin(ERR).sum():2}/{len(v):2}  true_err={(gv["truth"] == 'Wrong').sum():2}/{len(gv):2}"
              f"  outcomes={g.outcome.value_counts().to_dict()}")
    cols = ["note", "line", "entity", "metric", "period", "value_text", "direction", "planted", "truth", "bound_row", "gate", "evidence", "comment"]
    print("\nEVERY non-ok outcome (false alarms, misses, coverage losses):")
    print(df[df.outcome != "ok"][cols + ["outcome"]].to_string(index=False))
    print("\nWrong claims whose gate TYPE disagrees with the planted type:")
    print(w[(w.outcome == "detected") & (~w.type_ok.astype(bool))][cols].to_string(index=False))
    print("\nFuzzy/alias bindings (entity text differs from bound row):")
    fz = df[df.bound_row.notna() & (df.entity.str.upper() != df.bound_row.str.upper())][["note", "entity", "bound_row", "gt_entity"]].drop_duplicates()
    fz["bound_right"] = fz.bound_row == fz.gt_entity
    print(fz.to_string(index=False))

    pr = run_probes()
    print("\nPROBES (alternative extractor emissions):")
    print(pr[["note", "line", "entity", "period", "value_text", "direction", "truth", "bound_row", "gate", "outcome", "evidence", "comment"]].to_string(index=False))

    rx, sp = run_regex(df)
    f = rx[rx.found]
    print(f"\nREGEX extractor on the same {len(NOTES)} notes: gold claims {len(rx)}, found {len(f)} ({len(f)/len(rx):.0%}), spurious extra claims {len(sp)}")
    print(f"  entity right {f.ent_ok.sum()}/{len(f)} ({f.ent_ok.mean():.0%}); metric right {f.met_ok.sum()}/{len(f)} ({f.met_ok.mean():.0%}); "
          f"period right {f.per_ok.sum()}/{len(f)} ({f.per_ok.mean():.0%}); all three right {(f.ent_ok & f.met_ok & f.per_ok).sum()}/{len(f)} "
          f"({(f.ent_ok & f.met_ok & f.per_ok).mean():.0%}); direction right {(f.gold_dir == f.regex_dir).sum()}/{len(f)}")
    print("  regex-path outcomes vs GT:", f.regex_outcome.value_counts().to_dict())
    vr = f[f.regex_gate != "Unverifiable"]
    print(f"  regex-path claim_error_rate: {vr.regex_gate.isin(ERR).mean():.3f} ({vr.regex_gate.isin(ERR).sum()}/{len(vr)}) "
          f"on found claims; spurious claims labelled error: {(sp["gate"].isin(ERR).sum() if len(sp) else 0)}")
    for k, g in f.groupby("note"):
        print(f"   note {k}: n={len(g)} ent {g.ent_ok.mean():.0%} metric {g.met_ok.mean():.0%} period {g.per_ok.mean():.0%} all3 {(g.ent_ok & g.met_ok & g.per_ok).mean():.0%}")
    print("  regex entity errors:")
    print(f[~f.ent_ok][["note", "line", "value_text", "gold_entity", "regex_entity", "regex_gate", "truth"]].to_string(index=False))
    print("  regex metric/period errors:")
    print(f[~(f.met_ok & f.per_ok)][["note", "line", "value_text", "gold_metric", "regex_metric", "gold_period", "regex_period", "regex_gate", "truth"]].to_string(index=False))
    print("  gold claims regex never produced:")
    print(rx[~rx.found].to_string(index=False) if (~rx.found).any() else "  none")
    print("  spurious regex claims:")
    print(sp.to_string(index=False) if len(sp) else "  none")

    mc = monte_carlo()
    print("\nMONTE CARLO: gate label for synthetic wrong claims (entity correct, value wrong or from wrong period)")
    print(pd.crosstab([mc["kind"], mc["size"]], mc.label, normalize="index").round(3))
    print(pd.crosstab([mc["kind"], mc["size"]], mc.label))
    print(pd.crosstab([mc["kind"], mc["spec"]], mc.label, normalize="index").round(3))

    se = shorten_experiment()
    print("\nSHORTENED NAMES (systematic):")
    print(pd.crosstab([se.rule, se.unique], se.result, margins=True))
    print(se[se.result.isin(["wrong", "picked_other_of_ambiguous"])].head(40).to_string(index=False))

    out = HERE / "level2_results.json"
    json.dump({"claims": df.drop(columns=[]).to_dict("records"), "probes": pr.to_dict("records"),
               "regex": rx.to_dict("records"), "spurious": sp.to_dict("records")}, open(out, "w"), indent=1, default=str)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
