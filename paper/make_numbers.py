"""Write every number quoted in the paper as LaTeX macros, so the text always matches the data.

Usage: python paper/make_numbers.py
Reads runs/smoke_opus, results/summary.csv, paper/data/*.csv, paper/data/validation.json and the stress-suite
outputs in paper/stress/. Writes paper/numbers.tex and paper/table_pilot.tex.
"""
import json
from pathlib import Path

import pandas as pd

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
RUN = ROOT / "runs" / "smoke_opus"


def pct(x, d=1):
    return f"{x * 100:.{d}f}\\%"


def num(n):
    return f"{int(n):,}"


def interim_counts():
    """Rows before and after cleaning; computed once from the raw Excel and cached next to the clean CSV."""
    path = ROOT / "data" / "interim" / "clean_stats.json"
    if path.exists():
        return json.loads(path.read_text())
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    sheets = pd.read_excel(ROOT / "data" / "raw" / "online_retail_II.xlsx", sheet_name=None, dtype={"Invoice": str})
    first, second = [sheets[k] for k in sorted(sheets)]
    raw = len(first[~first["Invoice"].isin(set(second["Invoice"]))]) + len(second)
    clean = sum(1 for _ in open(ROOT / "data" / "interim" / "online_retail_clean.csv", encoding="utf-8")) - 1
    stats = {"raw_rows": raw, "clean_rows": clean}
    path.write_text(json.dumps(stats))
    return stats


def has_error(notes):
    """Notes with at least one flagged error at either level (after adjudication these are the real errors)."""
    return (notes.n_Unsupported + notes.n_Wrong_entity + notes.n_Wrong_metric + notes.n_Wrong_value) > 0


def pilot_macros():
    notes = pd.read_csv(RUN / "notes.csv")
    numbers = pd.read_csv(RUN / "numbers.csv")
    claims = pd.read_csv(RUN / "claims.csv")
    skipped = numbers[numbers.label == "Skipped"].evidence.value_counts()
    ok = notes[~notes.failed]
    by = ok.groupby("strategy")
    s1 = ok[ok.strategy == "S1"]
    m = {
        "NTables": num(len(pd.read_csv(ROOT / "data" / "tables" / "index.csv"))),
        "NNotes": num(len(notes)),
        "NMentions": num(len(numbers)),
        "NChecked": num(notes.n_numbers_checked.sum()),
        "NSkipped": num((numbers.label == "Skipped").sum()),
        "NSkipEntity": num(skipped.get("entity_name", 0)),
        "NSkipCount": num(skipped.get("counting", 0)),
        "NSkipDate": num(skipped.get("date_or_week", 0)),
        "NCell": num((numbers.label == "Supported_cell").sum()),
        "NDerived": num((numbers.label == "Supported_derived").sum()),
        "NClaims": num(len(claims)),
        "NVerifiable": num(notes.n_claims_verifiable.sum()),
        "NUnverifiable": num((claims.label == "Unverifiable").sum()),
        "NGroupClaims": num(claims.evidence.fillna("").str.startswith("group figure").sum()
                            + (claims.entity.fillna("").str.upper() == "GROUP").sum()
                            - ((claims.entity.fillna("").str.upper() == "GROUP") & claims.evidence.fillna("").str.startswith("group figure")).sum()),
        "SoneUnsupN": num(s1.n_Unsupported.sum()),
        "SoneChecked": num(s1.n_numbers_checked.sum()),
        "SoneUnsup": pct(s1.n_Unsupported.sum() / s1.n_numbers_checked.sum()),
        "SoneBind": pct((s1.n_Wrong_entity.sum() + s1.n_Wrong_metric.sum()) / s1.n_claims_verifiable.sum()),
        "VtwoSoneClaim": pct((s1.n_Wrong_entity.sum() + s1.n_Wrong_metric.sum() + s1.n_Wrong_value.sum()) / s1.n_claims_verifiable.sum()),
        "SthreeRevised": num((notes[notes.strategy == "S3"].s3_rounds_used > 0).sum()),
        "NotesErrSone": f"{int(has_error(notes[notes.strategy == 'S1']).sum())} of {int((notes.strategy == 'S1').sum())}",
        "FormatFail": num((~notes.format_ok).sum()),
        "CostTotal": f"\\${notes.cost_usd.sum():.2f}",
        "CostPerNote": f"\\${notes.cost_usd.mean():.3f}",
    }
    return m, by


def v1_macros(m):
    v = pd.read_csv(PAPER / "data" / "v1_v2_notes.csv")
    flags = pd.read_csv(PAPER / "data" / "v1_flags.csv")
    g = v.groupby("strategy")
    claim = g.v1_claim_err.sum() / g.v1_ver.sum()
    unsup = g.v1_unsup.sum() / g.v1_checked.sum()
    notes = pd.read_csv(RUN / "notes.csv")
    s1 = notes[notes.strategy == "S1"]
    v2_s1 = (s1.n_Wrong_entity.sum() + s1.n_Wrong_metric.sum() + s1.n_Wrong_value.sum()) / s1.n_claims_verifiable.sum()
    causes = flags.cause.value_counts()
    big = ["group or top-k figure read as the whole-table total", "level next to an LW/LY label read as a change",
           "direction word applied to a level"]
    m.update({
        "VoneSoneClaim": pct(claim["S1"]), "VoneStwoClaim": pct(claim["S2"]), "VoneSthreeClaim": pct(claim["S3"]),
        "VoneSoneUnsup": pct(unsup["S1"]), "VoneFlags": num(len(flags)),
        "VoneLoneFlags": num(v.v1_unsup.sum()), "VtwoLoneFlags": num(v.v2_unsup.sum()),
        "VoneRatio": f"{claim['S1'] / v2_s1:.0f}" if v2_s1 else "many",
        "CauseGroup": num(causes.get(big[0], 0)), "CauseLevel": num(causes.get(big[1], 0)),
        "CauseDirection": num(causes.get(big[2], 0)), "CauseRest": num(causes[[c for c in causes.index if c not in big]].sum()),
    })


def validation_macros(m):
    """Stress-suite, held-out and adjudication results. Written by hand into paper/data/validation.json."""
    path = PAPER / "data" / "validation.json"
    val = json.loads(path.read_text()) if path.exists() else {}
    pending = "\\textcolor{red}{[pending]}"
    suite = val.get("suites", {})
    for key in ["SuiteLoneItems", "SuiteLoneCorrect", "SuiteLoneFA", "SuiteLoneWrong", "SuiteLoneCaught",
                "SuiteLtwoItems", "SuiteLtwoFA", "SuiteLtwoWrong", "SuiteLtwoCaught"]:
        m[key] = num(suite[key]) if key in suite else pending
    m["HeldLone"] = val.get("held_out_level1", pending)
    m["HeldLtwo"] = val.get("held_out_level2", "")
    m["HeldTwo"] = val.get("held_out2", pending)
    rates = val.get("held_out2_rates", {})
    m["HeldTwoLoneFAPct"] = rates.get("l1_false_alarm_pct", pending)
    m["HeldTwoLtwoFAPct"] = rates.get("l2_false_alarm_pct", pending)
    if "l1_fa" in rates:  # expected precision of a Level 1 flag at the pilot's base rate (Bayes)
        notes = pd.read_csv(RUN / "notes.csv")
        base = notes.n_Unsupported.sum() / notes.n_numbers_checked.sum()  # adjudicated: every pilot flag was real
        fa = rates["l1_fa"] / rates["l1_correct"]
        recall = rates["l1_value_caught"] / rates["l1_value_wrong"]
        precision = recall * base / (recall * base + fa * (1 - base))
        m["PilotBaseRate"] = pct(base, 2)
        m["HeldTwoRecall"] = pct(recall, 0)
        m["ExpectedPrecision"] = pct(precision, 0)
        m["FalsePerReal"] = f"{(1 - precision) / precision:.0f}"
    else:
        for k in ["PilotBaseRate", "HeldTwoRecall", "ExpectedPrecision", "FalsePerReal"]:
            m[k] = pending
    m["Adjudication"] = val.get("adjudication", pending)
    real = val.get("v1_real_errors")
    flags = len(pd.read_csv(PAPER / "data" / "v1_flags.csv"))
    m["VoneFalsePct"] = (f"At least {(flags - real) / flags * 100:.0f} percent" if real is not None else pending)


def pilot_table(by):
    s = pd.read_csv(ROOT / "results" / "summary.csv")
    s = s[s.grouping == "strategy"].set_index("strategy")
    names = {"S1": "S1 Direct", "S2": "S2 Compute-then-write", "S3": "S3 Write-then-verify", "S4": "S4 Facts-template"}
    notes = pd.read_csv(RUN / "notes.csv").groupby("strategy")
    rows = []
    for k in ["S1", "S2", "S3", "S4"]:
        r = s.loc[k]

        def cell(rate, lo, hi):
            return f"{rate * 100:.1f} [{lo * 100:.1f}, {hi * 100:.1f}]"
        rows.append(" & ".join([names[k], num(r.n_numbers), cell(r.unsupported_rate, r.unsupported_ci_lo, r.unsupported_ci_hi),
                                num(r.n_claims), cell(r.binding_error_rate, r.binding_ci_lo, r.binding_ci_hi),
                                cell(r.claim_error_rate, r.claim_ci_lo, r.claim_ci_hi),
                                f"{int(has_error(notes.get_group(k)).sum())} / {len(notes.get_group(k))}",
                                f"\\${r.mean_cost_usd:.3f}"]) + " \\\\")
    head = ("\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
            "Strategy & Numbers & Unsupported \\% & Claims & Wrong entity or metric \\% & Any claim error \\% & Notes with an error & Cost per note \\\\\n\\midrule\n")
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n"


def size_table():
    """Pilot by table size, all strategies pooled."""
    notes = pd.read_csv(RUN / "notes.csv")
    rows = []
    for size in ["small", "medium", "large"]:
        g = notes[notes["size"] == size]
        errs = int((g.n_Wrong_entity + g.n_Wrong_metric + g.n_Wrong_value).sum())
        rows.append(" & ".join([size, num(g.table_id.nunique()), num(len(g)), num(g.n_numbers_checked.sum()),
                                f"{int(g.n_Unsupported.sum())} ({g.n_Unsupported.sum() / g.n_numbers_checked.sum() * 100:.1f})",
                                num(g.n_claims_verifiable.sum()), num(errs), f"{int(has_error(g).sum())} / {len(g)}"]) + " \\\\")
    head = ("\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
            "Table size & Tables & Notes & Numbers & Unsupported (\\%) & Claims & Claim errors & Notes with an error \\\\\n\\midrule\n")
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def validation_table():
    """Table 2: false alarms and caught errors of the gate on the three test sets (paper/data/validation.json)."""
    val = json.loads((PAPER / "data" / "validation.json").read_text())
    head = ("\\begin{tabular}{lrrrr}\n\\toprule\n"
            "Test set & L1 false alarms (\\%) & L1 errors caught (\\%) & L2 false alarms (\\%) & L2 errors detected (\\%) \\\\\n\\midrule\n")
    rows = [" & ".join(r) + " \\\\" for r in val.get("table_rows", [])]
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def main():
    m, by = pilot_macros()
    stats = interim_counts()
    m["NRawRows"], m["NCleanRows"] = num(stats["raw_rows"]), num(stats["clean_rows"])
    v1_macros(m)
    validation_macros(m)
    lines = ["% Generated by paper/make_numbers.py. Do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (PAPER / "table_pilot.tex").write_text(pilot_table(by), encoding="utf-8")
    (PAPER / "table_validation.tex").write_text(validation_table(), encoding="utf-8")
    (PAPER / "table_size.tex").write_text(size_table(), encoding="utf-8")
    for k, v in m.items():
        print(f"{k:16s} {v}")


if __name__ == "__main__":
    main()
