"""Build results/dashboard.html: one self-contained interactive page covering every run in runs/.

Usage: python src/dashboard.py   (analyze.py also calls this on every run)
The page embeds the per-note, per-number and per-claim results plus the source tables, and recomputes
rates and bootstrap intervals in the browser for whatever filters the reader picks.
"""
import base64
import json
import re
from datetime import datetime

import pandas as pd
import yaml

from config import ROOT, RESULTS_DIR, RUNS_DIR, TABLES_DIR, load_config

PAPER_DIR = ROOT / "paper"

TEMPLATE = __file__.replace("dashboard.py", "dashboard_template.html")
NUMBER_COLS = ["line_idx", "raw_text", "label", "evidence", "unit", "position"]
CLAIM_COLS = ["line_idx", "entity", "metric", "period", "value_text", "label", "evidence", "direction", "qualifier", "adjusted"]


def read_csv(path):
    """A run CSV, or an empty frame when the run produced no rows."""
    try:
        return pd.read_csv(path, keep_default_na=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def clean(v):
    """JSON-safe scalar: NaN becomes null, numpy numbers become plain Python numbers."""
    if v is None or (isinstance(v, float) and v != v):
        return None
    if hasattr(v, "item"):
        v = v.item()
        if isinstance(v, float) and v != v:
            return None
    return v


VIRTUAL = {  # claim-level virtual columns and the stored columns they are computed from
    "Revenue_WoW_Pct": ["WoW_Pct"], "Revenue_YoY_Pct": ["YoY_Pct"], "Revenue_WoW_Abs": ["Revenue_TW", "Revenue_LW"],
    "Revenue_YoY_Abs": ["Revenue_TW", "Revenue_LY"], "Units_WoW_Abs": ["Units_TW", "Units_LW"], "Units_WoW_Pct": ["Units_TW", "Units_LW"],
    "Orders_WoW_Abs": ["Orders_TW", "Orders_LW"], "Orders_WoW_Pct": ["Orders_TW", "Orders_LW"], "Share_LW": ["Revenue_LW"],
    "Share_WoW_Abs": ["Share_Pct", "Revenue_LW"], "Revenue_WoW_Ratio": ["Revenue_TW", "Revenue_LW"],
    "Revenue_YoY_Ratio": ["Revenue_TW", "Revenue_LY"], "Units_WoW_Ratio": ["Units_TW", "Units_LW"], "Orders_WoW_Ratio": ["Orders_TW", "Orders_LW"],
}


def evidence_targets(ev, table):
    """Cells and whole columns an evidence string points at, as {"c": [[entity, column], ...], "k": [column, ...]}.

    Done here rather than in the page because Python knows the exact entity names, which may contain
    dots, commas or plus signs ("WILLOW BRANCH LIGHTS.", "POPCORN HOLDER , SMALL").
    """
    cells, cols = [], []
    if not isinstance(ev, str) or not ev:
        return {"c": cells, "k": cols}
    names = [str(e) for e in table["Entity"]]
    columns = list(table.columns)

    def cell(ent, col):
        for c in VIRTUAL.get(col, [col]):
            if c in columns and [ent, c] not in cells:
                cells.append([ent, c])

    def column(col):
        for c in VIRTUAL.get(col, [col]):
            if c in columns and c not in cols:
                cols.append(c)

    ev = re.sub(r"^group figure( \(extractor said TOTAL\))?: ", "", ev)
    m = re.match(r"cell\((.*), ([A-Za-z_]+)\)=", ev)
    if m:
        cell(m.group(1), m.group(2))
        return {"c": cells, "k": cols}
    for m in re.finditer(r"group\[([^\]]+)\]\.([A-Za-z_]+)", ev):
        for ent in m.group(1).split("; "):
            cell(ent, m.group(2))
    for m in re.finditer(r"family\[([^\]]+)\]\.([A-Za-z_]+)", ev):
        frags = [" " + f.lower() + " " for f in m.group(1).split(" + ")]
        for ent in names:
            padded = " " + " ".join(re.findall(r"[a-z]+", ent.lower())) + " "
            if any(f in padded for f in frags):
                cell(ent, m.group(2))
    for m in re.finditer(r"sum_excluding\((.+?)\.([A-Za-z_]+)\)", ev):
        column(m.group(2))  # every row except the named one
    for m in re.finditer(r"per_(order|unit)\((.+?)\.Revenue_(TW|LW)\)", ev):
        cell(m.group(2), "Revenue_" + m.group(3))
        cell(m.group(2), ("Orders_" if m.group(1) == "order" else "Units_") + m.group(3))
    m = re.match(r"(?:sum|share)_top(\d+)\(Revenue_TW\)", ev)
    if m:
        for ent in table.sort_values("Revenue_TW", ascending=False)["Entity"].head(int(m.group(1))):
            cell(str(ent), "Revenue_TW")
    for m in re.finditer(r"(?<![\w.])(?:sum|mean|median|max|min)\(([A-Za-z_]+)\)", ev):
        column(m.group(1))
    if ev.startswith("count(rows)"):
        column("Entity")
    rest = re.sub(r"sum_excluding\([^)]*\)|per_(?:order|unit)\([^)]*\)|group\[[^\]]*\]|family\[[^\]]*\]", " ", ev)
    for ent in sorted(names, key=len, reverse=True):  # longest names first, so "X SMALL" wins over "X"
        for m in re.finditer(re.escape(ent) + r"\.([A-Za-z_]+)", rest):
            cell(ent, m.group(1))
        rest = re.sub(re.escape(ent) + r"\.[A-Za-z_]+", " ", rest)
    for m in re.finditer(r"TOTAL\.([A-Za-z_]+)", rest):
        column(m.group(1))
    return {"c": cells, "k": cols}


def utf16_offset(line, pos):
    """The page indexes strings in UTF-16 units; the gate counts code points. They differ after emoji."""
    return len(str(line)[:int(pos)].encode("utf-16-le")) // 2


def rows(df, cols, note_index, tables=None, lines=None):
    """Rows as compact lists: note position, the given columns, then highlight targets for the evidence."""
    if df.empty:
        return []
    df = df.copy()
    for c in cols:
        if c not in df:
            df[c] = None  # runs gated before a column existed
    out = []
    for rec in df[["note_id", "table_id"] + cols].itertuples(index=False):
        idx = note_index.get(rec[0])
        if idx is None:
            continue
        vals = [clean(v) for v in rec[2:]]
        if "position" in cols and lines is not None:  # convert to UTF-16 units for the page
            li, pi = cols.index("line_idx"), cols.index("position")
            note_lines = lines[idx]
            if vals[pi] is not None and vals[li] is not None and vals[li] < len(note_lines):
                vals[pi] = utf16_offset(note_lines[vals[li]], vals[pi])
        ev = vals[cols.index("evidence")] if "evidence" in cols else None
        out.append([idx] + vals + [evidence_targets(ev, tables[rec[1]]) if tables is not None else None])
    return out


def run_label(models):
    names = {"mock": "Mock model", "claude-opus-5": "Claude Opus 5", "gpt-4.1": "GPT-4.1",
             "llama-3.3-70b": "Llama 3.3 70B", "claude-haiku-4-5": "Claude Haiku 4.5"}
    return ", ".join(names.get(m, m) for m in models)


def run_payload(run_dir):
    notes = read_csv(run_dir / "notes.csv")
    if notes.empty:
        return None
    settings = json.loads((run_dir / "run_settings.json").read_text(encoding="utf-8"))
    notes = notes.sort_values("note_id").reset_index(drop=True)
    note_index = {nid: i for i, nid in enumerate(notes["note_id"])}  # note_id -> position in this run's list
    texts, out_notes = [], []
    for i, r in notes.iterrows():
        path = run_dir / "notes" / f"{r['note_id']}.json"
        rec = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"note": "", "trace": {}}
        texts.append([l for l in rec.get("note", "").splitlines() if l.strip()])  # same line split as the gate
        rounds = r.get("s3_round_rates")
        out_notes.append({
            "idx": i, "id": r["note_id"], "table": r["table_id"], "size": r["size"], "etype": r["entity_type"],
            "week": r["week"], "nRows": clean(r["n_rows"]), "strategy": r["strategy"], "model": r["model"],
            "repeat": clean(r["repeat"]), "failed": bool(r["failed"]), "formatOk": bool(r["format_ok"]),
            "nChecked": clean(r["n_numbers_checked"]), "nUns": clean(r["n_Unsupported"]),
            "nCell": clean(r["n_Supported_cell"]), "nDer": clean(r["n_Supported_derived"]), "nSkip": clean(r["n_Skipped"]),
            "nClaimsVer": clean(r["n_claims_verifiable"]), "nCorrect": clean(r["n_Correct"]),
            "nWE": clean(r["n_Wrong_entity"]), "nWM": clean(r["n_Wrong_metric"]), "nWV": clean(r["n_Wrong_value"]),
            "nUnv": clean(r["n_Unverifiable"]), "cost": clean(r["cost_usd"]), "latency": clean(r["latency_s"]),
            "tokIn": clean(r["tokens_in"]), "tokOut": clean(r["tokens_out"]), "calls": clean(r["n_calls"]),
            "s3": json.loads(rounds) if isinstance(rounds, str) and rounds.startswith("[") else None,
            "extractor": clean(r.get("extractor")) or "", "error": rec.get("trace", {}).get("error", ""),
        })
    models = settings.get("models", sorted(notes["model"].unique()))
    is_mock = list(models) == ["mock"]
    used = run_dir / "config_used.yaml"
    s3_rounds = yaml.safe_load(used.read_text(encoding="utf-8")).get("s3_max_rounds", 2) if used.exists() else None
    tables = {tid: pd.read_csv(TABLES_DIR / f"{tid}.csv") for tid in notes["table_id"].unique()}
    numbers = rows(read_csv(run_dir / "numbers.csv"), NUMBER_COLS, note_index, tables, texts)
    numbers.sort(key=lambda x: (x[0], x[1], x[6] if x[6] is not None else 0))  # note, line, position
    claims = rows(read_csv(run_dir / "claims.csv"), CLAIM_COLS, note_index, tables)
    overlay = v1_overlay(notes["note_id"], texts, [tables[t] for t in notes["table_id"]], numbers) if not is_mock else None
    return {
        "id": run_dir.name, "models": models, "isMock": is_mock, "label": run_label(models),
        "sub": f"{run_dir.name}, {len(notes)} notes" + (", planted errors" if is_mock else ""),
        "notes": out_notes, "texts": texts, "s3MaxRounds": s3_rounds,
        "numbers": numbers, "claims": claims, "flags": gate_flags(numbers, claims),
        "v1": overlay[0] if overlay else None, "v1Counts": overlay[1] if overlay else None,
    }


def digits(text):
    return re.sub(r"[^0-9.]", "", str(text))


def number_keys(numbers):
    """(note position, line, raw text, occurrence of that raw text in the line) for each number, in order."""
    keys, seen = [], {}
    for x in numbers:
        k = (x[0], x[1], x[2])
        keys.append(k + (seen.get(k, 0),))
        seen[k] = seen.get(k, 0) + 1
    return keys


def claim_number(keys, idx, li, value_text, taken):
    """The number a claim's value refers to: same note and line, same text, else same digits; None if absent."""
    same_line = [k for k in keys if k[0] == idx and k[1] == li]
    match = [k for k in same_line if k[2] == str(value_text)] or \
            [k for k in same_line if digits(k[2]) and digits(k[2]) == digits(value_text)]
    return next((k for k in match if k not in taken), match[0]) if match else None


def gate_flags(numbers, claims):
    """One 0/1 per number: the current gate flagged it at Level 1, or it is the value of a wrong claim."""
    keys = number_keys(numbers)
    flagged = {k for k, x in zip(keys, numbers) if x[3] == "Unsupported"}
    for c in claims:
        if str(c[6]).startswith("Wrong_"):
            k = claim_number(keys, c[0], c[1], c[5], flagged)
            if k:
                flagged.add(k)
    return [1 if k in flagged else 0 for k in keys]


def v1_overlay(note_ids, texts, table_of, numbers):
    """Which numbers the first gate (v1) flagged at either level, for notes the paper re-scored with it.

    Level 1 flags come from running paper/v1/number_gate_v1.py again (it is deterministic); claim flags
    come from paper/data/v1_flags.csv (written by paper/rescore_v1.py) and are matched to the number
    with the same value in the same line. Returns (one 0/1 per number, counts) or None.
    """
    flags_path, v1_path = PAPER_DIR / "data" / "v1_flags.csv", PAPER_DIR / "v1" / "number_gate_v1.py"
    if not (flags_path.exists() and v1_path.exists()):
        return None
    flags = pd.read_csv(flags_path)
    ids = list(note_ids)
    if not set(flags["note_id"]) & set(ids):
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("gate_v1", v1_path)
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)
    keys = number_keys(numbers)
    keyset = set(keys)
    flagged, l1, l2 = set(), 0, 0
    for idx, nid in enumerate(ids):
        occ = {}
        for r in v1.gate_level1("\n".join(texts[idx]), table_of[idx], nid):
            k = (idx, r["line_idx"], r["raw_text"])
            o = occ.get(k, 0)
            occ[k] = o + 1
            if r["label"] == "Unsupported" and k + (o,) in keyset:
                flagged.add(k + (o,))
                l1 += 1
    for _, f in flags[flags["note_id"].isin(ids)].iterrows():
        idx = ids.index(f["note_id"])
        if f["line"] not in texts[idx]:
            continue
        li = texts[idx].index(f["line"])
        l2 += 1
        k = claim_number(keys, idx, li, f["value_text"], flagged)
        if k:
            flagged.add(k)
    return [1 if k in flagged else 0 for k in keys], {"numberFlags": l1, "claimFlags": l2, "marked": len(flagged)}


def story():
    """Headline figures about the checker itself, from the paper's validation data (paper/data/validation.json)."""
    path = PAPER_DIR / "data" / "validation.json"
    pilot = RUNS_DIR / "smoke_opus" / "notes.csv"
    if not (path.exists() and pilot.exists()):
        return None
    val = json.loads(path.read_text(encoding="utf-8"))
    rates = val.get("held_out2_rates", {})
    notes = pd.read_csv(pilot)
    out = {"rows": val.get("table_rows", []), "v1RealErrors": val.get("v1_real_errors"), "pilotNotes": int(len(notes)),
           "v3Numbers": int(notes["n_Unsupported"].sum()),
           "v3Claims": int(notes[["n_Wrong_entity", "n_Wrong_metric", "n_Wrong_value"]].sum().sum())}
    v1_notes = PAPER_DIR / "data" / "v1_v2_notes.csv"
    if v1_notes.exists():
        out["v1Numbers"] = int(pd.read_csv(v1_notes)["v1_unsup"].sum())
    if "l1_fa" in rates:
        base = notes["n_Unsupported"].sum() / notes["n_numbers_checked"].sum()
        fa = rates["l1_fa"] / rates["l1_correct"]
        recall = rates["l1_value_caught"] / rates["l1_value_wrong"]
        precision = recall * base / (recall * base + fa * (1 - base))
        out.update({"base": float(base), "fa": float(fa), "recall": float(recall), "precision": float(precision),
                    "oneIn": int(round(1 / base)) if base else None})
    flags = PAPER_DIR / "data" / "v1_flags.csv"
    if flags.exists():
        out["v1ClaimFlags"] = int(len(pd.read_csv(flags)))
    return out


def paper_info():
    """Title, page count and size of the compiled paper (paper/numgate.pdf), or None if it has not been built."""
    pdf = PAPER_DIR / "numgate.pdf"
    if not pdf.exists():
        return None
    raw = pdf.read_bytes()
    tex = PAPER_DIR / "numgate.tex"
    m = re.search(r"\\title\{(.*?)\}\n", tex.read_text(encoding="utf-8")) if tex.exists() else None
    title = re.sub(r"\s*\\\\\s*", " ", m.group(1)).strip() if m else "NumGate paper"
    return {"title": title, "pages": len(re.findall(rb"/Type\s*/Page[^s]", raw)), "bytes": len(raw),
            "built": datetime.fromtimestamp(pdf.stat().st_mtime).strftime("%Y-%m-%d")}


def table_payload(table_ids):
    out = {}
    for tid in sorted(table_ids):
        t = pd.read_csv(TABLES_DIR / f"{tid}.csv")
        out[tid] = {"cols": list(t.columns),
                    "rows": [[clean(v) for v in row] for row in t.itertuples(index=False)]}
    return out


def build_data():
    cfg = load_config()
    runs = [p for p in (run_payload(d) for d in sorted(RUNS_DIR.iterdir()) if d.is_dir()) if p]
    # Real-model runs first (largest first), mock runs last, so the page opens on real results.
    runs.sort(key=lambda r: (r["isMock"], -len(r["notes"]), r["id"]))
    table_ids = {n["table"] for r in runs for n in r["notes"]}
    return {"meta": {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "s3MaxRounds": cfg["s3_max_rounds"],
                     "story": story(), "paper": paper_info()},
            "runs": runs, "tables": table_payload(table_ids)}


def write_dashboard(path=None):
    data = build_data()
    blob = json.dumps(data, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    blob = blob.replace("<", "\\u003c")  # keep "</script>" inside note text from closing the data block
    html = open(TEMPLATE, encoding="utf-8").read().replace("__NUMGATE_DATA__", blob)
    pdf = PAPER_DIR / "numgate.pdf"  # the paper ships inside the page so the download button works offline
    html = html.replace("__NUMGATE_PAPER__", base64.b64encode(pdf.read_bytes()).decode("ascii") if pdf.exists() else "")
    html = '<!doctype html>\n<html lang="en">\n' + html + "</html>\n"  # standards mode when opened as a file
    path = path or RESULTS_DIR / "dashboard.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path, data


if __name__ == "__main__":
    out, data = write_dashboard()
    print(f"Wrote {out} with {len(data['runs'])} runs: " + ", ".join(f"{r['id']} ({len(r['notes'])} notes)" for r in data["runs"]))
