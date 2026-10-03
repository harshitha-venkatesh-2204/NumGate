#!/usr/bin/env python3
"""Score the frozen Level 1 gate on held-out set 2 (paper/heldout2/L1).

Run AFTER FROZEN.json was written. Checks the notes/truth sha256 against FROZEN.json and the md5 of
src/number_gate.py against the frozen value, runs number_gate.gate_level1 on every note, aligns gate
records to truth mentions by line index and character-span overlap (pre-registered rule), and writes
results.csv (one row per registered truth mention), gate_records.csv (every gate record, with the
truth mention it matched) and metrics.json.
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper" / "heldout2" / "L1"
sys.path.insert(0, str(ROOT / "src"))
GATE_MD5 = "bcdb434ba30ad334a702493668a0ce39"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    frozen = json.loads((OUT / "FROZEN.json").read_text())
    assert sha256(OUT / "notes.jsonl") == frozen["notes_sha256"], "notes changed after freeze"
    assert sha256(OUT / "truth.csv") == frozen["truth_sha256"], "truth changed after freeze"
    gate_md5 = hashlib.md5((ROOT / "src" / "number_gate.py").read_bytes()).hexdigest()
    assert gate_md5 == GATE_MD5, f"gate md5 {gate_md5} != {GATE_MD5}"
    import number_gate  # noqa: E402

    notes = [json.loads(l) for l in open(OUT / "notes.jsonl")]
    truth = list(csv.DictReader(open(OUT / "truth.csv")))
    for t in truth:
        t["line"], t["start"], t["end"], t["is_word"] = int(t["line"]), int(t["start"]), int(t["end"]), int(t["is_word"])

    gate_rows = []
    for n in notes:
        table = pd.read_csv(ROOT / "data" / "tables" / f"{n['table_id']}.csv")
        lines = n["text"].split("\n")
        recs = number_gate.gate_level1(n["text"], table, n["note_id"])
        for k, r in enumerate(recs):
            li = int(r["line_idx"])
            line = lines[li]
            s = int(r["position"])  # position is relative to the note line
            if line[s:s + len(r["raw_text"])] != r["raw_text"]:
                off = line.find(r["sentence"])
                assert off >= 0, ("sentence not in line", r["sentence"], line)
                s = off + int(r["position"])
            assert line[s:s + len(r["raw_text"])] == r["raw_text"], ("span mismatch", r, line)
            gate_rows.append(dict(note_id=n["note_id"], table_id=n["table_id"], line=li + 1, rec_idx=k,
                                  g_start=s, g_end=s + len(r["raw_text"]), raw_text=r["raw_text"],
                                  value=r["value"], unit=r["unit"], is_rank=r["is_rank"], g_qualifier=r["qualifier"],
                                  g_direction=r["direction"], gate_label=r["label"], evidence=r["evidence"],
                                  matched_truth=""))

    # align: each truth mention -> the overlapping gate record with the largest overlap
    by_line = {}
    for g in gate_rows:
        by_line.setdefault((g["note_id"], g["line"]), []).append(g)
    results = []
    for t in truth:
        cands = []
        for g in by_line.get((t["note_id"], t["line"]), []):
            ov = min(t["end"], g["g_end"]) - max(t["start"], g["g_start"])
            if ov > 0:
                cands.append((ov, g))
        cands.sort(key=lambda x: -x[0])
        g = cands[0][1] if cands else None
        if g is not None:
            key = f"{t['note_id']}|{t['line']}|{t['idx']}"
            g["matched_truth"] = (g["matched_truth"] + ";" if g["matched_truth"] else "") + key
        gl = g["gate_label"] if g else "not_extracted"
        results.append({**{k: t[k] for k in ("note_id", "table_id", "line", "idx", "start", "end", "value_text", "label",
                                             "kind", "entity", "metric", "period", "qualifier", "direction_text",
                                             "true_value", "cited_value", "precision", "error_type", "error_src",
                                             "is_word")},
                        "gate_label": gl, "gate_raw_text": g["raw_text"] if g else "",
                        "gate_value": g["value"] if g else "", "gate_unit": g["unit"] if g else "",
                        "gate_is_rank": g["is_rank"] if g else "", "gate_evidence": g["evidence"] if g else "",
                        "n_overlapping_records": len(cands)})

    def outcome(r):
        lab, gl = r["label"], r["gate_label"]
        sup = gl.startswith("Supported")
        notx = gl in ("Skipped", "not_extracted")
        if lab == "C":
            return "false_alarm" if gl == "Unsupported" else ("correct_not_extracted" if notx else "true_pass")
        if lab == "W":
            return "caught" if gl == "Unsupported" else ("wrong_not_extracted" if notx else "miss")
        return "nm_flagged" if gl == "Unsupported" else ("nm_skipped" if notx else "nm_supported")

    for r in results:
        r["outcome"] = outcome(r)

    cols = list(results[0].keys())
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)
    with open(OUT / "gate_records.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gate_rows[0].keys()))
        w.writeheader()
        w.writerows(gate_rows)

    dig = [r for r in results if not r["is_word"]]
    wrd = [r for r in results if r["is_word"]]
    oc = Counter(r["outcome"] for r in dig)
    lab = Counter(r["label"] for r in dig)
    unmatched_gate = [g for g in gate_rows if not g["matched_truth"]]
    multi = [g for g in gate_rows if ";" in g["matched_truth"]]
    C, W = lab["C"], lab["W"]
    m = {
        "gate_md5": gate_md5,
        "notes": len(notes),
        "numeric_mentions": len(dig),
        "word_form_mentions": len(wrd),
        "word_form_outcomes": dict(Counter(r["outcome"] for r in wrd)),
        "gate_records": len(gate_rows),
        "gate_records_unmatched": len(unmatched_gate),
        "gate_records_matching_several_truth_mentions": len(multi),
        "correct": C,
        "false_alarms": oc["false_alarm"],
        "correct_not_extracted": oc["correct_not_extracted"],
        "correct_passed": oc["true_pass"],
        "wrong": W,
        "caught": oc["caught"],
        "misses": oc["miss"],
        "wrong_not_extracted": oc["wrong_not_extracted"],
        "non_measures": lab["N"] + lab["CNT"],
        "non_measures_N": lab["N"],
        "non_measures_CNT": lab["CNT"],
        "non_measures_skipped": oc["nm_skipped"],
        "non_measures_flagged": oc["nm_flagged"],
        "non_measures_supported": oc["nm_supported"],
        "false_alarm_rate_of_correct": round(oc["false_alarm"] / C, 4),
        "catch_rate_of_wrong": round(oc["caught"] / W, 4),
        "miss_rate_of_wrong": round(oc["miss"] / W, 4),
        "flag_precision": round(oc["caught"] / (oc["caught"] + oc["false_alarm"] + oc["nm_flagged"]), 4),
        "by_label_gate_label": {l: dict(Counter(r["gate_label"] for r in dig if r["label"] == l)) for l in ("C", "W", "N", "CNT")},
        "wrong_by_error_type": {et: dict(Counter(r["outcome"] for r in dig if r["label"] == "W" and r["error_type"] == et))
                                for et in sorted({r["error_type"] for r in dig if r["label"] == "W"})},
        "false_alarms_by_kind": dict(Counter(r["kind"] for r in dig if r["outcome"] == "false_alarm")),
        "by_table": {t: dict(Counter(r["outcome"] for r in dig if r["table_id"] == t)) for t in frozen["tables"]},
    }
    (OUT / "metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps({k: v for k, v in m.items() if not isinstance(v, dict)}, indent=1))
    print("wrong_by_error_type", json.dumps(m["wrong_by_error_type"]))
    print("by_label_gate_label", json.dumps(m["by_label_gate_label"]))
    if unmatched_gate:
        print("UNMATCHED GATE RECORDS:")
        for g in unmatched_gate:
            print(" ", g["note_id"], g["line"], g["raw_text"], g["gate_label"])
    if multi:
        print("RECORDS MATCHING SEVERAL TRUTH MENTIONS:")
        for g in multi:
            print(" ", g["note_id"], g["line"], g["raw_text"], g["matched_truth"])


if __name__ == "__main__":
    main()
