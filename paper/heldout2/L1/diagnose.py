#!/usr/bin/env python3
"""Diagnose every false alarm and miss of the frozen Level 1 gate on held-out set 2 (paper/heldout2/L1).

Run after score_gate.py, and after reading src/number_gate.py and docs/method.md. Reads results.csv, never
edits notes.jsonl or truth.csv. Writes diagnosed.csv (one row per false alarm, flagged non-measure and miss,
with a cause, a class and the section 7 bullet when one applies), caught_check.csv (for every caught wrong value:
would the gate have passed the correct value in the same sentence?) and adds the diagnosis fields to metrics.json.

Classes
  by_design  the failure is a limit docs/method.md section 7 documents and accepts:
             - bullet 9 "Level 1 cannot see binding errors: a real cell attached to the wrong row or period is
               Supported at Level 1 by design" (also applied to a real derived value of the same row for another
               period, and to a real top-k value for a different k: same binding principle, reported separately);
             - bullet 3 "a wrong number can coincide with an unrelated cell" (applied to cells and derived values of a
               compatible unit);
             - bullet 2 "a legitimate derived value outside the operation library is labelled Unsupported".
  rule_gap   a rule in src/number_gate.py misreads the text or misses a family, although the method as written
             (section 2) covers the case; fixable without changing the Level 1 question.
"""
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP, ROUND_FLOOR
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper" / "heldout2" / "L1"
sys.path.insert(0, str(ROOT / "src"))
import number_gate as ng  # noqa: E402

S7 = {
    "binding": "section 7 bullet 9: Level 1 cannot see binding errors (real cell on the wrong row or period is Supported by design)",
    "coincidence": "section 7 bullet 3: a wrong number can coincide with an unrelated cell",
    "library": "section 7 bullet 2: a legitimate derived value outside the operation library is labelled Unsupported",
}

# Hand-verified causes for false alarms and flagged non-measures (each checked with a counterfactual run of the
# frozen gate: removing the trigger makes the same value Supported). Key: (note_id, line, value_text).
FA_CAUSES = {
    ("SP47-2", 2, "£8,735.49"): ("level_signed_by_down_from", "rule_gap", "",
        "'£8,735.49 TW, down from £12,102.54 LW': infer_direction skips the period token TW and applies the "
        "'2.8% up on last week' pattern (up/down + compare word after the number), so the TW level is read as -8,735.49 "
        "and fails the sign check. Unsigned it matches sum(family[PAPER CHAIN KIT].Revenue_TW)=8,735.49."),
    ("SP47-4", 5, "£3,544"): ("level_signed_by_down_from", "rule_gap", "",
        "'£3,544 TW, down from £6,553 LY': same after-number rule signs the level -3,544; unsigned it matches "
        "cell(REGENCY CAKESTAND 3 TIER, Revenue_TW)=3,543.60 at 0 decimals."),
    ("LP16-2", 5, "£2,106.75"): ("level_signed_by_down_from + family_plural_not_recognised", "rule_gap", "",
        "Two causes. (1) '£2,106.75 TW, down from' signs the level negative. (2) 'the 3 wooden garden sets': families() "
        "matches word runs literally, so the plural 'sets' misses GARDEN SET (3 rows, sum 2,106.75); the one-word runs "
        "found are WOODEN (7 rows) and GARDEN (5 rows, adds WHITE WOOD GARDEN PLANT LADDER and CAKE STAND 3 TIER MAGIC "
        "GARDEN). With 'garden set' and no 'down' the value is Supported_derived."),
    ("LP16-2", 5, "£2,604.85"): ("family_plural_not_recognised", "rule_gap", "",
        "Unsigned LW level of the 3 garden sets; not found because the plural 'garden sets' misses family GARDEN SET "
        "(sum Revenue_LW 2,604.85). Supported_derived with 'garden set'."),
    ("LP16-2", 8, "£2,188.64"): ("family_short_word_ignored", "rule_gap", "",
        "'The 5 MUG lines': families() ignores one-word runs shorter than 6 letters, so MUG (5 rows ending in MUG, "
        "sum 2,188.64) is never a family; docs/method.md 2.4 describes families as runs of one to three words "
        "with no length limit."),
    ("MP10-3", 7, "£680.75"): ("mean_of_rest_not_in_library", "by_design", S7["library"],
        "'the remaining 29 products averaged £680.75': the library has sum_after_topk and share_after_topk but no "
        "mean of the rows after the top k (27,393.54 / 29 = 680.75). Documented generic limit; cheap to add."),
    ("LP16-3", 8, "£482.79"): ("mean_of_rest_not_in_library", "by_design", S7["library"],
        "'the other 105 products averaged £482.79': mean of rows after the top 10 (50,693.01 / 105) is not in the "
        "library. Documented generic limit; cheap to add."),
    ("SP47-2", 2, "2"): ("count_word_rule_needs_adjacent_noun", "rule_gap", "",
        "'The 2 PAPER CHAIN KIT lines': skip_reason() only treats a number as a count when the counting noun (lines) "
        "follows it directly or after one adjective; the family name in between defeats it, so the row count is "
        "checked as a measurement and nothing equals 2. The same rule left 20 other row counts unskipped; they "
        "matched unrelated values by coincidence (see nm_supported in metrics.json)."),
}

# Caught wrong values whose CORRECT value the gate would also flag (found by caught_check); these are latent false
# alarms, each confirmed by a counterfactual run of the frozen gate.
LATENT_CAUSES = {
    ("SC-3", 6, "£2,378.59"): ("level_signed_by_down_from", "rule_gap", "",
        "Correct £2,997.06 in 'combined for £2,997.06 TW, down from £5,591.70 LW' is read as -2,997.06 and flagged; "
        "without 'down' it is sum(group[EIRE; Germany].Revenue_TW)."),
    ("MP05-3", 7, "£710.70"): ("mean_of_rest_not_in_library", "by_design", S7["library"],
        "Correct £701.60 (mean of the 16 rows after the top 10) is not in the library."),
    ("LP11-3", 2, "£615.92"): ("mean_of_rest_not_in_library", "by_design", S7["library"],
        "Correct £636.92 (mean of the 88 rows after the top 3) is not in the library."),
    ("LP11-3", 8, "3.2x"): ("inverse_period_ratio_not_in_library", "by_design", S7["library"],
        "Correct 3.1x is Revenue_LY / Revenue_TW; the library only has the TW / LY ratio (0.32)."),
    ("LP16-2", 8, "1,818"): ("family_short_word_ignored", "rule_gap", "",
        "Correct 2,222 units for 'the 5 MUG lines' is not found because MUG (3 letters) is never a family."),
}


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(c - h, 4), round(c + h, 4)]


def classify_miss(r):
    """Cause for a wrong value the gate labelled Supported_*, from the gate's own evidence."""
    ev, et, ent = r["gate_evidence"], r["error_type"], r["entity"]
    m = re.match(r"cell\((.+), (\w+)\)=", ev)
    if m:
        ev_ent, ev_col = m.group(1), m.group(2)
        if et == "other_row":
            return ("real_cell_of_another_row", "by_design", S7["binding"],
                    f"cites {ev_ent} {ev_col}; the line is about {ent}")
        if et == "other_period":
            return ("real_cell_of_another_period_same_row", "by_design", S7["binding"],
                    f"cites {ev_col} of the same row as the claimed period")
        return ("coincidence_with_unrelated_cell", "by_design", S7["coincidence"],
                f"{et} value equals an unrelated cell {ev_ent} {ev_col} of a compatible unit")
    if ev.startswith("per_order(") and et == "other_period":
        return ("derived_value_of_another_period_same_row", "by_design", S7["binding"] + " (derived-value analogue)",
                "revenue per order of the same row for LW, cited as the TW figure")
    if ev.startswith("min(Units_TW)") and et == "other_row":
        return ("real_cell_of_another_row", "by_design", S7["binding"],
                "763 is France's Units_TW; it is also the table minimum, which the aggregate word 'across' made reachable")
    if re.match(r"share_top\d+\(", ev):
        return ("derived_value_of_a_different_topk_group", "by_design", S7["binding"] + " (group-size analogue)",
                "the value is the top-k share for a different k than the line states")
    if ev.startswith("share_outside(") and r["value_text"] == "57":
        return ("plain_number_matches_any_unit", "rule_gap", "",
                "'71 orders (LW 57)': 57 has no unit word after it, so it is 'plain' and may match any unit; it matched "
                "a percentage (share outside the PAPER CHAIN KIT family, 57.1%). The count context ('orders' before the "
                "bracket) is not used; a count-only match would find nothing and flag it.")
    return ("coincidence_with_unrelated_derived_value", "by_design", S7["coincidence"],
            f"{et} value equals an unrelated derived value of a compatible unit ({ev.split('=')[0]})")


ANTONYM = {"rose": "fell", "fell": "rose", "up": "down", "down": "up", "grew": "fell", "declined": "grew",
           "jumped": "fell", "dropped": "rose"}


def render_like(value_text, true, direction_text, qualifier):
    """Render the true value in the same style as the cited text (currency, k, %, x, decimals, sign)."""
    s, sign = value_text, ""
    if s[0] in "+-":
        sign, s = s[0], s[1:]
    pre = "£" if s.startswith("£") else ""
    s = s.lstrip("£")
    m = re.fullmatch(r"([\d,]+)(?:\.(\d+))?(k|x|%|st|nd|rd|th)?", s)
    dec, suf = len(m.group(2) or ""), m.group(3) or ""
    scale = Decimal(1000) if suf == "k" else Decimal(1)
    v = Decimal(repr(float(true))) / scale
    if not sign:
        v = abs(v)
    unit = Decimal(1).scaleb(-dec)
    if qualifier == "over":  # a true lower bound at the cited leading digit
        lead = Decimal(10) ** max(len(m.group(1).replace(",", "")) - 2, 0)  # keep two significant digits
        q = (abs(v) / lead).to_integral_value(rounding=ROUND_FLOOR) * lead
        if q == abs(v):
            q -= lead
    else:
        q = abs(v).quantize(unit, rounding=ROUND_HALF_UP)
    body = f"{q:,.{dec}f}" if "," in m.group(1) or (pre and suf != "k") else f"{q:.{dec}f}"
    new_sign = "" if not sign else ("-" if v < 0 else "+")
    return f"{new_sign}{pre}{body}{suf}"


def caught_check(results, notes):
    """For each caught wrong value, put the correct value (or direction) in its place and rerun the gate."""
    out = []
    for r in results:
        if r["outcome"] != "caught":
            continue
        n = notes[r["note_id"]]
        lines = n["text"].split("\n")
        li, s, e = int(r["line"]) - 1, int(r["start"]), int(r["end"])
        line = lines[li]
        if r["error_type"] == "direction":
            if r["direction_text"] == "sign":
                vt = r["value_text"]
                fixed_num = ("-" if vt[0] == "+" else "+") + vt[1:]
                new_line = line[:s] + fixed_num + line[e:]
            else:
                mm = re.search(r"(\w+)\s+$", line[:s])
                w = mm.group(1)
                new_line = line[:mm.start(1)] + ANTONYM[w] + line[mm.end(1):]
                s += len(ANTONYM[w]) - len(w)
                fixed_num = r["value_text"]
        else:
            fixed_num = render_like(r["value_text"], float(r["true_value"]), r["direction_text"], r["qualifier"])
            new_line = line[:s] + fixed_num + line[e:]
        lines2 = list(lines)
        lines2[li] = new_line
        table = pd.read_csv(ROOT / "data" / "tables" / f"{n['table_id']}.csv")
        recs = [x for x in ng.gate_level1("\n".join(lines2), table) if x["line_idx"] == li and x["position"] == s]
        lab = recs[0]["label"] if recs else "not_extracted"
        out.append(dict(note_id=r["note_id"], line=r["line"], value_text=r["value_text"], error_type=r["error_type"],
                        corrected_text=fixed_num if r["error_type"] != "direction" else new_line[max(0, s - 12):s + len(fixed_num)],
                        gate_label_on_correct_value=lab,
                        caught_for_right_reason=lab.startswith("Supported"),
                        evidence_on_correct_value=recs[0]["evidence"] if recs else ""))
    return out


def main():
    frozen = json.loads((OUT / "FROZEN.json").read_text())
    for f, k in [("notes.jsonl", "notes_sha256"), ("truth.csv", "truth_sha256")]:
        assert hashlib.sha256((OUT / f).read_bytes()).hexdigest() == frozen[k], f"{f} changed after freeze"
    notes = {json.loads(l)["note_id"]: json.loads(l) for l in open(OUT / "notes.jsonl")}
    results = list(csv.DictReader(open(OUT / "results.csv")))

    diag = []
    for r in results:
        oc = r["outcome"]
        if oc not in ("false_alarm", "nm_flagged", "miss"):
            continue
        key = (r["note_id"], int(r["line"]), r["value_text"])
        if oc in ("false_alarm", "nm_flagged"):
            cause, cls, s7, detail = FA_CAUSES[key]
        else:
            cause, cls, s7, detail = classify_miss(r)
        line_text = notes[r["note_id"]]["text"].split("\n")[int(r["line"]) - 1]
        diag.append(dict(note_id=r["note_id"], table_id=r["table_id"], line=r["line"], value_text=r["value_text"],
                         truth_label=r["label"], error_type=r["error_type"], true_value=r["true_value"],
                         outcome=oc, gate_label=r["gate_label"], gate_evidence=r["gate_evidence"],
                         cause=cause, diagnosis_class=cls, section7=s7, detail=detail, line_text=line_text))

    cc = caught_check(results, notes)
    for c in cc:
        if c["caught_for_right_reason"]:
            continue
        key = (c["note_id"], int(c["line"]), c["value_text"])
        cause, cls, s7, detail = LATENT_CAUSES[key]
        r = next(x for x in results if (x["note_id"], int(x["line"]), x["value_text"]) == key)
        diag.append(dict(note_id=r["note_id"], table_id=r["table_id"], line=r["line"], value_text=r["value_text"],
                         truth_label=r["label"], error_type=r["error_type"], true_value=r["true_value"],
                         outcome="caught_but_correct_value_also_unsupported", gate_label=r["gate_label"],
                         gate_evidence=f"correct value {c['corrected_text']} -> {c['gate_label_on_correct_value']}",
                         cause=cause, diagnosis_class=cls, section7=s7, detail=detail,
                         line_text=notes[r["note_id"]]["text"].split("\n")[int(r["line"]) - 1]))
    assert len({(d["note_id"], d["line"], d["value_text"], d["outcome"]) for d in diag}) == len(diag)
    with open(OUT / "diagnosed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(diag[0].keys()))
        w.writeheader()
        w.writerows(diag)
    with open(OUT / "caught_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cc[0].keys()))
        w.writeheader()
        w.writerows(cc)

    fa = [d for d in diag if d["outcome"] == "false_alarm"]
    ms = [d for d in diag if d["outcome"] == "miss"]
    fa_cause_counts = Counter()
    for d in fa:
        for c in d["cause"].split(" + "):
            fa_cause_counts[c] += 1
    m = json.loads((OUT / "metrics.json").read_text())
    m.update({
        "misses_by_design": sum(d["diagnosis_class"] == "by_design" for d in ms),
        "misses_rule_gaps": sum(d["diagnosis_class"] == "rule_gap" for d in ms),
        "misses_by_cause": dict(Counter(d["cause"] for d in ms)),
        "misses_by_design_strict_cells_or_coincidence": sum(d["cause"] in (
            "real_cell_of_another_row", "real_cell_of_another_period_same_row", "coincidence_with_unrelated_cell",
            "coincidence_with_unrelated_derived_value") for d in ms),
        "false_alarm_causes_by_mention": dict(fa_cause_counts),
        "false_alarms_by_class": dict(Counter(d["diagnosis_class"] for d in fa)),
        "false_alarm_mentions_with_two_causes": sum(" + " in d["cause"] for d in fa),
        "non_measure_flagged_causes": dict(Counter(d["cause"] for d in diag if d["outcome"] == "nm_flagged")),
        "notes_with_a_false_alarm": len({d["note_id"] for d in fa}),
        "notes_with_a_false_alarm_or_flagged_count": len({d["note_id"] for d in diag if d["outcome"] in ("false_alarm", "nm_flagged")}),
        "caught_for_right_reason": sum(c["caught_for_right_reason"] for c in cc),
        "caught_but_correct_value_also_not_supported": [f"{c['note_id']} L{c['line']} {c['value_text']} -> {c['corrected_text']}: {c['gate_label_on_correct_value']}"
                                                        for c in cc if not c["caught_for_right_reason"]],
    })
    W = m["wrong"]
    value_types = ("drift", "rounding", "group_total", "direction")
    wv = [r for r in results if r["label"] == "W" and r["error_type"] in value_types and r["is_word"] == "0"]
    wb = [r for r in results if r["label"] == "W" and r["error_type"] in ("other_row", "other_period") and r["is_word"] == "0"]
    m["catch_rate_value_errors"] = {"types": list(value_types), "caught": sum(r["outcome"] == "caught" for r in wv),
                                    "n": len(wv), "rate": round(sum(r["outcome"] == "caught" for r in wv) / len(wv), 4)}
    m["catch_rate_binding_errors"] = {"types": ["other_row", "other_period"], "caught": sum(r["outcome"] == "caught" for r in wb),
                                      "n": len(wb), "rate": round(sum(r["outcome"] == "caught" for r in wb) / len(wb), 4)}
    m["miss_rate_rule_gaps_of_wrong"] = round(m["misses_rule_gaps"] / W, 4)
    lat = [d for d in diag if d["outcome"] == "caught_but_correct_value_also_unsupported"]
    m["latent_false_alarms_on_corrected_values"] = len(lat)
    m["latent_false_alarm_causes"] = dict(Counter(d["cause"] for d in lat))
    all_fa = Counter()
    for d in fa + lat:
        for c in d["cause"].split(" + "):
            all_fa[c] += 1
    m["false_alarm_causes_including_latent"] = dict(all_fa)
    m["wilson95"] = {
        "false_alarm_rate_of_correct": wilson(m["false_alarms"], m["correct"]),
        "catch_rate_of_wrong": wilson(m["caught"], W),
        "catch_rate_value_errors": wilson(m["catch_rate_value_errors"]["caught"], m["catch_rate_value_errors"]["n"]),
        "catch_rate_binding_errors": wilson(m["catch_rate_binding_errors"]["caught"], m["catch_rate_binding_errors"]["n"]),
        "flag_precision": wilson(m["caught"], m["caught"] + m["false_alarms"] + m["non_measures_flagged"]),
    }
    notes_ids = sorted(notes)
    by_note = {n: [r for r in results if r["note_id"] == n and r["is_word"] == "0"] for n in notes_ids}
    m["note_level"] = {
        "notes": len(notes_ids),
        "notes_with_any_flag_on_correct_content": sum(any(r["outcome"] in ("false_alarm", "nm_flagged") for r in rs) for rs in by_note.values()),
        "notes_with_a_miss": sum(any(r["outcome"] == "miss" for r in rs) for rs in by_note.values()),
        "notes_with_every_planted_error_caught": sum(all(r["outcome"] == "caught" for r in rs if r["label"] == "W") for rs in by_note.values()),
        "notes_with_a_miss_excluding_binding_errors": sum(any(r["outcome"] == "miss" and r["error_type"] not in ("other_row", "other_period") for r in rs) for rs in by_note.values()),
    }
    (OUT / "metrics.json").write_text(json.dumps(m, indent=2))
    keys = ["misses_by_design", "misses_rule_gaps", "misses_by_cause", "misses_by_design_strict_cells_or_coincidence",
            "false_alarm_causes_by_mention", "false_alarms_by_class", "non_measure_flagged_causes",
            "notes_with_a_false_alarm", "caught_for_right_reason", "caught_but_correct_value_also_not_supported",
            "catch_rate_value_errors", "catch_rate_binding_errors", "latent_false_alarm_causes",
            "false_alarm_causes_including_latent", "wilson95", "note_level"]
    print(json.dumps({k: m[k] for k in keys}, indent=1))


if __name__ == "__main__":
    main()
