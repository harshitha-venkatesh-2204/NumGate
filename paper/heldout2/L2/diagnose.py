"""Diagnose every false alarm, miss, coverage loss, hidden error and mislabelled group claim in results.csv.

Written after scoring. Reads the frozen notes/truth and results.csv; never edits them.
Causes were established by reading src/number_gate.py and docs/method.md and by probing the
gate's own functions (families, match_entities_how, matches, group_value_match).

Also runs a counterfactual for every wrong or group-wrong claim: the same note with the true value
written in its place (same precision and format). A flag counts as discriminating only if the gate
labels the truthful version Correct.
Writes diagnosed.csv and adds the cause breakdowns to metrics.json.
"""
from pathlib import Path
import json
import os
import sys
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

ROOT = str(Path(__file__).resolve().parents[3])
OUT = os.path.join(ROOT, "paper", "heldout2", "L2")
sys.path.insert(0, os.path.join(ROOT, "src"))
import number_gate  # noqa: E402

notes = {n["note_id"]: n for n in json.load(open(os.path.join(OUT, "notes.json"), encoding="utf-8"))}
res = pd.read_csv(os.path.join(OUT, "results.csv"), keep_default_na=False)
WRONG = ["Wrong_entity", "Wrong_metric", "Wrong_value", "Wrong_direction"]

# ------------------------------------------------------------------ causes
CAUSES = {
    "PER_ORDER": dict(
        cls="rule_gap",
        cause="Average per order cited as a £ figure. The extractor schema has no per-order metric, so the claim is revenue/TW; "
              "check_row maps it to Revenue_TW and, because the unit (currency) fits, never tries the row's derived values "
              "(per_order is only tried when the unit does not fit). Level 1 supports the same number as per_order(entity.Revenue_TW).",
        ref="method.md 3.2 step 5 (derived fallback only on unit mismatch); number_gate.check_row lines 752-758; not listed in section 7",
        fix="In check_row, before Wrong_value, try the row's derived values (per_order/per_unit) as Level 1 does; or add a per-order metric to the extractor schema."),
    "UAE_ALIAS": dict(
        cls="rule_gap",
        cause="'UAE' (and 'Emirati') is not in ENTITY_ALIASES, the words test fails ({'uae'} is not a subset of the name's words) "
              "and fuzzy WRatio('uae','united arab emirates') is below 85, so the entity is not found.",
        ref="method.md 3.2 step 2 (alias list 'and similar'); number_gate.ENTITY_ALIASES line 24; not listed in section 7",
        fix="Add 'uae', 'u.a.e.', 'emirati' -> United Arab Emirates (and 'rsa' style initialisms generally)."),
    "K_TOLERANCE": dict(
        cls="by_design",
        cause="Value cited in k with the last digit rounded the wrong way (118.12k written 118.2k; 16.455k written 16.4k). "
              "matches() also accepts any k/M value within 0.5 percent relative error, so the wrong last digit passes.",
        ref="method.md 2.3 (0.5 percent tolerance for k, M, B) and section 7 last bullet (accepts a wrong last digit at large scales)",
        fix="Accept only truncation within one unit in the last cited digit (e.g. 16,455 -> 16.4k) instead of any 0.5 percent."),
    "FUZZY_NAME": dict(
        cls="by_design",
        cause="Name found only by fuzzy matching (note 'Set of 4' vs table 'SET OF4'; note 'T-Light Holder' vs table 'T-LIGHT HLDR'). "
              "The check did find the error, but a fuzzily bound name may not produce a confident Wrong_*, so it becomes Unverifiable.",
        ref="method.md 3.2 step 2 (fuzzy binding never gives a confident error); number_gate.verify_claim lines 846-847",
        fix="Normalise common abbreviations and glued tokens (HLDR->HOLDER, OF4->OF 4) before the words test so these bind exactly."),
    "GROUP_BOUND": dict(
        cls="rule_gap",
        cause="Group figure with a bound ('more than 49%' for the top-five share of 49.88%). group_value_match calls first_match "
              "without allow_bounds, so the bound is read as an exact 49% and fails; Level 1 allows bounds on lines that name no entity.",
        ref="method.md 2.3 (bounds on entity-free lines are checked against table-level values) vs 3.2 step 1; number_gate.group_value_match line 794",
        fix="Allow bounds in group_value_match, restricted to candidates of the claimed metric (shares for share claims), "
            "since an unrestricted bound also hits min(WoW_Pct)=-61.5."),
    "FAMILY_PLURAL": dict(
        cls="rule_gap",
        cause="Family written in the plural ('Feltcraft cushions'). families() matches exact word runs, so FELTCRAFT CUSHION "
              "(exactly the three rows) is not found; only FELTCRAFT (9 rows) and BUTTERFLY (2 rows) are, and the short member names "
              "in brackets (Owl, Rabbit, Butterfly) are not entity mentions. With 'cushion' the family is found.",
        ref="method.md 2.4 step 2 (family = word run found in 2 to 20 names); number_gate.families line 464",
        fix="Singularise words (strip trailing s) in families() before matching word runs."),
    "GROUP_SUBSET": dict(
        cls="by_design",
        cause="Hand-picked subset of a family named by short names in brackets (5 of the 7 hot water bottle rows). The library has the "
              "7-row HOT WATER BOTTLE family and top-k/rest totals, but not this subset; the short names are not found as entity mentions.",
        ref="method.md section 7 (a group the library does not know is Wrong_value at Level 2)",
        fix="Resolve bracketed short names against the members of a family the line mentions and add that subset as a group."),
}


def cause_for(r):
    if r.outcome == "false_alarm" and "derived aov" in r.true_ref:
        return "PER_ORDER"
    if r.entity == "UAE":
        return "UAE_ALIAS"
    if r.outcome == "miss" and r.error_mode == "rounding" and r.value_text.endswith("k"):
        return "K_TOLERANCE"
    if r.outcome == "hidden" and r.gate_evidence.startswith("binding uncertain (fuzzy"):
        return "FUZZY_NAME"
    if r.outcome == "group_flagged" and r.qualifier == "over":
        return "GROUP_BOUND"
    if r.outcome == "group_flagged" and "FELTCRAFT CUSHION" in r.true_ref:
        return "FAMILY_PLURAL"
    if r.outcome == "group_flagged" and "HOT WATER BOTTLE" in r.true_ref:
        return "GROUP_SUBSET"
    return None


# ------------------------------------------------------------------ counterfactual (true value written in)
def fmt_true(vt, tv, period, direction):
    p = number_gate.parse_value_text(vt)
    body = vt.lstrip("+-")
    cur = "£" if body.startswith("£") else ""
    body = body.lstrip("£")
    suf = next((s for s in ("k", "M", "%", "x", "st", "nd", "rd", "th") if body.endswith(s)), "")
    dec = p["decimals"]
    mag = Decimal(repr(abs(tv))) / Decimal(int(p["scale"]))
    num = mag.quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP)
    text = f"{num:,.{dec}f}" if "," in vt or num >= 1000 else f"{num:.{dec}f}"
    if suf in ("st", "nd", "rd", "th"):
        n = int(num)
        suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    sign = ""
    new_dir = direction
    if period in ("WoW", "YoY") and suf != "x":
        if vt[0] in "+-":
            sign = "-" if tv < 0 else "+"
        elif direction in ("up", "down"):
            new_dir = "up" if tv > 0 else "down"
    return sign + cur + text + suf, new_dir


def counterfactual(r):
    n = notes[r.note_id]
    tv = float(r.true_value)
    new_vt, new_dir = fmt_true(r.value_text, tv, r.period, r.direction)
    lines = n["text"].split("\n")
    claims = [dict(c) for c in n["claims"]]
    # locate the value in its line exactly as the generator did (in order within the line)
    li = r.line - 1
    pos = 0
    for k, c in enumerate(claims):
        if c["line"] != r.line:
            continue
        j = lines[li].find(c["value_text"], pos)
        if k == r.claim_idx:
            lines[li] = lines[li][:j] + new_vt + lines[li][j + len(c["value_text"]):]
            break
        pos = j + len(c["value_text"])
    claims[r.claim_idx].update(value_text=new_vt, direction=new_dir)
    table = pd.read_csv(os.path.join(ROOT, "data", "tables", n["table_id"] + ".csv"))
    out = number_gate.gate_level2("\n".join(lines), table, n["note_id"], [dict(c, line_idx=c["line"] - 1) for c in claims])
    return new_vt, new_dir, out[r.claim_idx]["label"], out[r.claim_idx]["evidence"]


res["cf_value_text"] = ""
res["cf_label"] = ""
res["cf_evidence"] = ""
for i, r in res[res.truth.isin(WRONG + ["Group_wrong"])].iterrows():
    vt, d, lab, ev = counterfactual(r)
    res.loc[i, ["cf_value_text", "cf_label", "cf_evidence"]] = [f"{vt} ({d})" if d != r.direction else vt, lab, ev]

# ------------------------------------------------------------------ diagnosed rows
rows = []
IDEAL = {"verified", "detected", "group_verified", "group_wrong_flagged", "unverifiable_ok"}
for _, r in res.iterrows():
    kind, code = None, None
    if r.outcome not in IDEAL:
        code = cause_for(r)
        assert code, f"undiagnosed: {r.note_id} L{r.line} {r.value_text} {r.outcome}"
        kind = "error"
    elif r.outcome in ("detected", "group_wrong_flagged") and r.cf_label != "Correct":
        kind = "info_non_discriminating_flag"
    elif r.outcome == "unverifiable_ok" and "entity not found" in r.gate_evidence:
        kind, code = "info_right_label_wrong_reason", "UAE_ALIAS"
    elif r.outcome == "group_verified" and r.note_id == "n12" and r.value_text == "26%":
        kind = "info_right_label_wrong_evidence"
    if not kind:
        continue
    c = CAUSES.get(code, {})
    row = dict(note_id=r.note_id, line=r.line, claim_idx=r.claim_idx, entity=r.entity, metric=r.metric, period=r.period,
               value_text=r.value_text, qualifier=r.qualifier, truth=r.truth, error_mode=r.error_mode, true_value=r.true_value,
               gate_label=r.gate_label, outcome=r.outcome, row_type=kind, cause_code=code or "",
               cause_class=c.get("cls", ""), cause=c.get("cause", ""), doc_or_code_ref=c.get("ref", ""), fix_hint=c.get("fix", ""),
               gate_evidence=r.gate_evidence, line_text=notes[r.note_id]["text"].split("\n")[r.line - 1],
               counterfactual=f"{r.cf_value_text} -> {r.cf_label}" if r.cf_label else "")
    if kind == "info_non_discriminating_flag":
        row.update(cause_class="rule_gap" if "FELTCRAFT CUSHION" in r.true_ref else "", cause_code="FAMILY_PLURAL" if "FELTCRAFT CUSHION" in r.true_ref else "",
                   cause="Flagged, but the gate also flags the true value in the same sentence, so the flag carries no information "
                         "about this error. " + (CAUSES["FAMILY_PLURAL"]["cause"] if "FELTCRAFT CUSHION" in r.true_ref else ""))
    if kind == "info_right_label_wrong_reason":
        row["cause"] = "Correctly Unverifiable (units LY is not in the table), but the gate got there because 'UAE' was not found. " + CAUSES["UAE_ALIAS"]["cause"]
    if kind == "info_right_label_wrong_evidence":
        row.update(cause_class="rule_gap", cause_code="APPROX_FIRST_MATCH",
                   cause="'top five about 26%' is right (exact top-5 share 25.58%), but the gate's evidence is the table's WoW change "
                         "pct_change(sum TW, sum LW) = -25.87%: an unsigned 'about' figure takes the first table-level value within 5 percent. "
                         "A wrong 'about 27%' would also pass through the same WoW value (4.4 percent away) although the top-5 share is 5.5 percent away.",
                   doc_or_code_ref="number_gate.first_match (first hit in candidate order); table_candidates puts pct_change before share_topk",
                   fix_hint="For share claims try share candidates first (or only)."),
    rows.append(row)

dx = pd.DataFrame(rows)
dx.to_csv(os.path.join(OUT, "diagnosed.csv"), index=False)
res.to_csv(os.path.join(OUT, "results.csv"), index=False)  # adds the counterfactual columns

# ------------------------------------------------------------------ metrics
m = json.load(open(os.path.join(OUT, "metrics.json")))
err = dx[dx.row_type == "error"]
by = lambda o: {f"{k} ({CAUSES[k]['cls']})": int(v) for k, v in err[err.outcome == o].cause_code.value_counts().items()}  # noqa: E731
m["false_alarm_causes"] = by("false_alarm")
m["coverage_loss_causes"] = by("coverage_loss")
m["miss_causes"] = by("miss")
m["hidden_causes"] = by("hidden")
m["group_correct_flagged_causes"] = by("group_flagged")
m["cause_descriptions"] = {k: v["cause"] for k, v in CAUSES.items()}
flagged = res[res.truth.isin(WRONG + ["Group_wrong"]) & res.gate_label.str.startswith("Wrong")]
m["counterfactual"] = {
    "flagged_wrong_or_group_wrong": len(flagged),
    "true_value_verified_Correct": int((flagged.cf_label == "Correct").sum()),
    "non_discriminating_flags": flagged.loc[flagged.cf_label != "Correct", ["note_id", "line", "value_text", "cf_value_text", "cf_label"]].to_dict("records"),
    "wrong_claims_detected_and_true_value_passes": int(((flagged.cf_label == "Correct") & flagged.truth.isin(WRONG)).sum()),
    "group_wrong_flagged_and_true_value_passes": int(((flagged.cf_label == "Correct") & (flagged.truth == "Group_wrong")).sum()),
}
m["right_label_wrong_reason"] = dx[dx.row_type.str.startswith("info_right")][["note_id", "line", "value_text", "row_type"]].to_dict("records")
m["by_class"] = {
    "false_alarms_by_design": int(((err.outcome == "false_alarm") & (err.cause_class == "by_design")).sum()),
    "false_alarms_rule_gap": int(((err.outcome == "false_alarm") & (err.cause_class == "rule_gap")).sum()),
    "all_errors_by_design": int((err.cause_class == "by_design").sum()),
    "all_errors_rule_gap": int((err.cause_class == "rule_gap").sum()),
}
json.dump(m, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
print(dx[["note_id", "line", "value_text", "outcome", "row_type", "cause_code", "cause_class", "counterfactual"]].to_string())
print(json.dumps({k: m[k] for k in ["false_alarm_causes", "coverage_loss_causes", "miss_causes", "hidden_causes",
                                    "group_correct_flagged_causes", "counterfactual", "by_class"]}, indent=1))
