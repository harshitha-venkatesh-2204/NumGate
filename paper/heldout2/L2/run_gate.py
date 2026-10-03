"""Run the frozen Level 2 gate on the frozen heldout2/L2 notes and score it against truth.csv.

Order of operations (held-out protocol): FROZEN.json checksums are verified, the md5 of
src/number_gate.py is verified, then the gate runs. Nothing in notes.json or truth.csv is changed.

Adapter: the claims in notes.json number lines from 1 (line 1 = first bullet). The pipeline's
extractor prompt numbers lines from 0 and llm_claims() maps "line" -> "line_idx", so here
line_idx = line - 1. No other field is touched.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

import pandas as pd

ROOT = str(Path(__file__).resolve().parents[3])
OUT = os.path.join(ROOT, "paper", "heldout2", "L2")
GATE_MD5 = "bcdb434ba30ad334a702493668a0ce39"
sys.path.insert(0, os.path.join(ROOT, "src"))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


frozen = json.load(open(os.path.join(OUT, "FROZEN.json")))
for f in ("notes.json", "truth.csv"):
    assert sha(os.path.join(OUT, f)) == frozen["files"][f], f"{f} changed after freezing"
md5 = hashlib.md5(open(os.path.join(ROOT, "src", "number_gate.py"), "rb").read()).hexdigest()
if md5 != GATE_MD5:
    sys.exit(f"STOP: src/number_gate.py md5 is {md5}, expected {GATE_MD5}")

import number_gate  # noqa: E402  (imported only after the md5 check)

notes = json.load(open(os.path.join(OUT, "notes.json"), encoding="utf-8"))
truth = pd.read_csv(os.path.join(OUT, "truth.csv"), keep_default_na=False)

recs = []
for n in notes:
    table = pd.read_csv(os.path.join(ROOT, "data", "tables", n["table_id"] + ".csv"))
    claims = [dict(c, line_idx=c["line"] - 1) for c in n["claims"]]
    out = number_gate.gate_level2(n["text"], table, n["note_id"], claims)
    assert len(out) == len(claims)
    for i, (c, r) in enumerate(zip(n["claims"], out)):
        assert r["value_text"] == c["value_text"] and r["line_idx"] == c["line"] - 1
        recs.append(dict(note_id=n["note_id"], claim_idx=i, gate_label=r["label"], gate_evidence=r["evidence"],
                         gate_adjusted=r["adjusted"], gate_metric=r["metric"], gate_period=r["period"],
                         gate_qualifier=r["qualifier"]))

res = truth.merge(pd.DataFrame(recs), on=["note_id", "claim_idx"], how="left", validate="one_to_one")
assert res.gate_label.notna().all()

WRONG_TRUTH = ["Wrong_entity", "Wrong_metric", "Wrong_value", "Wrong_direction"]
is_err = res.gate_label.str.startswith("Wrong")


def outcome(r):
    t, g = r.truth, r.gate_label
    err = g.startswith("Wrong")
    if t == "Correct":
        return "false_alarm" if err else "coverage_loss" if g == "Unverifiable" else "verified"
    if t in WRONG_TRUTH:
        return "detected" if err else "miss" if g == "Correct" else "hidden"
    if t == "Group_correct":
        return "group_flagged" if err else "group_verified" if g == "Correct" else "group_coverage_loss"
    if t == "Group_wrong":
        return "group_wrong_flagged" if err else "group_wrong_missed" if g == "Correct" else "group_wrong_hidden"
    if t == "Unverifiable":
        return "unverifiable_ok" if g == "Unverifiable" else "unverifiable_mislabelled"
    raise ValueError(t)


res["outcome"] = res.apply(outcome, axis=1)
res["type_agrees"] = ""
det = res.outcome == "detected"
res.loc[det, "type_agrees"] = (res.loc[det, "gate_label"] == res.loc[det, "expected_gate"]).map({True: "yes", False: "no"})
res.to_csv(os.path.join(OUT, "results.csv"), index=False)


def n(mask):
    return int(mask.sum())


correct = res.truth == "Correct"
wrong = res.truth.isin(WRONG_TRUTH)
gcor, gwr, unv = res.truth == "Group_correct", res.truth == "Group_wrong", res.truth == "Unverifiable"
m = {
    "claims": len(res), "notes": len(notes),
    "correct": n(correct),
    "false_alarms": n(correct & is_err),
    "coverage_losses": n(correct & (res.gate_label == "Unverifiable")),
    "correct_verified": n(correct & (res.gate_label == "Correct")),
    "wrong": n(wrong),
    "detected": n(wrong & is_err),
    "misses": n(wrong & (res.gate_label == "Correct")),
    "hidden": n(wrong & (res.gate_label == "Unverifiable")),
    "type_agreement": n(det & (res.gate_label == res.expected_gate)),
    "group_correct": n(gcor),
    "group_correct_verified": n(gcor & (res.gate_label == "Correct")),
    "group_correct_flagged": n(gcor & is_err),
    "group_correct_unverifiable": n(gcor & (res.gate_label == "Unverifiable")),
    "group_wrong": n(gwr),
    "group_wrong_flagged": n(gwr & is_err),
    "group_wrong_missed": n(gwr & (res.gate_label == "Correct")),
    "unverifiable": n(unv),
    "unverifiable_ok": n(unv & (res.gate_label == "Unverifiable")),
}
m["false_alarm_rate"] = round(m["false_alarms"] / m["correct"], 4)
m["coverage_loss_rate"] = round(m["coverage_losses"] / m["correct"], 4)
m["detection_rate"] = round(m["detected"] / m["wrong"], 4)
m["miss_rate"] = round(m["misses"] / m["wrong"], 4)
m["type_agreement_rate"] = round(m["type_agreement"] / m["detected"], 4) if m["detected"] else None
flag_any = is_err
m["flag_precision"] = round(n(flag_any & (wrong | gwr)) / n(flag_any), 4)
m["false_alarm_rate_excluding_per_order"] = {
    "correct_claims": n(correct & ~res.true_ref.str.contains("derived aov")),
    "false_alarms": n(correct & is_err & ~res.true_ref.str.contains("derived aov"))}
# note level: what a reader of one note sees
ok_claim = res.truth.isin(["Correct", "Group_correct"])
bad_claim = res.truth.isin(WRONG_TRUTH + ["Group_wrong"])
per_note = res.groupby("note_id").apply(lambda g: pd.Series({
    "false_flag": bool((g.truth.isin(["Correct", "Group_correct"]) & g.gate_label.str.startswith("Wrong")).any()),
    "has_error": bool(g.truth.isin(WRONG_TRUTH + ["Group_wrong"]).any()),
    "all_errors_flagged": bool((~g.truth.isin(WRONG_TRUTH + ["Group_wrong"]) | g.gate_label.str.startswith("Wrong")).all())}),
    include_groups=False)
m["note_level"] = {"notes": len(per_note), "notes_with_a_wrongly_flagged_correct_claim": int(per_note.false_flag.sum()),
                   "notes_with_planted_errors": int(per_note.has_error.sum()),
                   "notes_with_every_planted_error_flagged": int((per_note.has_error & per_note.all_errors_flagged).sum())}
# 95% percentile bootstrap over notes (1,000 resamples), as in docs/method.md section 5
rng = __import__("numpy").random.default_rng(0)
ids = res.note_id.unique()
boot = {"false_alarm_rate": [], "detection_rate": [], "flag_precision": []}
for _ in range(1000):
    s = pd.concat([res[res.note_id == i] for i in rng.choice(ids, len(ids))])
    e = s.gate_label.str.startswith("Wrong")
    c_, w_ = s.truth == "Correct", s.truth.isin(WRONG_TRUTH)
    boot["false_alarm_rate"].append((c_ & e).sum() / c_.sum())
    boot["detection_rate"].append((w_ & e).sum() / w_.sum())
    boot["flag_precision"].append((e & s.truth.isin(WRONG_TRUTH + ["Group_wrong"])).sum() / e.sum())
m["bootstrap_95ci_by_note"] = {k: [round(float(pd.Series(v).quantile(0.025)), 4), round(float(pd.Series(v).quantile(0.975)), 4)]
                               for k, v in boot.items()}
m["by_truth"] = {t: res[res.truth == t].gate_label.value_counts().to_dict() for t in res.truth.unique()}
m["detection_by_error_mode"] = {k: {"n": len(g), "detected": n(g.gate_label.str.startswith("Wrong")),
                                    "type_agrees": n(g.type_agrees == "yes")}
                                for k, g in res[wrong].groupby("error_mode")}
m["false_alarms_by_kind"] = res[correct & is_err].gate_label.value_counts().to_dict()
json.dump(m, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in m.items() if not isinstance(v, dict)}, indent=1))
print(json.dumps(m["by_truth"], indent=1))
print(json.dumps(m["detection_by_error_mode"], indent=1))
