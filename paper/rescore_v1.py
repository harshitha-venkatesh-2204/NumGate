"""Re-score the Opus smoke notes with the original (v1) gate and compare with the current (v2) gate.

Usage: python paper/rescore_v1.py   (no API calls: v1 extractor replies are read from cache/)
Writes paper/data/v1_v2_notes.csv and paper/data/v1_flags.csv.
"""
import importlib.util, json, glob, re, sys, os
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v1")
sys.path.insert(0, os.path.join(ROOT, "src"))
import models, number_gate as v2
spec = importlib.util.spec_from_file_location("gate_v1", os.path.join(V1, "number_gate_v1.py"))
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)
old_sys = open(os.path.join(V1, "extract_system_v1.txt"), encoding="utf-8").read()
user_t = open(os.path.join(ROOT, "prompts", "extract_user.txt"), encoding="utf-8").read()  # unchanged since v1


def cause(r):
    """Why v1 flagged this claim, from its evidence and the claim fields (used for the paper's error analysis)."""
    ev, ent, per = r.v1_evidence, str(r.entity), str(r.period)
    if "direction wrong" in ev and per in ("TW", "LW", "LY"):
        return "direction word applied to a level"
    if ent.upper() == "TOTAL" or ent == "":
        return "group or top-k figure read as the whole-table total"
    m = re.search(r"value is .*?\.(\w+)=.*claimed (\w+)", ev)
    if m and re.match(r"(Revenue|Units|Orders)_(LW|LY|TW)$", m.group(1)) and m.group(2).endswith("_Abs"):
        return "level next to an LW/LY label read as a change"
    if m and {m.group(1).split("_")[0], m.group(2).split("_")[0]} <= {"Units", "Orders"} and m.group(1).split("_")[0] != m.group(2).split("_")[0]:
        return "unit noun ignored (units vs orders)"
    if r.v1_label == "Wrong_entity" and "belongs to" in ev:
        return "value given to the wrong entity in a list"
    if "family" in str(r.line).lower() or re.search(r"\b(combined|colourways|lines)\b", str(r.line).lower()):
        return "family total bound to one member"
    return "other extractor misreading"


rows = []
summ = []
for f in sorted(glob.glob(os.path.join(ROOT, "runs/smoke_opus/notes/*.json"))):
    rec = json.load(open(f)); note = rec["note"]
    table = pd.read_csv(os.path.join(ROOT, "data/tables", rec["table_id"] + ".csv"))
    lines = [l for l in note.splitlines() if l.strip()]
    user = user_t.format(note_lines="\n".join(f"{i}: {l}" for i, l in enumerate(lines)))
    path = os.path.join(ROOT, "cache", "claude-haiku-4-5", models.cache_key("claude-haiku-4-5", old_sys, user, 0, "") + ".json")
    v1_src = "old_llm"
    claims = None
    if os.path.exists(path):
        text = json.load(open(path))["text"]
        try:
            claims = json.loads(text[text.find("{"): text.rfind("}") + 1])["claims"]
            for c in claims: c["line_idx"] = int(c.get("line", 0) or 0)
        except Exception:
            claims = None
    if claims is None:  # v1 fell back to its regex extractor when the reply was missing or cut off
        claims, v1_src = v1.regex_claims(note, table), "v1_regex_fallback"
    l1_v1 = v1.gate_level1(note, table, rec["note_id"]); l2_v1 = v1.gate_level2(note, table, rec["note_id"], claims)
    l1_v2 = v2.gate_level1(note, table, rec["note_id"])
    s1, s2 = v1.summarize(l1_v1, l2_v1), v2.summarize(l1_v2, [])
    summ.append({"note_id": rec["note_id"], "strategy": rec["strategy"], "v1_src": v1_src,
                 "v1_unsup": s1["n_Unsupported"], "v1_checked": s1["n_numbers_checked"],
                 "v1_claim_err": s1["n_Wrong_entity"] + s1["n_Wrong_metric"] + s1["n_Wrong_value"], "v1_bind": s1["n_Wrong_entity"] + s1["n_Wrong_metric"],
                 "v1_ver": s1["n_claims_verifiable"], "v2_unsup": s2["n_Unsupported"], "v2_checked": s2["n_numbers_checked"]})
    # same claims, re-verified by v2 (isolates verification changes from extractor-prompt changes)
    l2_v2_same = v2.gate_level2(note, table, rec["note_id"], [dict(c) for c in claims]) if v1_src == "old_llm" else []
    for a, b in zip(l2_v1, l2_v2_same or [None] * len(l2_v1)):
        if a["label"].startswith("Wrong"):
            rows.append({"note_id": rec["note_id"], "strategy": rec["strategy"], "line": lines[a["line_idx"]] if a["line_idx"] < len(lines) else "",
                         "entity": a["entity"], "metric": a["metric"], "period": a["period"], "value_text": a["value_text"],
                         "v1_label": a["label"], "v1_evidence": a["evidence"], "v2_same_claims": b["label"] if b else "",
                         "v2_evidence": b["evidence"] if b else "", "v2_adjusted": b["adjusted"] if b else ""})
S = pd.DataFrame(summ); R = pd.DataFrame(rows).fillna("")
R["cause"] = R.apply(cause, axis=1)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
S.to_csv(os.path.join(OUT, "v1_v2_notes.csv"), index=False); R.to_csv(os.path.join(OUT, "v1_flags.csv"), index=False)
g = S.groupby("strategy")
print(pd.DataFrame({"v1_L1": g.v1_unsup.sum() / g.v1_checked.sum(), "v2_L1": g.v2_unsup.sum() / g.v2_checked.sum(),
                    "v1_claim_err": g.v1_claim_err.sum() / g.v1_ver.sum(), "v1_binding": g.v1_bind.sum() / g.v1_ver.sum(),
                    "v1_flags": g.v1_claim_err.sum(), "v1_regex_notes": g.v1_src.apply(lambda s: (s != "old_llm").sum())}).round(4).to_string())
print("total v1 claim flags:", len(R), " L1 flags v1:", S.v1_unsup.sum(), " v2:", S.v2_unsup.sum())
print(R.cause.value_counts().to_string())
