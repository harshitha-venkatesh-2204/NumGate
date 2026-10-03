"""Generation strategies. Each takes a table and returns (note_text, trace).

S1 Direct, S2 Compute-then-write, S3 Write-then-verify, S4 Facts-template.
The trace records every prompt, raw response, code run, token count, latency and cost.
"""
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import number_gate
from config import load_config, read_prompt
from models import complete
from table_text import csv_head, money, pct, to_markdown

CFG = load_config()


def new_trace(strategy, model):
    return {"strategy": strategy, "model": model, "calls": [], "code_runs": [], "rounds": [],
            "failed": False, "error": ""}


def call(trace, step, model, system, user, repeat):
    """One model call, recorded in the trace."""
    resp = complete(model, system, user, temperature=0, max_tokens=CFG["max_tokens"], cache_tag=f"r{repeat}")
    trace["calls"].append({"step": step, "system": system, "user": user, **resp})
    return resp["text"]


def finish(trace):
    """Add per-note totals to the trace."""
    calls = trace["calls"]
    trace["n_calls"] = len(calls)
    trace["tokens_in"] = sum(c["input_tokens"] for c in calls)
    trace["tokens_out"] = sum(c["output_tokens"] for c in calls)
    trace["cost_usd"] = sum(c["cost_usd"] for c in calls)  # nominal cost, also counted when replayed from cache
    trace["latency_s"] = sum(c["latency_s"] for c in calls)
    return trace


def clean_note(text):
    """Keep the bullet lines only; drop any preamble a model adds despite the rules."""
    lines = [l.rstrip() for l in text.strip().splitlines() if l.strip()]
    bullets = [l for l in lines if re.match(r"^\s*[-*•]\s+", l)]
    return "\n".join(re.sub(r"^\s*[-*•]\s+", "- ", l) for l in (bullets or lines))


def user_fields(table, meta):
    return {"week": meta["week"], "entity_type": meta["entity_type"], "n_rows": len(table),
            "columns_guide": read_prompt("columns_guide.txt"), "table_markdown": to_markdown(table)}


def system_prompt(name):
    return read_prompt(name).format(note_rules=read_prompt("note_rules.txt"))

# ---------------------------------------------------------------- S1


def s1_direct(table, meta, model, repeat, trace=None):
    trace = trace or new_trace("S1", model)
    text = call(trace, "write", model, system_prompt("s1_system.txt"),
                read_prompt("s1_user.txt").format(**user_fields(table, meta)), repeat)
    return clean_note(text), finish(trace)

# ---------------------------------------------------------------- S2


def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    return m.group(1) if m else text


def clean_error(text, tmp, csv_path):
    """Replace machine-specific paths in an error message, so the retry prompt (and its cache key) is the same on every run."""
    for path, placeholder in [(tmp, "<tmpdir>"), (os.path.dirname(str(csv_path)), "<tables>"),
                              (sys.prefix, "<python>"), (sys.base_prefix, "<python>")]:
        for variant in sorted({str(path), os.path.realpath(str(path))}, key=len, reverse=True):  # longest first
            text = text.replace(variant, placeholder)
    return text


def run_facts_script(code, csv_path, timeout):
    """Run model-written code on the CSV in a subprocess. Returns (facts or None, error text, stdout)."""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "facts.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, str(script), str(csv_path)], cwd=tmp, capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, f"timed out after {timeout} seconds", ""
        stderr = clean_error(proc.stderr, tmp, csv_path)[-2000:]  # clean first, then cut, so the text is stable
    if proc.returncode != 0:
        return None, stderr, proc.stdout[-2000:]
    try:
        facts = json.loads(proc.stdout.strip())
        facts = facts.get("facts", facts) if isinstance(facts, dict) else facts
        if not isinstance(facts, list):
            return None, "stdout is not a JSON array of facts", proc.stdout[-2000:]
        facts = [f for f in facts if isinstance(f, dict) and isinstance(f.get("value"), (int, float))
                 and not isinstance(f.get("value"), bool) and math.isfinite(f["value"])][:25]
    except (ValueError, AttributeError, TypeError) as err:
        return None, f"stdout is not a JSON array of facts: {err}", proc.stdout[-2000:]
    if not facts:
        return None, "the script printed no usable facts", proc.stdout[-2000:]
    return facts, "", proc.stdout[-2000:]


def s2_compute_then_write(table, meta, model, repeat):
    trace = new_trace("S2", model)
    code_user = read_prompt("s2_code_user.txt").format(
        week=meta["week"], entity_type=meta["entity_type"], n_rows=len(table),
        schema="\n".join(f"- {c}: {t}" for c, t in table.dtypes.astype(str).items()),
        columns_guide=read_prompt("columns_guide.txt"), head_csv=csv_head(table))
    code_system = read_prompt("s2_code_system.txt")
    user = code_user
    facts = None
    for attempt in range(2):  # first try, then one retry with the error message
        code = extract_code(call(trace, f"code_{attempt}", model, code_system, user, repeat))
        facts, error, stdout = run_facts_script(code, meta["csv_path"], CFG["s2_timeout_seconds"])
        trace["code_runs"].append({"attempt": attempt, "code": code, "error": error, "stdout": stdout})
        if facts:
            break
        user = read_prompt("s2_code_retry_user.txt").format(original_request=code_user, code=code, error=error)
    if not facts:
        trace["failed"], trace["error"] = True, "facts script failed twice"
        return "", finish(trace)
    trace["facts"] = facts
    text = call(trace, "write", model, system_prompt("s2_write_system.txt"),
                read_prompt("s2_write_user.txt").format(week=meta["week"], entity_type=meta["entity_type"],
                                                        facts_json=json.dumps(facts, indent=1)), repeat)
    return clean_note(text), finish(trace)

# ---------------------------------------------------------------- S3


def flagged_numbers(note, table):
    records = number_gate.gate_level1(note, table)
    checked = [r for r in records if r["label"] != "Skipped"]
    flagged = [r for r in checked if r["label"] == "Unsupported"]
    return flagged, len(checked)


def s3_write_then_verify(table, meta, model, repeat):
    trace = new_trace("S3", model)
    note, _ = s1_direct(table, meta, model, repeat, trace)  # the draft is the S1 note for this repeat
    for rnd in range(CFG["s3_max_rounds"] + 1):
        flagged, n_checked = flagged_numbers(note, table)
        trace["rounds"].append({"round": rnd, "n_checked": n_checked, "n_unsupported": len(flagged),
                                "unsupported_rate": len(flagged) / n_checked if n_checked else None,
                                "note": note})
        if not flagged or rnd == CFG["s3_max_rounds"]:
            break
        flagged_text = "\n".join(f"- line {r['line_idx'] + 1}: \"{r['raw_text']}\" in: {r['sentence']}" for r in flagged)
        fields = dict(user_fields(table, meta), draft=note, flagged=flagged_text)
        note = clean_note(call(trace, f"revise_{rnd + 1}", model, system_prompt("s3_revise_system.txt"),
                               read_prompt("s3_revise_user.txt").format(**fields), repeat))
    return note, finish(trace)

# ---------------------------------------------------------------- S4


def facts_packet(table):
    """Deterministic facts, each number already formatted as it may appear in the note."""
    t = table.sort_values("Rank")
    tw, lw = t["Revenue_TW"].sum(), t["Revenue_LW"].sum()
    lines = [f"Total: revenue {money(tw)} TW, revenue {money(lw)} LW, revenue change {pct((tw - lw) / lw * 100, True)} WoW"]
    if t["Revenue_LY"].notna().any() and t["Revenue_LY"].sum() > 0:
        ly = t["Revenue_LY"].sum()
        lines[0] += f", revenue change {pct((tw - ly) / ly * 100, True)} YoY"
    for _, r in t.head(3).iterrows():
        line = f"{r['Entity']}: rank {int(r['Rank'])}, revenue {money(r['Revenue_TW'])} TW, share {pct(r['Share_Pct'])} TW"
        if not np.isnan(r["WoW_Pct"]):
            line += f", revenue change {pct(r['WoW_Pct'], True)} WoW"
        lines.append(line)
    movers = t.dropna(subset=["WoW_Pct"])
    if len(movers):
        for label, r in [("largest WoW increase", movers.loc[movers["WoW_Pct"].idxmax()]),
                         ("largest WoW decrease", movers.loc[movers["WoW_Pct"].idxmin()])]:
            lines.append(f"{r['Entity']} ({label}): revenue change {pct(r['WoW_Pct'], True)} WoW, "
                         f"revenue {money(r['Revenue_TW'])} TW versus {money(r['Revenue_LW'])} LW")
    yoy = t.dropna(subset=["YoY_Pct"])
    if len(yoy):
        for label, r in [("largest YoY increase", yoy.loc[yoy["YoY_Pct"].idxmax()]),
                         ("largest YoY decrease", yoy.loc[yoy["YoY_Pct"].idxmin()])]:
            lines.append(f"{r['Entity']} ({label}): revenue change {pct(r['YoY_Pct'], True)} YoY, "
                         f"revenue {money(r['Revenue_LY'])} LY")
    o = t.loc[t["Orders_TW"].idxmax()]
    lines.append(f"{o['Entity']} (most orders): orders {int(o['Orders_TW']):,} TW versus {int(o['Orders_LW']):,} LW")
    return "\n".join(f"- {l}" for l in lines)


def s4_facts_template(table, meta, model, repeat):
    trace = new_trace("S4", model)
    packet = facts_packet(table)
    trace["packet"] = packet
    text = call(trace, "write", model, system_prompt("s4_system.txt"),
                read_prompt("s4_user.txt").format(week=meta["week"], entity_type=meta["entity_type"], packet=packet),
                repeat)
    return clean_note(text), finish(trace)


STRATEGIES = {"S1": s1_direct, "S2": s2_compute_then_write, "S3": s3_write_then_verify, "S4": s4_facts_template}
