"""A deterministic stand-in for an LLM, so the whole pipeline runs with no API key.

It writes plausible notes from the table in the prompt and plants a few wrong numbers on purpose:
drifted values (unsupported), values taken from another entity (wrong entity), and flipped
directions. Error rates differ by strategy so smoke runs show the expected ordering.
"""
import hashlib
import json
import re

import numpy as np

from table_text import from_markdown, money, pct

ERROR_RATE = {"S1": 0.18, "S2": 0.07, "S4": 0.03}  # chance that any one number in a line is wrong
FIX_RATE = 0.8  # chance the mock fixes a flagged line when asked to revise


def rng_for(*parts):
    seed = int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def respond(system, user, cache_tag=""):
    rng = rng_for(system, user, cache_tag)  # same prompt and repeat gives the same answer
    if "writes short, correct pandas code" in system:
        return mock_code(user, rng)
    if "already computed from the sales table" in system:
        return mock_from_facts(user, rng)
    if "could not match some numbers" in system:
        return mock_revise(user, rng)
    if "facts packet" in system:
        return mock_from_packet(user, rng)
    if "extract numeric claims" in system:
        return json.dumps({"claims": []})
    return mock_note(user, rng)

# ---------------------------------------------------------------- S1: note from the table


def entity_type_of(user):
    m = re.search(r"Entity type: (\w+)", user)
    return m.group(1) if m else "entity"


def slot(value, kind, col=None, entity=None, compact=False):
    return {"value": float(value), "kind": kind, "col": col, "entity": entity, "compact": compact}


def truth_lines(t, entity_type):
    """Error-free note as a list of lines; each line is a list of text pieces and number slots."""
    plural = {"country": "countries", "product": "products"}.get(entity_type, "entities")
    t = t.sort_values("Rank").reset_index(drop=True)
    top, lines = t.iloc[0], []
    e = top["Entity"]
    lines.append([f"- {e} led revenue with ", slot(top["Revenue_TW"], "money", "Revenue_TW", e, True), " TW, a ",
                  slot(top["Share_Pct"], "pct", "Share_Pct", e), f" share of the {len(t)} {plural} shown"])
    if not np.isnan(top["WoW_Pct"]):
        lines.append([f"- {e} revenue was ", slot(top["WoW_Pct"], "pct_dir", "WoW_Pct", e), " WoW from ",
                      slot(top["Revenue_LW"], "money", "Revenue_LW", e), " LW"])
    if len(t) > 1:
        sec = t.iloc[1]
        lines.append([f"- {sec['Entity']} ranked ", slot(2, "rank"), " with ",
                      slot(sec["Revenue_TW"], "money", "Revenue_TW", sec["Entity"]), " TW"])
    movers = t.iloc[1:].dropna(subset=["WoW_Pct"])
    if len(movers):
        g = movers.loc[movers["WoW_Pct"].idxmax()]
        if g["WoW_Pct"] > 0:
            lines.append([f"- {g['Entity']} posted the largest WoW gain, up ", slot(g["WoW_Pct"], "pct", "WoW_Pct", g["Entity"]),
                          " to ", slot(g["Revenue_TW"], "money", "Revenue_TW", g["Entity"]), " TW"])
        d = movers.loc[movers["WoW_Pct"].idxmin()]
        if d["WoW_Pct"] < 0:
            lines.append([f"- {d['Entity']} revenue fell ", slot(d["WoW_Pct"], "pct", "WoW_Pct", d["Entity"]),
                          " WoW to ", slot(d["Revenue_TW"], "money", "Revenue_TW", d["Entity"]), " TW"])
    if not np.isnan(top["YoY_Pct"]):
        lines.append([f"- {e} revenue was ", slot(top["YoY_Pct"], "pct_dir", "YoY_Pct", e), " YoY versus ",
                      slot(top["Revenue_LY"], "money", "Revenue_LY", e), " LY"])
    else:
        lines.append([f"- {e} sold ", slot(top["Units_TW"], "count", "Units_TW", e), " units TW versus ",
                      slot(top["Units_LW"], "count", "Units_LW", e), " units LW"])
    o = t.loc[t["Orders_TW"].idxmax()]
    lines.append([f"- {o['Entity']} had the most orders with ", slot(o["Orders_TW"], "count", "Orders_TW", o["Entity"]),
                  " orders TW versus ", slot(o["Orders_LW"], "count", "Orders_LW", o["Entity"]), " LW"])
    tw, lw = t["Revenue_TW"].sum(), t["Revenue_LW"].sum()
    lines.append([f"- Total revenue across the {plural} shown was ", slot(tw, "money", compact=True), " TW, ",
                  slot((tw - lw) / lw * 100, "pct_dir"), " WoW"])
    return lines[:8]


def perturb(s, t, rng):
    """Return a wrong copy of a slot: drift, another entity's value, or a flipped direction."""
    s = dict(s)
    kind = rng.choice(["drift", "swap", "flip"], p=[0.5, 0.3, 0.2])
    others = t[t["Entity"] != s["entity"]][s["col"]].dropna() if s["col"] else []
    if kind == "flip" and s["kind"] == "pct_dir":
        s["value"] = -s["value"]
    elif kind == "swap" and len(others):
        s["value"] = float(rng.choice(others.values))
    else:
        s["value"] *= float(rng.choice([1.06, 0.92, 1.11, 0.95]))
    return s


def render_slot(s):
    v = s["value"]
    if s["kind"] == "money":
        return money(v, s["compact"])
    if s["kind"] == "pct":
        return pct(v)
    if s["kind"] == "pct_dir":
        return ("up " if v >= 0 else "down ") + pct(v)
    if s["kind"] == "rank":
        n = int(round(v))
        return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n if n < 20 else n % 10, 'th') }"
    return f"{int(round(v)):,}"


def render(lines, t=None, rng=None, error_rate=0.0):
    out = []
    for line in lines:
        pieces = []
        for piece in line:
            if isinstance(piece, dict):
                if rng is not None and rng.random() < error_rate:
                    piece = perturb(piece, t, rng)
                pieces.append(render_slot(piece))
            else:
                pieces.append(piece)
        out.append("".join(pieces))
    return "\n".join(out)


def mock_note(user, rng):
    t = from_markdown(user)
    if t is None:
        return "- No table was provided."
    return render(truth_lines(t, entity_type_of(user)), t, rng, ERROR_RATE["S1"])

# ---------------------------------------------------------------- S3: revise flagged lines


def mock_revise(user, rng):
    t = from_markdown(user)
    draft = user.split("Draft note:\n", 1)[1].split("\n\nNumbers that could not", 1)[0].splitlines()
    flagged = {int(n) - 1 for n in re.findall(r"^- line (\d+):", user, flags=re.MULTILINE)}
    truth = render(truth_lines(t, entity_type_of(user))).splitlines()
    fixed = [truth[i] if i in flagged and i < len(truth) and rng.random() < FIX_RATE else line
             for i, line in enumerate(draft)]
    return "\n".join(fixed)

# ---------------------------------------------------------------- S2: code, then note from facts

GOOD_SCRIPT = '''import json
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1])
facts = []


def add(entity, metric, period, value, formula):
    if pd.notna(value):
        facts.append({"entity": entity, "metric": metric, "period": period,
                      "value": round(float(value), 2), "formula": formula})


tw, lw = df["Revenue_TW"].sum(), df["Revenue_LW"].sum()
add("TOTAL", "revenue", "TW", tw, "sum(Revenue_TW)")
add("TOTAL", "revenue", "WoW", round((tw - lw) / lw * 100, 1) if lw else None, "(sum TW - sum LW) / sum LW * 100")
for _, r in df.sort_values("Revenue_TW", ascending=False).head(5).iterrows():
    add(r["Entity"], "revenue", "TW", r["Revenue_TW"], "Revenue_TW")
    add(r["Entity"], "share", "TW", r["Share_Pct"], "Share_Pct")
    add(r["Entity"], "revenue", "WoW", r["WoW_Pct"], "WoW_Pct")
movers = df.dropna(subset=["WoW_Pct"])
if len(movers):
    up, down = movers.loc[movers["WoW_Pct"].idxmax()], movers.loc[movers["WoW_Pct"].idxmin()]
    add(up["Entity"], "revenue", "WoW", up["WoW_Pct"], "max WoW_Pct")
    add(down["Entity"], "revenue", "WoW", down["WoW_Pct"], "min WoW_Pct")
top = df.sort_values("Revenue_TW", ascending=False).iloc[0]
add(top["Entity"], "revenue", "YoY", top["YoY_Pct"], "YoY_Pct")
add(top["Entity"], "units", "TW", top["Units_TW"], "Units_TW")
add(top["Entity"], "orders", "TW", top["Orders_TW"], "Orders_TW")
print(json.dumps(facts[:25]))
'''
BUGGY_SCRIPT = GOOD_SCRIPT.replace('df["Revenue_TW"].sum()', 'df["Revenue"].sum()')


def mock_code(user, rng):
    buggy = rng.random() < 0.15 and "failed with this error" not in user  # exercise the retry path
    return "```python\n" + (BUGGY_SCRIPT if buggy else GOOD_SCRIPT) + "```"


def fact_line(f):
    e = "Total" if f["entity"] == "TOTAL" else f["entity"]
    v, metric, period = float(f["value"]), f["metric"], f["period"]
    if metric == "share":
        return f"- {e} held a {v:.1f}% share of revenue TW"
    if metric == "rank":
        return f"- {e} ranked #{int(v)} by revenue TW"
    if period in {"WoW", "YoY"}:
        return f"- {e} {metric} was {'up' if v >= 0 else 'down'} {abs(v):.1f}% {period}"
    if metric == "revenue":
        return f"- {e} revenue was {money(v)} {period}"
    return f"- {e} had {int(round(v)):,} {metric} {period}"


def mock_from_facts(user, rng):
    raw = user.split("Facts (JSON):\n", 1)[1].rsplit("\n\nWrite the insight note", 1)[0]
    facts = json.loads(raw)
    lines = []
    for f in facts[:8]:
        if rng.random() < ERROR_RATE["S2"]:
            f = dict(f, value=float(f["value"]) * float(rng.choice([1.06, 0.92, 1.11])))
        lines.append(fact_line(f))
    return "\n".join(lines)

# ---------------------------------------------------------------- S4: narrate a facts packet


def drift_one_number(line, rng):
    nums = list(re.finditer(r"\d[\d,]*\.\d+", line))
    if not nums:
        return line
    m = nums[int(rng.integers(len(nums)))]
    decimals = len(m.group(0).split(".")[1])
    wrong = float(m.group(0).replace(",", "")) * 1.07
    return line[:m.start()] + f"{wrong:,.{decimals}f}" + line[m.end():]


def mock_from_packet(user, rng):
    packet = user.split("Facts packet:\n", 1)[1].split("\n\nWrite the insight note", 1)[0]
    lines = [l.lstrip("- ").strip() for l in packet.splitlines() if l.strip()][:8]
    return "\n".join("- " + (drift_one_number(l, rng) if rng.random() < ERROR_RATE["S4"] else l) for l in lines)
