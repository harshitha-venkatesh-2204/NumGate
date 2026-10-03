"""The number gate: checks every number in a note against its source table.

Level 1 (number support): is each cited number a table cell or a fixed derived quantity?
Level 2 (claim binding): is each (entity, metric, period, value) claim bound to the right cell?
The rules are written out in plain language in docs/method.md.
"""
import itertools
import json
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

# ---------------------------------------------------------------- column metadata

COLUMN_UNIT = {  # unit of every numeric table column
    "Revenue_TW": "currency", "Revenue_LW": "currency", "Revenue_LY": "currency",
    "Units_TW": "count", "Units_LW": "count", "Orders_TW": "count", "Orders_LW": "count",
    "WoW_Pct": "percent", "YoY_Pct": "percent", "Share_Pct": "percent", "Rank": "rank",
}
MEASURES = {"Revenue": "currency", "Units": "count", "Orders": "count"}  # measures with TW/LW(/LY) columns

ENTITY_ALIASES = {  # common short names models use for UCI country labels
    "uk": "United Kingdom", "u.k.": "United Kingdom", "britain": "United Kingdom",
    "great britain": "United Kingdom", "ireland": "EIRE", "the netherlands": "Netherlands",
    "holland": "Netherlands", "south africa": "RSA", "united states": "USA", "us": "USA",
}
TOTAL_WORDS = {"total", "overall", "all countries", "all products", "all entities", "the business",
               "company", "portfolio"}

UP_WORDS = {"up", "rose", "rise", "rises", "rising", "risen", "grew", "grow", "grows", "growing", "growth",
            "increased", "increase", "increases", "increasing", "gained", "gain", "gains", "jumped", "jump",
            "surged", "surge", "climbed", "climb", "soared", "higher", "improved", "uplift", "added"}
DOWN_WORDS = {"down", "fell", "fall", "falls", "falling", "fallen", "declined", "decline", "declines",
              "declining", "decreased", "decrease", "decreasing", "dropped", "drop", "drops", "dropping",
              "lost", "loss", "slipped", "slip", "slid", "plunged", "plunge", "tumbled", "lower",
              "contracted", "contraction", "shrank", "dip", "dipped"}
AFTER_DIRECTION_WORDS = (UP_WORDS | DOWN_WORDS) - {"up", "down", "added", "lost"}  # "12% decline", not "12%, up"
LEVEL_WORDS = {"to", "at", "from", "reaching", "reached", "totaling", "totalling"}  # "fell to £5k" is a level
COUNT_NOUNS = r"units?|orders?|invoices?|customers?|items?|transactions?|pieces?|baskets?"
COUNTING_NOUNS = r"countries|products|markets|regions|entities|lines|skus|stores|weeks|days|months|categories|rows|of them"
PERIOD_PATTERNS = [  # (regex, period label) used by the regex claim extractor
    (r"\bwow\b|week[- ]over[- ]week|week[- ]on[- ]week|vs\.? ?lw\b|versus lw\b|vs\.? last week", "WoW"),
    (r"\byoy\b|year[- ]over[- ]year|year[- ]on[- ]year|vs\.? ?ly\b|versus ly\b|vs\.? last year", "YoY"),
    (r"\blw\b|last week|prior week|previous week", "LW"),
    (r"\bly\b|last year|prior year|same week last year", "LY"),
    (r"\btw\b|this week", "TW"),
]

# ---------------------------------------------------------------- number extraction

NUMBER_RE = re.compile(r"""
    (?<![\w.,])
    (?P<sign>[-+−])?
    (?P<cur>[£$€]|(?:GBP|USD|EUR)\s?)?
    (?P<sign2>[-+−])?
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    (?P<ord>st|nd|rd|th)?
    (?P<mult>\s?(?:bn|mn|[kKmMB]|thousand|million|billion)(?![A-Za-z]))?
    (?P<unit>\s?(?:%|percentage\s?points?|percent|per\s?cent|pct|pp|p\.p\.|x(?![A-Za-z])|times\b|GBP\b|USD\b|EUR\b))?
""", re.VERBOSE)

DATE_RE = re.compile(r"\b\d{4}-W\d{1,2}\b|\b\d{4}-\d{2}-\d{2}\b|\b(?:iso\s+)?(?:week|wk)\s?#?\d{1,2}\b|\bW\d{1,2}\b",
                     re.IGNORECASE)
SCALES = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}


def tokens_before(text, pos, n):
    return re.findall(r"[#\w.'-]+|[£$€]", text[:pos].lower())[-n:]


def tokens_after(text, pos, n):
    return re.findall(r"[#\w.'-]+", text[pos:].lower())[:n]


def find_entity_spans(line, entities):
    """Entity mentions in a line as (start, end, entity). Exact (case-insensitive) first, then aliases, then fuzzy."""
    low = line.lower()
    spans = []

    def free(s, e):
        return all(e <= a or s >= b for a, b, _ in spans)

    for ent in sorted(entities, key=len, reverse=True):
        for m in re.finditer(r"(?<!\w)" + re.escape(ent.lower()) + r"(?!\w)", low):
            if free(m.start(), m.end()):
                spans.append((m.start(), m.end(), ent))
    entity_set = set(entities)
    for alias, ent in ENTITY_ALIASES.items():
        if ent not in entity_set:
            continue
        for m in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", low):
            if free(m.start(), m.end()) and not (alias == "us" and line[m.start():m.end()] != "US"):
                spans.append((m.start(), m.end(), ent))
    for ent in entities:  # fuzzy pass for long names written slightly differently
        if len(ent) < 8 or any(e == ent for _, _, e in spans):
            continue
        al = fuzz.partial_ratio_alignment(ent.lower(), low, score_cutoff=95)
        if al and free(al.dest_start, al.dest_end):
            spans.append((al.dest_start, al.dest_end, ent))
    return sorted(spans)


def split_sentences(line):
    return [s for s in re.split(r"(?<=[.!?;])\s+(?=[A-Z£$€(])", line) if s.strip()]


def parse_match(m, text):
    """Turn one regex match into a number record (value, unit, precision, sign info)."""
    num = m.group("num")
    decimals = len(num.split(".")[1]) if "." in num else 0  # precision at the cited scale
    cited = float(num.replace(",", ""))  # value at the cited scale, e.g. 1.2 for "£1.2M"
    mult = (m.group("mult") or "").strip().lower()
    scale = SCALES.get(mult, 1.0)
    unit_txt = (m.group("unit") or "").strip().lower()
    cur = m.group("cur") or ""
    after = tokens_after(text, m.end(), 1)
    before = tokens_before(text, m.start(), 2)

    is_rank = bool(m.group("ord")) or (before[-1:] and before[-1] in {"#", "no.", "rank", "ranked", "ranks"}) \
        or text[max(0, m.start() - 1):m.start()] == "#"
    if cur or unit_txt in {"gbp", "usd", "eur"}:
        unit = "currency"
    elif unit_txt in {"x", "times"}:
        unit = "plain"
    elif unit_txt:
        unit = "percent"
    elif is_rank or (after and re.fullmatch(COUNT_NOUNS, after[0])):
        unit = "count"
    else:
        unit = "plain"

    sign_txt = m.group("sign") or m.group("sign2") or ""
    explicit_sign = -1 if sign_txt in {"-", "−"} else (1 if sign_txt == "+" else 0)
    direction = "none"
    if not explicit_sign and unit in {"percent", "currency", "count", "plain"} and not is_rank:
        direction = infer_direction(text, m.start(), m.end())
    sign = explicit_sign or {"up": 1, "down": -1}.get(direction, 0)  # 0 means unsigned
    value = cited * scale * (sign or 1)
    return {"raw_text": m.group(0).strip(), "value": value, "cited": cited * (sign or 1), "decimals": decimals,
            "scale": scale, "unit": unit, "signed": sign != 0, "direction": direction, "is_rank": bool(is_rank),
            "start": m.start(), "end": m.end()}


def infer_direction(text, start, end):
    """Direction word before the number (within 3 tokens) or right after it, as in '12% decline'."""
    clause = re.split(r"[,;:()]|\band\b|\bwhile\b|\bbut\b|\bwhereas\b", text[:start])[-1]
    before = re.findall(r"[\w'-]+", clause.lower())
    if before and before[-1] in LEVEL_WORDS:
        return "none"
    for word in reversed(before[-3:]):
        if word in UP_WORDS:
            return "up"
        if word in DOWN_WORDS:
            return "down"
        if re.fullmatch(r"\d[\d,.]*", word):
            break
    after = [w for w in tokens_after(text, end, 3) if w not in {"wow", "yoy", "lw", "ly", "tw"}]
    if after and after[0] in AFTER_DIRECTION_WORDS:
        return "up" if after[0] in UP_WORDS else "down"
    return "none"


def skip_reason(rec, text):
    """Why a number is not a checkable figure (years, 'top 3', '2 of the 5 regions'), else None."""
    if rec["unit"] != "plain" and rec["unit"] != "count" or rec["decimals"] or rec["scale"] != 1:
        return None
    if rec["is_rank"]:
        return None
    n = abs(rec["cited"])
    if "," not in rec["raw_text"] and 1990 <= n <= 2035 and rec["unit"] == "plain":
        return "year"
    if n > 200:
        return None
    before = tokens_before(text, rec["start"], 2)
    after = " ".join(tokens_after(text, rec["end"], 3))
    if before and before[-1] in {"top", "bottom", "first", "last", "next", "past"}:
        return "counting"
    if re.match(r"(of|out of)( the)? \d", after):
        return "counting"
    if len(before) == 2 and before[-1] == "of" and re.fullmatch(r"\d+", before[-2]):
        return "counting"
    if len(before) == 2 and before == ["of", "the"] and re.search(r"\d+\s+of the\s*$", text[:rec["start"]].lower()):
        return "counting"
    if re.match(r"(" + COUNTING_NOUNS + r")\b", after):
        return "counting"
    return None


def extract_numbers(text, entities=()):
    """Every numeric mention in a note, one record per number, with Skipped ones marked."""
    records = []
    for line_idx, line in enumerate(l for l in text.splitlines() if l.strip()):
        ent_spans = find_entity_spans(line, entities)
        masked = [(a, b, "entity_name") for a, b, _ in ent_spans]
        masked += [(m.start(), m.end(), "date_or_week") for m in DATE_RE.finditer(line)]
        for sentence in split_sentences(line):
            offset = line.find(sentence)
            for m in NUMBER_RE.finditer(sentence):
                rec = parse_match(m, sentence)
                a, b = rec["start"] + offset, rec["end"] + offset
                reason = next((r for s, e, r in masked if a < e and b > s), None) or skip_reason(rec, sentence)
                rec.update({"line_idx": line_idx, "line": line, "sentence": sentence, "position": a,
                            "skip_reason": reason or "",
                            "line_entities": sorted({e for _, _, e in ent_spans})})
                records.append(rec)
    return records


def parse_value_text(value_text):
    """Parse a single cited value such as '£1.2M' or '-4.5%' (used for extractor output)."""
    recs = [r for r in extract_numbers(str(value_text)) if r["skip_reason"] != "date_or_week"]
    return recs[0] if recs else None

# ---------------------------------------------------------------- matching


def matches(true_values, rec, force_signed=None):
    """Boolean array: which true values round to the cited value at the cited precision.

    A value matches if |true/scale - cited| <= half a unit in the last cited digit.
    Values cited in k, M or B also match within 0.5 percent relative error.
    Unsigned citations (no sign and no direction word) are compared on absolute values.
    """
    t = np.asarray(true_values, dtype=float)
    signed = rec["signed"] if force_signed is None else force_signed
    cited = rec["cited"] if signed else abs(rec["cited"])
    if not signed:
        t = np.abs(t)
    tol = 0.5 * 10 ** (-rec["decimals"]) + 1e-9
    ok = np.abs(t / rec["scale"] - cited) <= tol
    if rec["scale"] > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            ok |= np.abs(t - cited * rec["scale"]) <= 0.005 * np.abs(t)
    return ok & np.isfinite(t)


def unit_ok(cand_unit, rec):
    if rec["is_rank"]:
        return cand_unit == "rank"
    if rec["unit"] == "plain":
        return True
    return cand_unit == rec["unit"]


def pct(a, b):
    return (a - b) / b * 100 if b else np.nan

# ---------------------------------------------------------------- candidate library


def cell_candidates(table):
    out = []
    for _, row in table.iterrows():
        for col, unit in COLUMN_UNIT.items():
            if col in row and pd.notna(row[col]):
                out.append((float(row[col]), unit, f"cell({row['Entity']}, {col})", row["Entity"]))
    return out


def table_candidates(table):
    """Entity-free derived values: column aggregates, total period changes, top-k cumulative revenue."""
    out = []
    for col, unit in COLUMN_UNIT.items():
        if col == "Rank" or col not in table or table[col].notna().sum() == 0:
            continue
        s = table[col].dropna().astype(float)
        agg_unit = "percent" if unit == "percent" else unit
        out += [(s.sum(), agg_unit, f"sum({col})"), (s.mean(), agg_unit, f"mean({col})"),
                (s.median(), agg_unit, f"median({col})"), (s.max(), agg_unit, f"max({col})"),
                (s.min(), agg_unit, f"min({col})")]
    out.append((float(len(table)), "count", "count(rows)"))
    for m, unit in MEASURES.items():
        tw = table[f"{m}_TW"].sum()
        for per in ["LW", "LY"]:
            col = f"{m}_{per}"
            if col not in table or table[col].notna().sum() == 0:
                continue
            base = table[col].sum()
            out += [(tw - base, unit, f"diff(sum({m}_TW), sum({col}))"),
                    (pct(tw, base), "percent", f"pct_change(sum({m}_TW), sum({col}))"),
                    (tw / base * 100 if base else np.nan, "percent", f"pct_of(sum({m}_TW), sum({col}))"),
                    (tw / base if base else np.nan, "ratio", f"ratio(sum({m}_TW), sum({col}))")]
    ranked = table.sort_values("Revenue_TW", ascending=False)["Revenue_TW"].astype(float)
    total = ranked.sum()
    for k in range(2, min(10, len(ranked)) + 1):
        top = ranked.iloc[:k].sum()
        out += [(top, "currency", f"sum_top{k}(Revenue_TW)"), (top / total * 100, "percent", f"share_top{k}(Revenue_TW)")]
    return out


def entity_candidates(table, entities):
    """Derived values anchored on the entities named in the line."""
    out = []
    rows = table.set_index("Entity")
    named = [e for e in entities if e in rows.index]
    for e in named:
        r = rows.loc[e]
        for m, unit in MEASURES.items():
            tw = float(r[f"{m}_TW"])
            for per in ["LW", "LY"]:
                col = f"{m}_{per}"
                if col not in r or pd.isna(r[col]):
                    continue
                base = float(r[col])
                out += [(tw - base, unit, f"diff({e}.{m}_TW, {e}.{col})"),
                        (pct(tw, base), "percent", f"pct_change({e}.{m}_TW, {e}.{col})"),
                        (tw / base * 100 if base else np.nan, "percent", f"pct_of({e}.{m}_TW, {e}.{col})"),
                        (tw / base if base else np.nan, "ratio", f"ratio({e}.{m}_TW, {e}.{col})")]
        for col in ["Revenue_TW", "Revenue_LW", "Units_TW", "Units_LW", "Orders_TW", "Orders_LW"]:
            total = table[col].sum()
            v = float(r[col])
            mean = table[col].mean()
            out += [(v / total * 100 if total else np.nan, "percent", f"share({e}.{col})"),
                    (v - mean, COLUMN_UNIT[col], f"diff({e}.{col}, mean({col}))"),
                    (pct(v, mean), "percent", f"pct_change({e}.{col}, mean({col}))"),
                    (v / mean if mean else np.nan, "ratio", f"ratio({e}.{col}, mean({col}))")]
    for a, b in itertools.permutations(named, 2):
        for col, unit in COLUMN_UNIT.items():
            if col == "Rank" or pd.isna(rows.loc[a, col]) or pd.isna(rows.loc[b, col]):
                continue
            va, vb = float(rows.loc[a, col]), float(rows.loc[b, col])
            out += [(va - vb, unit, f"diff({a}.{col}, {b}.{col})"),
                    (va + vb, unit, f"sum({a}.{col}, {b}.{col})")]
            if unit != "percent":
                out += [(pct(va, vb), "percent", f"pct_change({a}.{col}, {b}.{col})"),
                        (va / vb if vb else np.nan, "ratio", f"ratio({a}.{col}, {b}.{col})")]
    if len(named) > 2:
        for col in ["Revenue_TW", "Share_Pct", "Units_TW", "Orders_TW"]:
            out.append((float(rows.loc[named, col].sum()), COLUMN_UNIT[col], f"sum({'+'.join(named)}.{col})"))
    return out


def first_match(cands, rec):
    if not cands:
        return None
    usable = [c for c in cands if unit_ok(c[1], rec)]
    if not usable:
        return None
    hit = np.flatnonzero(matches([c[0] for c in usable], rec))
    return usable[hit[0]] if len(hit) else None


def support_number(rec, table, cells, table_derived):
    """Label one number: Supported_cell, Supported_derived or Unsupported, with evidence."""
    named = set(rec["line_entities"])
    own_cells = [c for c in cells if c[3] in named]
    hit = first_match(own_cells, rec) or first_match(cells, rec)
    if hit:
        return "Supported_cell", f"{hit[2]}={hit[0]:g}"
    hit = first_match(table_derived, rec) or first_match(entity_candidates(table, rec["line_entities"]), rec)
    if hit:
        return "Supported_derived", f"{hit[2]}={hit[0]:.6g}"
    return "Unsupported", ""


def gate_level1(note_text, table, note_id=""):
    """One record per numeric mention in the note."""
    entities = table["Entity"].astype(str).tolist()
    cells = cell_candidates(table)
    table_derived = table_candidates(table)
    out = []
    for rec in extract_numbers(note_text, entities):
        if rec["skip_reason"]:
            label, evidence = "Skipped", rec["skip_reason"]
        else:
            label, evidence = support_number(rec, table, cells, table_derived)
        out.append({"note_id": note_id, "line_idx": rec["line_idx"], "sentence": rec["sentence"],
                    "raw_text": rec["raw_text"], "value": rec["value"], "unit": "rank" if rec["is_rank"] else rec["unit"],
                    "direction": rec["direction"], "position": rec["position"], "label": label, "evidence": evidence})
    return out

# ---------------------------------------------------------------- level 2: claim binding


def claim_frame(table):
    """Per entity, every value a claim can bind to, keyed by virtual column name. Includes a TOTAL row."""
    frame = {}
    has_ly = table["Revenue_LY"].notna().any()
    for _, r in table.iterrows():
        f = {c: float(r[c]) if pd.notna(r[c]) else np.nan for c in COLUMN_UNIT}
        for m in MEASURES:
            f[f"{m}_WoW_Abs"] = f[f"{m}_TW"] - f[f"{m}_LW"]
            f[f"{m}_WoW_Pct"] = pct(f[f"{m}_TW"], f[f"{m}_LW"])
        f["Revenue_WoW_Pct"] = f["WoW_Pct"]
        f["Revenue_YoY_Pct"] = f["YoY_Pct"]
        f["Revenue_YoY_Abs"] = f["Revenue_TW"] - f["Revenue_LY"]
        f["Share_LW"] = f["Revenue_LW"] / table["Revenue_LW"].sum() * 100 if table["Revenue_LW"].sum() else np.nan
        frame[str(r["Entity"])] = f
    tot = {c: float(table[c].sum()) for c in COLUMN_UNIT if c not in {"WoW_Pct", "YoY_Pct", "Share_Pct", "Rank"}}
    if not has_ly:
        tot["Revenue_LY"] = np.nan
    for m in MEASURES:
        tot[f"{m}_WoW_Abs"] = tot[f"{m}_TW"] - tot[f"{m}_LW"]
        tot[f"{m}_WoW_Pct"] = pct(tot[f"{m}_TW"], tot[f"{m}_LW"])
    tot["WoW_Pct"] = tot["Revenue_WoW_Pct"]
    tot["Revenue_YoY_Abs"] = tot["Revenue_TW"] - tot["Revenue_LY"]
    tot["Revenue_YoY_Pct"] = tot["YoY_Pct"] = pct(tot["Revenue_TW"], tot["Revenue_LY"]) if has_ly else np.nan
    tot["Share_Pct"] = 100.0
    frame["TOTAL"] = tot
    return frame


def resolve_column(metric, period, unit):
    """Map (metric, period, unit) to a virtual column name, or None if the table cannot answer it."""
    metric = (metric or "revenue").lower()
    period = (period or "TW").upper().replace("-", "")
    period = {"WOW": "WoW", "YOY": "YoY"}.get(period, period)
    if metric == "rank":
        return "Rank"
    if metric == "share":
        return {"TW": "Share_Pct", "LW": "Share_LW"}.get(period)
    measure = {"revenue": "Revenue", "sales": "Revenue", "units": "Units", "orders": "Orders"}.get(metric)
    if not measure:
        return None
    if period in {"TW", "LW", "LY"}:
        col = f"{measure}_{period}"
        return col if col in COLUMN_UNIT else None
    if period in {"WoW", "YoY"}:
        if measure != "Revenue" and period == "YoY":
            return None  # the table has no LY units or orders
        return f"{measure}_{period}_{'Pct' if unit == 'percent' else 'Abs'}"
    return None


def match_entity(name, entities):
    if not name:
        return None
    low = name.strip().lower()
    if low in TOTAL_WORDS or low.startswith("total"):
        return "TOTAL"
    for e in entities:
        if e.lower() == low:
            return e
    alias = ENTITY_ALIASES.get(low)
    if alias in entities:
        return alias
    best = process.extractOne(name, entities, scorer=fuzz.WRatio, processor=str.lower, score_cutoff=88)
    return best[0] if best else None


def verify_claim(claim, table, frame=None):
    """Label one claim: Correct, Wrong_entity, Wrong_metric, Wrong_value or Unverifiable."""
    frame = frame or claim_frame(table)
    entities = table["Entity"].astype(str).tolist()
    rec = claim.get("parsed") or parse_value_text(claim.get("value_text", ""))
    if rec is None:
        return "Unverifiable", "value not parseable"
    if claim.get("direction") in {"up", "down"} and not rec["signed"]:  # extractor-supplied direction
        sign = 1 if claim["direction"] == "up" else -1
        rec = dict(rec, cited=abs(rec["cited"]) * sign, signed=True)
    row = match_entity(claim.get("entity"), entities)
    if row is None:
        return "Unverifiable", f"entity not found: {claim.get('entity')}"
    col = resolve_column(claim.get("metric"), claim.get("period"), rec["unit"])
    targets = [col] if col else []  # columns the claim may bind to
    if col and rec["unit"] == "plain" and col.endswith("_Abs"):
        targets.append(col.replace("_Abs", "_Pct"))  # a unitless change could be either
    targets = [c for c in targets if pd.notna(frame[row].get(c, np.nan))]
    if not targets:
        return "Unverifiable", f"no column for metric={claim.get('metric')} period={claim.get('period')}"
    for c in targets:
        if matches([frame[row][c]], rec)[0]:
            return "Correct", f"{row}.{c}={frame[row][c]:.6g}"
    col = targets[0]
    if rec["signed"] and any(matches([frame[row][c]], rec, force_signed=False)[0] for c in targets):
        return "Wrong_value", f"{row}.{col}={frame[row][col]:.6g} (direction wrong)"
    for other, f in frame.items():
        for c in targets:
            if other != row and matches([f.get(c, np.nan)], rec)[0]:
                return "Wrong_entity", f"value belongs to {other}.{c}={f[c]:.6g}; claimed {row}"
    for c, v in frame[row].items():
        if c not in targets and matches([v], rec)[0]:
            return "Wrong_metric", f"value is {row}.{c}={v:.6g}; claimed {col}"
    return "Wrong_value", f"{row}.{col}={frame[row][col]:.6g}"


def regex_claims(note_text, table):
    """Rule-based claim extractor used for the mock model (and as a fallback)."""
    entities = table["Entity"].astype(str).tolist()
    claims = []
    for line_idx, line in enumerate(l for l in note_text.splitlines() if l.strip()):
        spans = find_entity_spans(line, entities)
        low = line.lower()
        total_spans = [(m.start(), m.end(), "TOTAL") for m in re.finditer(r"\b(total|overall)\b", low)]
        mentions = sorted(spans + total_spans)
        nums = [r for r in extract_numbers(line, entities) if not r["skip_reason"]]
        prev_end = 0
        for r in nums:
            before_mentions = [m for m in mentions if m[1] <= r["position"]]
            entity = before_mentions[-1][2] if before_mentions else (mentions[0][2] if mentions else "")
            window_start = max(prev_end, before_mentions[-1][1] if before_mentions else 0)
            window = low[window_start:r["position"]]
            after = low[r["position"] + len(r["raw_text"]):][:30]
            metric = guess_metric(window, after, r, claims[-1]["metric"] if claims and claims[-1]["line_idx"] == line_idx else None)
            period = guess_period(window, after, r, metric)
            claims.append({"line_idx": line_idx, "entity": entity, "metric": metric, "period": period,
                           "value_text": r["raw_text"], "direction": r["direction"], "parsed": r})
            prev_end = r["position"] + len(r["raw_text"])
    return claims


METRIC_WORDS = [("rank", r"\brank"), ("share", r"\bshare"), ("units", r"\bunits?\b"),
                ("orders", r"\borders?\b|\binvoices?\b"), ("revenue", r"\brevenue\b|\bsales\b")]


def metric_word(text, nearest_to_end):
    """The metric keyword closest to the number: last one before it, or first one after it."""
    hits = [(m.start(), name) for name, pat in METRIC_WORDS for m in re.finditer(pat, text)]
    if not hits:
        return None
    return max(hits)[1] if nearest_to_end else min(hits)[1]


def guess_metric(before, after, rec, previous):
    """Metric for a number: keyword just before it, else right after it, else the previous claim's metric."""
    if rec["is_rank"]:
        return "rank"
    if rec["unit"] == "currency":
        return "revenue"
    after_near = re.split(r"[,;(]|\d", after, maxsplit=1)[0]  # stop at the next clause or number
    metric = metric_word(before, True) or metric_word(after_near, False) or previous or "revenue"
    return "revenue" if metric == "rank" else metric


def guess_period(before, after, rec, metric):
    if metric in {"rank", "share"}:
        return "TW"
    near_after = " ".join(after.split()[:2])
    for pattern, label in PERIOD_PATTERNS:
        if re.search(pattern, near_after):
            return label
    for pattern, label in PERIOD_PATTERNS:
        if re.search(pattern, before):
            return label
    if rec["unit"] == "percent" or rec["direction"] != "none":
        return "WoW"
    return "TW"


def llm_claims(note_text, table, extractor_model, complete_fn):
    """LLM claim extractor with a fixed JSON schema. Returns (claims, raw_response)."""
    from config import read_prompt
    lines = [l for l in note_text.splitlines() if l.strip()]
    numbered = "\n".join(f"{i}: {l}" for i, l in enumerate(lines))
    user = read_prompt("extract_user.txt").format(note_lines=numbered)
    resp = complete_fn(extractor_model, read_prompt("extract_system.txt"), user, temperature=0, max_tokens=4000)
    text = resp["text"]
    try:
        payload = json.loads(text[text.find("{"): text.rfind("}") + 1])
        claims = payload["claims"]
    except (ValueError, KeyError, TypeError):
        return None, resp
    for c in claims:
        c["line_idx"] = int(c.get("line", c.get("line_idx", 0)) or 0)
    return claims, resp


def gate_level2(note_text, table, note_id="", claims=None):
    """One record per claim. Claims come from the LLM extractor or, if None, the regex extractor."""
    frame = claim_frame(table)
    claims = regex_claims(note_text, table) if claims is None else claims
    out = []
    for c in claims:
        label, evidence = verify_claim(c, table, frame)
        out.append({"note_id": note_id, "line_idx": c.get("line_idx", 0), "entity": c.get("entity", ""),
                    "metric": c.get("metric", ""), "period": c.get("period", ""),
                    "value_text": c.get("value_text", ""), "direction": c.get("direction", "none"),
                    "label": label, "evidence": evidence})
    return out

# ---------------------------------------------------------------- note-level summary

NUMBER_LABELS = ["Supported_cell", "Supported_derived", "Unsupported", "Skipped"]
CLAIM_LABELS = ["Correct", "Wrong_entity", "Wrong_metric", "Wrong_value", "Unverifiable"]
CLAIM_ERRORS = ["Wrong_entity", "Wrong_metric", "Wrong_value"]


def summarize(numbers, claims):
    counts = {f"n_{l}": sum(r["label"] == l for r in numbers) for l in NUMBER_LABELS}
    checked = counts["n_Supported_cell"] + counts["n_Supported_derived"] + counts["n_Unsupported"]
    counts["n_numbers_checked"] = checked
    counts["unsupported_rate"] = counts["n_Unsupported"] / checked if checked else np.nan
    counts.update({f"n_{l}": sum(r["label"] == l for r in claims) for l in CLAIM_LABELS})
    verifiable = sum(counts[f"n_{l}"] for l in ["Correct"] + CLAIM_ERRORS)
    counts["n_claims_verifiable"] = verifiable
    counts["claim_error_rate"] = sum(counts[f"n_{l}"] for l in CLAIM_ERRORS) / verifiable if verifiable else np.nan
    return counts
