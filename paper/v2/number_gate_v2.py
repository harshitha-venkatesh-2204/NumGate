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

ENTITY_ALIASES = {  # short names and demonyms models use for UCI country labels
    "uk": "United Kingdom", "u.k.": "United Kingdom", "britain": "United Kingdom", "great britain": "United Kingdom",
    "british": "United Kingdom", "ireland": "EIRE", "irish": "EIRE", "the netherlands": "Netherlands",
    "holland": "Netherlands", "dutch": "Netherlands", "south africa": "RSA", "united states": "USA", "us": "USA",
    "american": "USA", "german": "Germany", "french": "France", "spanish": "Spain", "belgian": "Belgium",
    "swiss": "Switzerland", "portuguese": "Portugal", "italian": "Italy", "australian": "Australia",
    "japanese": "Japan", "swedish": "Sweden", "norwegian": "Norway", "danish": "Denmark", "finnish": "Finland",
    "austrian": "Austria", "polish": "Poland", "greek": "Greece", "cypriot": "Cyprus", "icelandic": "Iceland",
    "israeli": "Israel", "lebanese": "Lebanon", "maltese": "Malta", "brazilian": "Brazil", "canadian": "Canada",
    "czech": "Czech Republic", "lithuanian": "Lithuania", "bahraini": "Bahrain", "saudi": "Saudi Arabia",
}
TOTAL_WORDS = {"total", "overall", "all countries", "all products", "all entities", "the business",
               "company", "portfolio", "all rows", "all markets"}
GROUP_WORDS = {"group", "others", "rest", "remaining", "top five", "top 5", "top three", "top 3", "top ten", "top 10"}
AGG_WORDS = (r"\b(total|totals|totalled|totaled|totalling|totaling|overall|top|all|across|combined|together|average|mean|"
             r"median|rest|other|others|remaining|portfolio|business|aggregate|collectively|jointly|between them|lines|"
             r"colourways|family|sum|summed|listed|shown|whole|entire|table|every)\b")

UP_WORDS = {"up", "rose", "rise", "rises", "rising", "risen", "grew", "grow", "grows", "growing", "growth",
            "increased", "increase", "increases", "increasing", "gained", "gain", "gains", "jumped", "jump",
            "surged", "surge", "climbed", "climb", "soared", "higher", "improved", "uplift", "added"}
DOWN_WORDS = {"down", "fell", "fall", "falls", "falling", "fallen", "declined", "decline", "declines",
              "declining", "decreased", "decrease", "decreasing", "dropped", "drop", "drops", "dropping",
              "lost", "loss", "slipped", "slip", "slid", "plunged", "plunge", "tumbled", "lower",
              "contracted", "contraction", "shrank", "dip", "dipped"}
AFTER_DIRECTION_WORDS = (UP_WORDS | DOWN_WORDS) - {"up", "down", "added", "lost"}
LEVEL_WORDS = {"to", "at", "from", "reaching", "reached", "totaling", "totalling", "was", "were", "is"}
PERIOD_TOKENS = {"wow", "yoy", "lw", "ly", "tw", "week-on-week", "year-on-year", "week-over-week", "year-over-year"}
CHANGE_NOUNS = {"decline", "drop", "gain", "swing", "increase", "decrease", "change", "fall", "rise", "loss", "uplift",
                "movement", "difference", "shift", "shortfall", "improvement", "growth", "contraction"}
COMPARE_WORDS = {"on", "vs", "vs.", "versus", "from", "against", "compared", "over"}
HEDGES = {  # words before a number that make it approximate or a bound
    "approx": ["about", "around", "roughly", "approximately", "nearly", "almost", "circa", "c.", "~", "some",
               "close to", "just under", "just over", "just below", "just above", "approx.", "approx"],
    "over": ["more than", "over", "above", "exceeding", "at least", "upwards of", "north of", "in excess of"],
    "under": ["less than", "under", "below", "at most", "fewer than", "short of", "no more than"],
}
COUNT_NOUNS = r"units?|orders?|invoices?|customers?|items?|transactions?|pieces?|baskets?"
COUNTING_NOUNS = r"countries|products|markets|regions|entities|lines|skus|stores|weeks|days|months|categories|rows|of them|colourways|items listed"
FAMILY_STOP = {"of", "the", "and", "set", "a", "in", "with", "for", "to", "on", "at", "by", "or"}
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
    (?P<sign>[-+−–])?
    (?P<cur>[£$€]|(?:GBP|USD|EUR)\s?)?
    (?P<sign2>[-+−–])?
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    (?P<ord>st|nd|rd|th)?
    (?P<mult>\s?(?:bn|mn|[kKmMB]|thousand|million|billion)(?![A-Za-z]))?
    (?P<unit>\s?(?:%|percentage\s?points?|percent|per\s?cent|pct|pc\b|pp\b|pts\b|points\b|p\.p\.|x(?![A-Za-z])|times\b|-?fold\b|GBP\b|USD\b|EUR\b))?
""", re.VERBOSE)

DATE_RE = re.compile(r"\b\d{4}-W\d{1,2}\b|\b\d{4}-\d{2}-\d{2}\b|\b(?:iso\s+)?(?:week|wk)\s?#?\d{1,2}\b|\bW\d{1,2}\b",
                     re.IGNORECASE)
SCALES = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}
MINUS_SIGNS = {"-", "−", "–"}  # hyphen, minus sign, en dash


def tokens_before(text, pos, n):
    return re.findall(r"[#\w.'-]+|[£$€~]", text[:pos].lower())[-n:]


def tokens_after(text, pos, n):
    return [t.rstrip(".") for t in re.findall(r"[#\w.'-]+", text[pos:].lower())[:n]]


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
    # never split after abbreviations such as "vs." or "c."
    return [s for s in re.split(r"(?<!\bvs\.)(?<!\bc\.)(?<=[.!?;])\s+(?=[A-Z£$€(])", line) if s.strip()]


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

    is_rank = bool(m.group("ord")) or (before[-1:] and before[-1] in {"#", "no.", "no#", "rank", "ranked", "ranks", "ranking"}) \
        or text[max(0, m.start() - 1):m.start()] == "#"
    is_ratio = unit_txt in {"x", "times"} or unit_txt.endswith("fold")
    if cur or unit_txt in {"gbp", "usd", "eur"}:
        unit = "currency"
    elif is_ratio:
        unit = "plain"
    elif unit_txt:
        unit = "percent"
    elif is_rank or (after and re.fullmatch(COUNT_NOUNS, after[0])):
        unit = "count"
    else:
        unit = "plain"

    sign_txt = m.group("sign") or m.group("sign2") or ""
    explicit_sign = -1 if sign_txt in MINUS_SIGNS else (1 if sign_txt == "+" else 0)
    direction = "none"
    if not explicit_sign and not is_rank:
        direction = infer_direction(text, m.start(), m.end())
    sign = explicit_sign or {"up": 1, "down": -1}.get(direction, 0)  # 0 means unsigned
    value = cited * scale * (sign or 1)
    return {"raw_text": m.group(0).strip(), "value": value, "cited": cited * (sign or 1), "decimals": decimals,
            "scale": scale, "unit": unit, "signed": sign != 0, "direction": direction, "is_rank": bool(is_rank),
            "is_ratio": is_ratio, "qualifier": hedge_of(text, m.start()), "start": m.start(), "end": m.end()}


def hedge_of(text, start):
    """approx / over / under when a hedge phrase sits just before the number, else none."""
    before = " " + " ".join(re.findall(r"[\w.'~-]+|~", text[:start].lower())[-3:]) + " "
    for kind in ["approx", "over", "under"]:  # "just over" is approx, so approx is checked first
        for phrase in HEDGES[kind]:
            if before.rstrip().endswith(" " + phrase) or before.rstrip().endswith(phrase) and phrase == "~":
                return kind
    return "none"


def infer_direction(text, start, end):
    """Direction word before the number (within 3 tokens) or right after it, as in '12% decline'."""
    clause = re.split(r"[,;:()]|\band\b|\bwhile\b|\bbut\b|\bwhereas\b", text[:start])[-1]
    before = re.findall(r"[\w'-]+", clause.lower())
    for word in reversed(before[-3:]):
        if word in LEVEL_WORDS:  # "fell to just £515" states a level, even with a hedge in between
            break
        if word in UP_WORDS:
            return "up"
        if word in DOWN_WORDS:
            return "down"
        if re.fullmatch(r"\d[\d,.]*", word):
            break
    after = [w for w in tokens_after(text, end, 4) if w not in PERIOD_TOKENS]
    if not after:
        return "none"
    if after[0] in {"above", "ahead"}:  # "0.3% above LY"
        return "up"
    if after[0] in {"below", "behind"}:
        return "down"
    if after[0] in AFTER_DIRECTION_WORDS:
        return "up" if after[0] in UP_WORDS else "down"
    if len(after) > 1 and after[0] in {"up", "down"} and after[1] in COMPARE_WORDS:
        return "up" if after[0] == "up" else "down"  # "2.8% up on last week"
    return "none"


def skip_reason(rec, text):
    """Why a number is not a checkable figure (years, 'top 3', '2 of the 5 regions'), else None."""
    if rec["unit"] != "plain" and rec["unit"] != "count" or rec["decimals"] or rec["scale"] != 1:
        return None
    if rec["is_rank"] or rec["is_ratio"]:
        return None
    n = abs(rec["cited"])
    if "," not in rec["raw_text"] and 1990 <= n <= 2035 and rec["unit"] == "plain":
        return "year"
    if n > 200:
        return None
    before = tokens_before(text, rec["start"], 2)
    after = " ".join(tokens_after(text, rec["end"], 4))
    if before and before[-1] in {"top", "bottom", "first", "last", "next", "past"}:
        return "counting"
    if re.match(r"(of|out of)( the)?( top| bottom)? \d", after):
        return "counting"
    if len(before) == 2 and before[-1] == "of" and re.fullmatch(r"\d+", before[-2]):
        return "counting"
    if len(before) == 2 and before == ["of", "the"] and re.search(r"\d+\s+of the\s*$", text[:rec["start"]].lower()):
        return "counting"
    if re.match(r"((listed|shown|ranked|tracked|reported|other|remaining|named|trading|active) )?(" + COUNTING_NOUNS + r")\b", after):
        return "counting"
    return None


def extract_numbers(text, entities=()):
    """Every numeric mention in a note, one record per number, with Skipped ones marked."""
    records = []
    for line_idx, line in enumerate(l for l in text.splitlines() if l.strip()):
        # "No.3" means rank 3; rewrite to "No#3" (same length, so positions still point into the line)
        norm = re.sub(r"\b(No)\.(?=\d)", r"\1#", line, flags=re.IGNORECASE)
        ent_spans = find_entity_spans(norm, entities)
        masked = [(a, b, "entity_name") for a, b, _ in ent_spans]
        masked += [(m.start(), m.end(), "date_or_week") for m in DATE_RE.finditer(norm)]
        for sentence in split_sentences(norm):
            offset = norm.find(sentence)
            prev = None  # the previous number in this sentence
            for m in NUMBER_RE.finditer(sentence):
                rec = parse_match(m, sentence)
                if prev and prev["is_rank"] and re.fullmatch(r"\s*(to|through|and|-|\u2013)\s*", sentence[prev["end"]:m.start()]):
                    rec.update(is_rank=True, unit="count", direction="none", signed=False, cited=abs(rec["cited"]))  # "ranks 1 to 6"
                after = tokens_after(sentence, m.end(), 1)
                rec["period_hint"] = after[0].upper() if after and after[0] in {"tw", "lw", "ly"} else ""
                prev = rec
                a, b = rec["start"] + offset, rec["end"] + offset
                reason = next((r for s, e, r in masked if a < e and b > s), None) or skip_reason(rec, sentence)
                rec.update({"line_idx": line_idx, "line": line, "sentence": line[offset:offset + len(sentence)],
                            "position": a, "skip_reason": reason or "",
                            "line_entities": sorted({e for _, _, e in ent_spans})})
                records.append(rec)
    return records


def parse_value_text(value_text):
    """Parse a single cited value such as '£1.2M' or '-4.5%' (used for extractor output)."""
    recs = [r for r in extract_numbers(str(value_text)) if r["skip_reason"] != "date_or_week"]
    return recs[0] if recs else None

# ---------------------------------------------------------------- matching


def matches(true_values, rec, force_signed=None, allow_bounds=True):
    """Boolean array: which true values round to the cited value at the cited precision.

    A value matches if it rounds half-up to the cited value at the cited precision:
    -half <= |true|/scale - |cited| < half, where half is half a unit in the last cited digit (6.5 rounds to 7, not 6).
    Values cited in k, M or B also match within 0.5 percent relative error.
    Unsigned citations (no sign and no direction word) are compared on absolute values; signed ones must also agree in sign.
    Hedged citations: "about X" matches within 5 percent; "more than X" / "less than X" match as bounds
    when allow_bounds is true (the caller only allows bounds against the line's own entities).
    """
    t = np.asarray(true_values, dtype=float)
    signed = rec["signed"] if force_signed is None else force_signed
    cited = rec["cited"]
    same_sign = ((np.sign(t) == np.sign(cited)) | (t == 0) | (cited == 0)) if signed else np.ones(t.shape, dtype=bool)
    ta, ca = np.abs(t), abs(cited)  # magnitudes; the sign is checked separately
    half = 0.5 * 10 ** (-rec["decimals"])
    diff = ta / rec["scale"] - ca
    ok = (diff >= -half - 1e-9) & (diff < half - 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        target = ca * rec["scale"]
        if rec["scale"] > 1:
            ok |= np.abs(ta - target) <= 0.005 * ta
        q = rec.get("qualifier", "none")
        if q == "approx":
            ok |= np.abs(ta - target) <= 0.05 * ta
        elif q in {"over", "under"} and allow_bounds:
            ok |= (ta >= target - half * rec["scale"]) if q == "over" else (ta <= target + half * rec["scale"])
    return ok & same_sign & np.isfinite(t)


def unit_ok(cand_unit, rec):
    """Ranks match only Rank, ratios only ratios, and a plain number anything except those two."""
    if rec["is_rank"]:
        return cand_unit == "rank"
    if rec.get("is_ratio"):
        return cand_unit == "ratio"
    if cand_unit == "rank":
        return False
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
        if unit == "percent":  # a sum or median of percentages is not a meaningful figure
            out += [(s.mean(), unit, f"mean({col})"), (s.max(), unit, f"max({col})"), (s.min(), unit, f"min({col})")]
        else:
            out += [(s.sum(), unit, f"sum({col})"), (s.mean(), unit, f"mean({col})"),
                    (s.median(), unit, f"median({col})"), (s.max(), unit, f"max({col})"), (s.min(), unit, f"min({col})")]
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
    share_lw_total = table["Revenue_LW"].sum()
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
        share_lw = float(r["Revenue_LW"]) / share_lw_total * 100 if share_lw_total else np.nan
        share_tw = float(r["Revenue_TW"]) / table["Revenue_TW"].sum() * 100  # unrounded, unlike the Share_Pct column
        out.append((share_tw - share_lw, "percent", f"share_change_pp({e}.Share_Pct, {e}.Share_LW)"))
        for m, unit in MEASURES.items():  # everything except this entity ("non-UK markets")
            rest_tw = float(table[f"{m}_TW"].sum() - r[f"{m}_TW"])
            rest_lw = float(table[f"{m}_LW"].sum() - r[f"{m}_LW"])
            out += [(rest_tw, unit, f"sum_excluding({e}.{m}_TW)"), (rest_lw, unit, f"sum_excluding({e}.{m}_LW)"),
                    (pct(rest_tw, rest_lw), "percent", f"pct_change(sum_excluding({e}.{m}_TW), sum_excluding({e}.{m}_LW))")]
        for m in ["Revenue", "Units"]:  # revenue per order and per unit
            for per in ["TW", "LW"]:
                orders = float(r[f"Orders_{per}"])
                if m == "Revenue":
                    out.append((float(r[f"Revenue_{per}"]) / orders if orders else np.nan, "currency", f"per_order({e}.Revenue_{per})"))
                    units = float(r[f"Units_{per}"])
                    out.append((float(r[f"Revenue_{per}"]) / units if units else np.nan, "currency", f"per_unit({e}.Revenue_{per})"))
    for a, b in itertools.permutations(named, 2):
        for col, unit in COLUMN_UNIT.items():
            if col == "Rank" or pd.isna(rows.loc[a, col]) or pd.isna(rows.loc[b, col]):
                continue
            va, vb = float(rows.loc[a, col]), float(rows.loc[b, col])
            out += [(va - vb, unit, f"diff({a}.{col}, {b}.{col})")]
            if unit != "percent":
                out += [(pct(va, vb), "percent", f"pct_change({a}.{col}, {b}.{col})"),
                        (va / vb if vb else np.nan, "ratio", f"ratio({a}.{col}, {b}.{col})")]
    return out


def families(line, entities, limit=6):
    """Product families the line mentions: a word run (e.g. 'LANDMARK FRAME', 'hot water bottle') found in 2 to 12 names."""
    words = re.findall(r"[a-z]+", line.lower())
    padded = {e: " " + " ".join(re.findall(r"[a-z]+", e.lower())) + " " for e in entities}
    found = {}
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            gram = words[i:i + n]
            if all(w in FAMILY_STOP for w in gram) or (n == 1 and len(gram[0]) < 6):
                continue
            frag = " ".join(gram)
            members = tuple(sorted(e for e, p in padded.items() if f" {frag} " in p))
            if 2 <= len(members) <= 12 and members not in found:
                found[members] = frag.upper()
    fams = sorted(found.items(), key=lambda kv: -len(kv[1]))[:limit]
    return [(frag, list(members)) for members, frag in fams]


def group_values(table, members, label):
    """Sums, changes and shares over a group of rows (named together in a line, or one product family)."""
    t = table.set_index("Entity").loc[members]
    out = []
    for m, unit in MEASURES.items():
        tw = float(t[f"{m}_TW"].sum())
        out.append((tw, unit, f"sum({label}.{m}_TW)"))
        for per in ["LW", "LY"]:
            col = f"{m}_{per}"
            if col not in t or t[col].isna().all():
                continue
            base = float(t[col].sum())
            out += [(base, unit, f"sum({label}.{col})"), (tw - base, unit, f"diff(sum({label}.{m}_TW), sum({label}.{col}))"),
                    (pct(tw, base), "percent", f"pct_change(sum({label}.{m}_TW), sum({label}.{col}))")]
    for col in ["Revenue_TW", "Revenue_LW", "Units_TW", "Orders_TW"]:
        total = table[col].sum()
        out.append((float(t[col].sum()) / total * 100 if total else np.nan, "percent", f"share({label}.{col})"))
    out.append((float(t["Share_Pct"].sum()), "percent", f"sum({label}.Share_Pct)"))
    return out


def group_candidates(table, line, named):
    """Group values for the entities named together in the line, and for product families the line mentions."""
    out = []
    if len(named) >= 2:
        out += group_values(table, named, f"group[{'; '.join(named)}]")
    fams = families(line, table["Entity"].astype(str).tolist())
    for frag, members in fams:
        out += group_values(table, members, f"family[{frag}]")
    if len(fams) >= 2:
        union = sorted({e for _, ms in fams for e in ms})
        if len(union) <= 24:
            out += group_values(table, union, f"family[{' + '.join(f for f, _ in fams)}]")
    return out


def first_match(cands, rec, allow_bounds=False):
    if not cands:
        return None
    usable = [c for c in cands if unit_ok(c[1], rec)]
    if not usable:
        return None
    hit = np.flatnonzero(matches([c[0] for c in usable], rec, allow_bounds=allow_bounds))
    return usable[hit[0]] if len(hit) else None


def support_number(rec, table, cells, table_derived, line_derived):
    """Label one number: Supported_cell, Supported_derived or Unsupported, with evidence.

    Search order: cells of the entities named in the line, values derived from those entities and
    their groups or families, table-level aggregates (only when the line names no entity or uses an
    aggregate word), and finally any other cell of the table.
    """
    named = set(rec["line_entities"])
    own_cells = [c for c in cells if c[3] in named]
    if rec.get("period_hint"):  # "£0.00 LY": look in LY columns first, so the evidence names the right cell
        own_cells.sort(key=lambda c: not c[2].rstrip(")").endswith("_" + rec["period_hint"]))
    other_cells = [c for c in cells if c[3] not in named]
    aggregate_ok = not named or re.search(AGG_WORDS, rec["line"].lower())
    steps = [("Supported_cell", own_cells, True), ("Supported_derived", line_derived, True),
             ("Supported_derived", table_derived if aggregate_ok else [], not named),
             ("Supported_cell", other_cells, not named)]
    for label, cands, bounds in steps:
        hit = first_match(cands, rec, allow_bounds=bounds)
        if hit:
            return label, f"{hit[2]}={hit[0]:.6g}"
    return "Unsupported", ""


def gate_level1(note_text, table, note_id=""):
    """One record per numeric mention in the note."""
    entities = table["Entity"].astype(str).tolist()
    cells = cell_candidates(table)
    table_derived = table_candidates(table)
    line_cache = {}  # line index -> derived values for that line's entities, groups and families
    out = []
    for rec in extract_numbers(note_text, entities):
        if rec["skip_reason"]:
            label, evidence = "Skipped", rec["skip_reason"]
        else:
            if rec["line_idx"] not in line_cache:
                line_cache[rec["line_idx"]] = (entity_candidates(table, rec["line_entities"])
                                               + group_candidates(table, rec["line"], rec["line_entities"]))
            label, evidence = support_number(rec, table, cells, table_derived, line_cache[rec["line_idx"]])
        out.append({"note_id": note_id, "line_idx": rec["line_idx"], "sentence": rec["sentence"],
                    "raw_text": rec["raw_text"], "value": rec["value"], "unit": rec["unit"], "is_rank": rec["is_rank"],
                    "qualifier": rec["qualifier"], "direction": rec["direction"], "position": rec["position"],
                    "label": label, "evidence": evidence})
    return out

# ---------------------------------------------------------------- level 2: claim binding


def claim_frame(table):
    """Per entity, every value a claim can bind to, keyed by virtual column name. Includes a TOTAL row."""
    frame = {}
    has_ly = table["Revenue_LY"].notna().any()
    lw_total = table["Revenue_LW"].sum()
    for _, r in table.iterrows():
        f = {c: float(r[c]) if pd.notna(r[c]) else np.nan for c in COLUMN_UNIT}
        for m in MEASURES:
            f[f"{m}_WoW_Abs"] = f[f"{m}_TW"] - f[f"{m}_LW"]
            f[f"{m}_WoW_Pct"] = pct(f[f"{m}_TW"], f[f"{m}_LW"])
            f[f"{m}_WoW_Ratio"] = f[f"{m}_TW"] / f[f"{m}_LW"] if f[f"{m}_LW"] else np.nan
        f["Revenue_WoW_Pct"] = f["WoW_Pct"]
        f["Revenue_YoY_Pct"] = f["YoY_Pct"]
        f["Revenue_YoY_Abs"] = f["Revenue_TW"] - f["Revenue_LY"]
        f["Revenue_YoY_Ratio"] = f["Revenue_TW"] / f["Revenue_LY"] if f["Revenue_LY"] else np.nan
        f["Share_LW"] = f["Revenue_LW"] / lw_total * 100 if lw_total else np.nan
        f["Share_WoW_Abs"] = f["Share_Pct"] - f["Share_LW"]
        frame[str(r["Entity"])] = f
    tot = {c: float(table[c].sum()) for c in COLUMN_UNIT if c not in {"WoW_Pct", "YoY_Pct", "Share_Pct", "Rank"}}
    if not has_ly:
        tot["Revenue_LY"] = np.nan
    for m in MEASURES:
        tot[f"{m}_WoW_Abs"] = tot[f"{m}_TW"] - tot[f"{m}_LW"]
        tot[f"{m}_WoW_Pct"] = pct(tot[f"{m}_TW"], tot[f"{m}_LW"])
        tot[f"{m}_WoW_Ratio"] = tot[f"{m}_TW"] / tot[f"{m}_LW"] if tot[f"{m}_LW"] else np.nan
    tot["WoW_Pct"] = tot["Revenue_WoW_Pct"]
    tot["Revenue_YoY_Abs"] = tot["Revenue_TW"] - tot["Revenue_LY"]
    tot["Revenue_YoY_Pct"] = tot["YoY_Pct"] = pct(tot["Revenue_TW"], tot["Revenue_LY"]) if has_ly else np.nan
    tot["Revenue_YoY_Ratio"] = tot["Revenue_TW"] / tot["Revenue_LY"] if has_ly and tot["Revenue_LY"] else np.nan
    tot["Share_Pct"] = 100.0
    frame["TOTAL"] = tot
    return frame


def normalize_period(period):
    period = str(period or "TW").strip().upper().replace("-", "").replace(" ", "")
    return {"WOW": "WoW", "YOY": "YoY", "WEEKONWEEK": "WoW", "YEARONYEAR": "YoY"}.get(period, period)


def resolve_column(metric, period, unit, is_ratio=False):
    """Map (metric, period, unit) to a virtual column name, or None if the table cannot answer it."""
    metric = (metric or "revenue").lower()
    period = normalize_period(period)
    if metric == "rank":
        return "Rank"
    if metric == "share":
        return {"TW": "Share_Pct", "LW": "Share_LW", "WoW": "Share_WoW_Abs"}.get(period)
    measure = {"revenue": "Revenue", "sales": "Revenue", "units": "Units", "orders": "Orders"}.get(metric)
    if not measure:
        return None
    if period in {"TW", "LW", "LY"}:
        col = f"{measure}_{period}"
        return col if col in COLUMN_UNIT else None
    if period in {"WoW", "YoY"}:
        if measure != "Revenue" and period == "YoY":
            return None  # the table has no LY units or orders
        kind = "Ratio" if is_ratio else "Pct" if unit == "percent" else "Abs"
        return f"{measure}_{period}_{kind}"
    return None


def base_column(col):
    """The period column a change is measured from, e.g. Revenue_LW for Revenue_WoW_Pct."""
    m = re.match(r"(Revenue|Units|Orders)_(WoW|YoY)_", col or "")
    return f"{m.group(1)}_{'LW' if m.group(2) == 'WoW' else 'LY'}" if m else None


def clean_name(name):
    name = re.sub(r"\(.*?\)", " ", str(name)).strip()
    name = re.sub(r"^(the)\s+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"['’]s$", "", name)
    name = re.sub(r"\s+(lines?|products?|item|sales|revenue)$", "", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip()


def match_entities(name, entities):
    """Candidate rows for an entity name: exact, alias, unique-owner token match, or fuzzy ties.

    Several candidates mean the name is ambiguous (e.g. 'Hanging heart holder' fits the WHITE and the
    RED product); the claim is then checked against each of them.
    """
    if not name:
        return []
    low = str(name).strip().lower()
    if low in TOTAL_WORDS or low.startswith("total") or re.search(
            r"\b(all|every)\b.*\b(products|countries|markets|entities|rows|lines|items)\b", low):
        return ["TOTAL"]
    for variant in [low, clean_name(name).lower()] + [p.strip().lower() for p in re.findall(r"\((.*?)\)", str(name))]:
        for e in entities:
            if e.lower() == variant:
                return [e]
        if ENTITY_ALIASES.get(variant) in entities:
            return [ENTITY_ALIASES[variant]]
    base = clean_name(name).lower()
    tokens = set(re.findall(r"[a-z0-9]+", base))
    if tokens:
        owners = [e for e in entities if tokens <= set(re.findall(r"[a-z0-9]+", e.lower()))]
        if owners:
            return owners if len(owners) <= 5 else []  # a name that fits many rows cannot be bound
    scored = process.extract(base, entities, scorer=fuzz.WRatio, processor=str.lower, score_cutoff=85, limit=None)
    if not scored:
        return []
    best = scored[0][1]
    return [e for e, s, _ in scored if s >= best - 2][:5]


def match_entity(name, entities):
    """The single best row for a name (first candidate of match_entities), or None."""
    rows = match_entities(name, entities)
    return rows[0] if rows else None


def exact_entity(name, entities):
    """The row an entity name names exactly (case-insensitive, or by alias), else None."""
    low = str(name or "").strip().lower()
    for e in entities:
        if e.lower() == low:
            return e
    alias = ENTITY_ALIASES.get(low)
    return alias if alias in entities else None


def is_group_name(name):
    """The extractor marks figures about several entities with GROUP (see prompts/extract_system.txt)."""
    low = str(name or "").strip().lower()
    return low == "group" or low.startswith("group") or low in GROUP_WORDS


def check_row(rec, row, metric, period, frame, line_entities):
    """Verify a parsed value against one candidate row. Returns (label, evidence)."""
    col = resolve_column(metric, period, rec["unit"], rec.get("is_ratio", False))
    targets = [col] if col else []  # columns the claim may bind to
    if col and rec["unit"] == "plain" and col.endswith("_Abs") and not rec.get("is_ratio"):
        targets.append(col.replace("_Abs", "_Pct"))  # a unitless change could be either
    live = [c for c in targets if pd.notna(frame[row].get(c, np.nan))]
    if not live:
        same_row = [c for c, v in frame[row].items() if matches([v], rec, allow_bounds=False)[0]]
        if same_row:
            return "Wrong_metric", f"value is {row}.{same_row[0]}={frame[row][same_row[0]]:.6g}; claimed {col}"
        base = base_column(col)
        if base and frame[row].get(base) == 0:
            return "Wrong_value", f"{row}.{base}=0, so no {period} change exists"
        return "Unverifiable", f"no value in the table for metric={metric} period={period}"
    for c in live:
        if matches([frame[row][c]], rec)[0]:
            return "Correct", f"{row}.{c}={frame[row][c]:.6g}"
    col = live[0]
    if rec["signed"] and any(matches([frame[row][c]], rec, force_signed=False)[0] for c in live):
        return "Wrong_value", f"{row}.{col}={frame[row][col]:.6g} (direction wrong)"
    for c, v in frame[row].items():  # same row, another column: a period or metric mix-up
        if c not in live and matches([v], rec, allow_bounds=False)[0]:
            return "Wrong_metric", f"value is {row}.{c}={v:.6g}; claimed {col}"
    others = [o for o, f in frame.items() if o != row and any(matches([f.get(c, np.nan)], rec, allow_bounds=False)[0] for c in live)]
    named = [o for o in others if o in line_entities]
    if named:
        return "Wrong_entity", f"value belongs to {named[0]}.{col}={frame[named[0]][col]:.6g}; claimed {row}"
    if len(others) == 1 and col != "Rank":
        return "Wrong_entity", f"value belongs to {others[0]}.{col}={frame[others[0]][col]:.6g}; claimed {row}"
    note = f"; value also matches {len(others)} other rows" if others else ""
    return "Wrong_value", f"{row}.{col}={frame[row][col]:.6g}{note}"


def group_value_match(rec, table, line, named):
    """A group, family, top-k, rest-of-table or table-level value (Level 1's library for this line) that the value matches."""
    named = sorted(named)
    cands = group_candidates(table, line, named) + entity_candidates(table, named) + table_candidates(table)
    return first_match(cands, rec)


def group_cue(line, named, table):
    """Does the line talk about a subset of rows (a family, several named rows, the top k)?"""
    return bool(families(line, table["Entity"].astype(str).tolist())) or len(named) >= 2 or bool(
        re.search(r"\b(top|combined|together|lines|colourways|these|both|between them|family)\b", line.lower()))


def verify_claim(claim, table, frame=None, line_entities=(), line=""):
    """Label one claim: Correct, Wrong_entity, Wrong_metric, Wrong_value or Unverifiable."""
    frame = frame or claim_frame(table)
    entities = table["Entity"].astype(str).tolist()
    rec = claim.get("parsed") or parse_value_text(claim.get("value_text", ""))
    if rec is None:
        return "Unverifiable", "value not parseable"
    rec = dict(rec)
    metric = (claim.get("metric") or "revenue").lower()
    period = normalize_period(claim.get("period"))
    if claim.get("qualifier") in {"approx", "over", "under"}:
        rec["qualifier"] = claim["qualifier"]
    # A direction only signs a change; "fell to £1,632.90 TW" states a level.
    if claim.get("direction") in {"up", "down"} and not rec["signed"] and period in {"WoW", "YoY"} and metric != "rank":
        rec.update(cited=abs(rec["cited"]) * (1 if claim["direction"] == "up" else -1), signed=True)
    if period in {"TW", "LW", "LY"} or metric == "rank":
        rec.update(cited=abs(rec["cited"]), signed=False)
    if not exact_entity(claim.get("entity"), entities) and is_group_name(claim.get("entity")):
        if not line:
            return "Unverifiable", "claim about a group of entities"
        hit = group_value_match(rec, table, line, line_entities)
        if hit:
            return "Correct", f"group figure: {hit[2]}={hit[0]:.6g}"
        return "Wrong_value", "group figure matches no group, family, top-k or table total that the line refers to"
    rows = match_entities(claim.get("entity"), entities)
    if not rows:
        return "Unverifiable", f"entity not found or too ambiguous: {claim.get('entity')}"
    results = [(r, check_row(rec, r, metric, period, frame, line_entities)) for r in rows]
    if rows == ["TOTAL"] and results[0][1][0] != "Correct" and line and group_cue(line, line_entities, table):
        hit = group_value_match(rec, table, line, line_entities)  # "hot water bottle lines totalled £12,111" is a subset
        if hit:
            return "Correct", f"group figure (extractor said TOTAL): {hit[2]}={hit[0]:.6g}"
    for r, (label, evidence) in results:
        if label == "Correct":
            return label, evidence + (f" (name fits {len(rows)} rows)" if len(rows) > 1 else "")
    if len(rows) > 1:  # a short name that fits several rows and matches none of them cannot be bound
        return "Unverifiable", f"ambiguous name fits {', '.join(rows)}; no match"
    return results[0][1]


def text_hints(claim, line, occurrence):
    """Period or metric stated right next to the value in the note ('£2,729.22 LW', '8 orders', '17.5% WoW')."""
    vt = str(claim.get("value_text", "")).strip()
    if not vt or not line:
        return {}
    # only standalone occurrences: "33" must not be found inside "£680.33"
    starts = [m.start() for m in re.finditer(r"(?<![\d.,])" + re.escape(vt) + r"(?!\d|[.,]\d)", line)]
    if not starts:
        return {}
    pos = starts[min(occurrence, len(starts) - 1)]
    after = tokens_after(line, pos + len(vt), 2)
    hints = {"qualifier": hedge_of(line, pos)}
    raw_before = tokens_before(line, pos, 4)
    before3 = [w for w in raw_before if w not in {"a", "an", "of", "by", "some", "the", "at", "absolute"}]
    after3 = tokens_after(line, pos + len(vt), 3)
    is_level = bool(raw_before) and raw_before[-1] in {"from", "to", "reaching", "reached"}  # "from £10,902.59 on LW"
    if before3 and before3[-1] in CHANGE_NOUNS and not is_level:  # "a decline of £51.1k", "swing at £1,492.90"
        hints["change"] = "YoY" if any(w in {"yoy", "ly", "year"} for w in after3) else "WoW"
    followed_by_value = len(after3) == 3 and bool(re.match(r"[£$€\d]", after3[2]))  # "versus LW £3,792.77" compares two levels
    if len(after) == 2 and after[0] in COMPARE_WORDS and after[1] in {"lw", "ly"} and not is_level and not followed_by_value:
        hints["change"] = "WoW" if after[1] == "lw" else "YoY"  # "£16.8k versus LW"
    labels = {"tw": "TW", "lw": "LW", "ly": "LY", "wow": "WoW", "yoy": "YoY"}
    label_before = re.search(r"\b(TW|LW|LY)\s*:?\s*$", line[:pos])  # "LW: £3,792.77", but not "TW (-£3.5k)"
    if after and after[0] in labels and re.match(r"\s+\w", line[pos + len(vt):]):
        hints["period"] = labels[after[0]]
    elif label_before:
        hints["period"] = label_before.group(1)
    if after and re.fullmatch(r"units?", after[0]):
        hints["metric"] = "units"
    elif after and re.fullmatch(r"orders?|invoices?", after[0]):
        hints["metric"] = "orders"
    elif after and after[0] == "share":
        hints["metric"] = "share"
    return hints


def apply_hints(claim, line, occurrence):
    """Let an explicit label next to the value override the extractor, and record that it did."""
    hints = text_hints(claim, line, occurrence)
    notes = []
    parsed = claim.get("parsed") or parse_value_text(claim.get("value_text", ""))
    unit = parsed["unit"] if parsed else "plain"
    if hints.get("qualifier", "none") != "none" and claim.get("qualifier", "none") in {None, "", "none"}:
        notes.append(f"hedge '{hints['qualifier']}' from text")
        claim = dict(claim, qualifier=hints["qualifier"])
    p = hints.get("period")
    if not p and hints.get("change") and unit in {"currency", "count"} and normalize_period(claim.get("period")) in {"TW", "LW", "LY"}:
        notes.append(f"period {normalize_period(claim.get('period'))}->{hints['change']} from change wording")
        claim = dict(claim, period=hints["change"])
    if parsed and parsed.get("is_ratio"):
        p = None  # "125.9x its £45.57 LW": the label belongs to the next value
    if p and p != normalize_period(claim.get("period")):
        level_label = p in {"TW", "LW", "LY"}
        if (level_label and unit != "percent") or (not level_label and unit == "percent"):
            notes.append(f"period {normalize_period(claim.get('period'))}->{p} from text")
            claim = dict(claim, period=p)
    m = hints.get("metric")
    if m and m != str(claim.get("metric", "")).lower() and not (m == "share" and unit != "percent"):
        notes.append(f"metric {claim.get('metric')}->{m} from text")
        claim = dict(claim, metric=m)
    return claim, "; ".join(notes)


def regex_claims(note_text, table):
    """Rule-based claim extractor used for the mock model."""
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
    """LLM claim extractor with a fixed JSON schema. Returns (claims or None, last raw response).

    If the reply is not valid JSON, it asks once more with a stricter reminder; if that also fails,
    it returns None so the caller can exclude the note instead of mixing in a different extractor.
    """
    from config import read_prompt
    lines = [l for l in note_text.splitlines() if l.strip()]
    numbered = "\n".join(f"{i}: {l}" for i, l in enumerate(lines))
    user = read_prompt("extract_user.txt").format(note_lines=numbered)
    system = read_prompt("extract_system.txt")
    resp = None
    for attempt in range(2):
        prompt = user if attempt == 0 else user + "\n\nReturn only the JSON object, with no other text."
        resp = complete_fn(extractor_model, system, prompt, temperature=0, max_tokens=16000)  # long notes carry 50+ claims
        text = resp["text"]
        try:
            payload = json.loads(text[text.find("{"): text.rfind("}") + 1])
            claims = payload["claims"]
            if not isinstance(claims, list):
                raise TypeError("claims is not a list")
        except (ValueError, KeyError, TypeError):
            continue
        claims = [c for c in claims if isinstance(c, dict)]
        for c in claims:
            for k in ("entity", "metric", "period", "value_text", "direction", "qualifier"):
                v = c.get(k)
                if v is not None and not isinstance(v, str):  # a model may return a number or a list here
                    c[k] = str(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
            try:
                c["line_idx"] = int(c.get("line", c.get("line_idx", 0)) or 0)
            except (TypeError, ValueError):
                c["line_idx"] = 0
        return claims, resp
    return None, resp


def gate_level2(note_text, table, note_id="", claims=None):
    """One record per claim. Claims come from the LLM extractor or, if None, the regex extractor."""
    frame = claim_frame(table)
    entities = table["Entity"].astype(str).tolist()
    lines = [l for l in note_text.splitlines() if l.strip()]
    line_entities = {i: {e for _, _, e in find_entity_spans(l, entities)} for i, l in enumerate(lines)}
    claims = regex_claims(note_text, table) if claims is None else claims
    seen = {}  # (line, value_text) -> how many times this value was already used in the line
    out = []
    for c in claims:
        li = c.get("line_idx", 0)
        line = lines[li] if 0 <= li < len(lines) else ""
        key = (li, str(c.get("value_text", "")))
        occurrence = seen.get(key, 0)
        seen[key] = occurrence + 1
        c, adjusted = apply_hints(c, line, occurrence)
        label, evidence = verify_claim(c, table, frame, line_entities.get(li, set()), line)
        out.append({"note_id": note_id, "line_idx": li, "entity": c.get("entity", ""),
                    "metric": c.get("metric", ""), "period": normalize_period(c.get("period")),
                    "value_text": c.get("value_text", ""), "direction": c.get("direction", "none"),
                    "qualifier": c.get("qualifier", "none") or "none", "adjusted": adjusted,
                    "label": label, "evidence": evidence})
    return out

# ---------------------------------------------------------------- note-level summary

NUMBER_LABELS = ["Supported_cell", "Supported_derived", "Unsupported", "Skipped"]
CLAIM_LABELS = ["Correct", "Wrong_entity", "Wrong_metric", "Wrong_value", "Unverifiable"]
CLAIM_ERRORS = ["Wrong_entity", "Wrong_metric", "Wrong_value"]
BINDING_ERRORS = ["Wrong_entity", "Wrong_metric"]  # the number exists but is attached to the wrong entity, metric or period


def summarize(numbers, claims):
    counts = {f"n_{l}": sum(r["label"] == l for r in numbers) for l in NUMBER_LABELS}
    checked = counts["n_Supported_cell"] + counts["n_Supported_derived"] + counts["n_Unsupported"]
    counts["n_numbers_checked"] = checked
    counts["unsupported_rate"] = counts["n_Unsupported"] / checked if checked else np.nan
    counts.update({f"n_{l}": sum(r["label"] == l for r in claims) for l in CLAIM_LABELS})
    verifiable = sum(counts[f"n_{l}"] for l in ["Correct"] + CLAIM_ERRORS)
    counts["n_claims_verifiable"] = verifiable
    counts["claim_error_rate"] = sum(counts[f"n_{l}"] for l in CLAIM_ERRORS) / verifiable if verifiable else np.nan
    counts["binding_error_rate"] = sum(counts[f"n_{l}"] for l in BINDING_ERRORS) / verifiable if verifiable else np.nan
    return counts
