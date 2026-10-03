#!/usr/bin/env python3
"""NumGate held-out set 2, Level 1 (paper/heldout2/L1): note and truth generator.

Protocol position: written and run BEFORE the gate was run and BEFORE src/number_gate.py,
docs/method.md, README.md, tests/, runs/, results/ or other paper/ folders were opened.
Inputs read: prompts/note_rules.txt, prompts/extract_system.txt, prompts/columns_guide.txt and the
seven assigned tables in data/tables/.

What it does
  * Builds 28 notes (4 per table) in the style of prompts/note_rules.txt. Every numeric mention in
    every line is registered with a truth label:
        C   correct measurement            W   deliberately wrong measurement
        N   not a measurement (week label, year, digits inside a product name)
        CNT count of rows ("top 3", "the 7 DOOR MAT lines", "the other 88 products")
  * Each C/W measurement carries a small expression tree over table cells. The builder evaluates it
    with pandas/float; verify() re-evaluates it independently with the csv module and Decimal, then
    checks the label with a half-up rounding test at the precision written in the note.
  * Wrong items are typed: drift, direction, other_row, other_period (value from another period or
    metric), rounding (wrong rounding at the cited precision), group_total.
  * Completeness: every digit in every line must fall inside a registered mention, and number words
    (two, three, ...) must be registered too (flagged is_word=1).

Pre-registered counting rule: the primary mention count is over digit-form mentions (is_word=0).
Word-form counts ("the three bunting lines") are registered as CNT with is_word=1 and reported
separately, because they contain no digits.

Outputs (this folder only): notes.jsonl, truth.csv. FROZEN.json is written by freeze.py.
"""
import csv
import json
import math
import re
import sys
from decimal import Decimal, ROUND_HALF_UP, ROUND_FLOOR, getcontext
from pathlib import Path

import pandas as pd

getcontext().prec = 50
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper" / "heldout2" / "L1"
TABLE_DIR = ROOT / "data" / "tables"

TABLES = ["small_country_2010-W21", "small_product_2011-W47", "small_product_2010-W07",
          "medium_product_2011-W10", "medium_product_2010-W05", "large_product_2011-W11",
          "large_product_2010-W16"]

NUMBER_WORDS = r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twice|half|halved|" \
               r"double|doubled|triple|tripled|quarter|third|dozen|quadrupled)\b"
WORD_VAL = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}

# ----------------------------------------------------------------------------------------------
# expression trees
# ----------------------------------------------------------------------------------------------
def X_cell(e, c): return {"op": "cell", "ent": e, "col": c}
def X_sum(c, ents): return {"op": "sum", "col": c, "ents": ents}
def X_mean(c, ents): return {"op": "mean", "col": c, "ents": ents}
def X_sub(a, b): return {"op": "sub", "a": a, "b": b}
def X_div(a, b): return {"op": "div", "a": a, "b": b}
def X_chg(a, b): return {"op": "chg", "a": a, "b": b}      # 100 * (a / b - 1)
def X_shr(a, b): return {"op": "shr", "a": a, "b": b}      # 100 * a / b


def ev_float(x, T):
    op = x["op"]
    if op == "cell":
        v = T.loc[x["ent"], x["col"]]
        assert not pd.isna(v), ("blank cell", x)
        return float(v)
    if op in ("sum", "mean"):
        ents = list(T.index) if x["ents"] == "ALL" else x["ents"]
        vals = []
        for e in ents:
            v = T.loc[e, x["col"]]
            assert not pd.isna(v), ("blank cell", e, x["col"])
            vals.append(float(v))
        s = math.fsum(vals)
        return s if op == "sum" else s / len(vals)
    a, b = ev_float(x["a"], T), ev_float(x["b"], T)
    return {"sub": a - b, "div": a / b, "chg": 100 * (a / b - 1), "shr": 100 * a / b}[op]


def subst_ent(x, new):
    x = json.loads(json.dumps(x))
    def walk(n):
        if n["op"] == "cell":
            n["ent"] = new
        elif n["op"] in ("sub", "div", "chg", "shr"):
            walk(n["a"]); walk(n["b"])
        else:
            raise AssertionError("row substitution only on cell-based expressions")
    walk(x)
    return x


def subst_col(x, mapping):
    x = json.loads(json.dumps(x))
    def walk(n):
        if n["op"] in ("cell", "sum", "mean"):
            if n["col"] in mapping:
                n["col"] = mapping[n["col"]]
        else:
            walk(n["a"]); walk(n["b"])
    walk(x)
    return x


# ----------------------------------------------------------------------------------------------
# formatting (half-up, via Decimal)
# ----------------------------------------------------------------------------------------------
FMTS = {  # name: (prefix, suffix, scale, decimals, commas)
    "gbp2": ("£", "", 1, 2, True), "gbp0": ("£", "", 1, 0, True),
    "gbpk0": ("£", "k", 1000, 0, False), "gbpk1": ("£", "k", 1000, 1, False),
    "gbpk2": ("£", "k", 1000, 2, False),
    "pct0": ("", "%", 1, 0, True), "pct1": ("", "%", 1, 1, True),
    "int": ("", "", 1, 0, True), "dec1": ("", "", 1, 1, False), "dec2": ("", "", 1, 2, False),
    "x1": ("", "x", 1, 1, False), "ord": ("", "", 1, 0, False),
}


def Dd(v):
    return v if isinstance(v, Decimal) else Decimal(repr(float(v)))


def unit_of(fmt):
    _, _, scale, dp, _ = FMTS[fmt]
    return Decimal(scale) * Decimal(1).scaleb(-dp)


def ordinal(n):
    n = int(n)
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def render(v, fmt, signed=False):
    pre, suf, scale, dp, commas = FMTS[fmt]
    d = Dd(v) / Decimal(scale)
    q = d.quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP)
    neg = q < 0
    q = abs(q)
    if fmt == "ord":
        body = ordinal(q)
    else:
        body = f"{q:,.{dp}f}" if commas else f"{q:.{dp}f}"
    s = f"{pre}{body}{suf}"
    if signed:
        s = ("-" if neg else "+") + s
    else:
        assert not neg, ("negative value rendered unsigned", v, fmt)
    return s


def wrong_round(true, fmt):
    """A value one unit off at the displayed precision, in the direction a sloppy rounding makes:
    truncate when half-up would round up, round up when half-up would round down."""
    unit = unit_of(fmt)
    d = abs(Dd(true)) / unit
    fl = d.to_integral_value(rounding=ROUND_FLOOR)
    frac = d - fl
    assert frac != 0, "rounding error needs a value with a discarded remainder"
    w = fl if frac >= Decimal("0.5") else fl + 1
    sign = -1 if Dd(true) < 0 else 1
    return float(sign * w * unit)


# ----------------------------------------------------------------------------------------------
# builder
# ----------------------------------------------------------------------------------------------
class Builder:
    def __init__(self):
        self.notes, self.rows, self.tables = [], [], {}
        self.buf, self.cur, self.T = [], None, None

    def table(self, tid):
        if tid not in self.tables:
            T = pd.read_csv(TABLE_DIR / f"{tid}.csv")
            assert T.Entity.is_unique
            self.tables[tid] = T.set_index("Entity")
        return self.tables[tid]

    def start(self, note_id, tid, style):
        assert not self.buf
        self.cur = dict(note_id=note_id, table_id=tid, style=style, lines=[])
        self.T = self.table(tid)

    def reg(self, anchor, pieces):
        self.buf.append((anchor, pieces))

    def line(self, body):
        text = "- " + body
        assert "—" not in text and "–" not in text, "no em/en dashes"
        pos, ms = 0, []
        for anchor, pieces in self.buf:
            i = text.find(anchor, pos)
            assert i >= 0, ("anchor not found", anchor, text)
            for off, m in pieces:
                m["start"], m["end"] = i + off, i + off + len(m["value_text"])
                assert text[m["start"]:m["end"]] == m["value_text"], (m, text)
                ms.append(m)
            pos = i + len(anchor)
        self.buf = []
        covered = set()
        for m in ms:
            rng = set(range(m["start"], m["end"]))
            assert not (rng & covered), ("overlapping mentions", text)
            covered |= rng
        for k, ch in enumerate(text):
            if ch.isdigit():
                assert k in covered, ("unregistered digit", k, text)
        for mm in re.finditer(NUMBER_WORDS, text):  # prose words only; product names are upper case
            assert any(m["start"] == mm.start() for m in ms), ("unregistered number word", mm.group(0), text)
        ln = len(self.cur["lines"]) + 1
        ms.sort(key=lambda m: m["start"])
        for j, m in enumerate(ms):
            m.update(note_id=self.cur["note_id"], table_id=self.cur["table_id"], line=ln, idx=j)
        self.cur["lines"].append(text)
        self.rows.extend(ms)

    def end(self):
        n = self.cur
        assert 6 <= len(n["lines"]) <= 8, (n["note_id"], len(n["lines"]))
        self.notes.append(dict(note_id=n["note_id"], table_id=n["table_id"], style=n["style"],
                               n_lines=len(n["lines"]), text="\n".join(n["lines"])))
        self.cur = None

    # -- mention constructors --------------------------------------------------------------
    def nonmeasure(self, text, label, kind, entity="", count=None, is_word=0):
        pieces = []
        for m in re.finditer(r"\d+", text):
            pieces.append((m.start(), self._row(value_text=m.group(0), label=label, kind=kind,
                                                entity=entity, is_word=is_word, count=count)))
        if is_word:
            pieces.append((0, self._row(value_text=text, label=label, kind=kind, entity=entity,
                                        is_word=1, count=count)))
        if pieces:
            self.reg(text, pieces)
        return text

    def _row(self, **kw):
        base = dict(value_text="", label="", kind="", entity="", metric="", period="", qualifier="none",
                    direction_text="", true_value="", cited_value="", precision="", error_type="",
                    error_src="", expr="", is_word=0, count="")
        base.update({k: ("" if v is None else v) for k, v in kw.items()})
        return base

    def meas(self, expr, fmt, *, metric, period, entity, kind, err=None, q="none", show=None,
             signed=False, verbs=None, dirw=None):
        T = self.T
        true = ev_float(expr, T)
        etype, esrc, cited = "", "", true
        if err is not None:
            if err == "round":
                basis = true
                if expr["op"] == "cell" and expr["col"] in ("Share_Pct", "WoW_Pct", "YoY_Pct"):
                    # the cell is already rounded to 0.1; decide the sloppy direction from the raw value
                    row = T.loc[expr["ent"]]
                    if expr["col"] == "Share_Pct":
                        basis = 100 * row["Revenue_TW"] / T["Revenue_TW"].sum()
                    else:
                        den = row["Revenue_LW" if expr["col"] == "WoW_Pct" else "Revenue_LY"]
                        basis = 100 * (row["Revenue_TW"] / den - 1)
                    assert render(basis, fmt) == render(true, fmt), "raw and cell disagree"
                etype, cited = "rounding", wrong_round(basis, fmt)
            elif err == "dir":
                etype = "direction"
            elif err[0] == "drift":
                etype, cited = "drift", float(err[1])
            elif err[0] == "row":
                x2 = subst_ent(expr, err[1])
                etype, cited, esrc = "other_row", ev_float(x2, T), json.dumps(x2)
            elif err[0] == "col":
                mapping = err[1] if isinstance(err[1], dict) else {expr["col"]: err[1]}
                x2 = subst_col(expr, mapping)
                etype, cited, esrc = "other_period", ev_float(x2, T), json.dumps(x2)
            elif err[0] == "grp":       # wrong membership for every sum/mean in the expression
                x2 = json.loads(json.dumps(expr))
                def walk(n):
                    if n["op"] in ("sum", "mean"):
                        n["ents"] = err[1]
                    elif n["op"] != "cell":
                        walk(n["a"]); walk(n["b"])
                walk(x2)
                etype, cited, esrc = "group_total", ev_float(x2, T), json.dumps(x2)
            elif err[0] == "grpx":      # explicit alternative expression
                etype, cited, esrc = "group_total", ev_float(err[1], T), json.dumps(err[1])
            elif err[0] == "grpv":      # arithmetic slip in a group total
                etype, cited = "group_total", float(err[1])
            else:
                raise ValueError(err)
        if show is not None:
            assert err is None
            cited = float(show)
        direction_text, prefix = "", ""
        if verbs:
            up = true > 0
            if etype == "direction":
                up = not up
            prefix = (verbs[0] if up else verbs[1]) + " "
            direction_text = "up" if up else "down"
            disp = abs(cited)
            vt = render(disp, fmt)
        elif dirw:
            direction_text = dirw
            disp = abs(cited)
            vt = render(disp, fmt)
        elif signed:
            disp = -cited if etype == "direction" else cited
            vt = render(disp, fmt, signed=True)
            direction_text = "sign"
        else:
            assert etype != "direction"
            disp = cited
            vt = render(disp, fmt)
        row = self._row(value_text=vt, label="W" if etype else "C", kind=kind, entity=entity, metric=metric,
                        period=period, qualifier=q, direction_text=direction_text, true_value=repr(true),
                        cited_value=repr(float(disp)), precision=str(unit_of(fmt)), error_type=etype,
                        error_src=esrc, expr=json.dumps(expr))
        self.reg(vt, [(0, row)])
        return prefix + vt


B = Builder()

# ----------------------------------------------------------------------------------------------
# note-writing helpers (bound to the current table B.T)
# ----------------------------------------------------------------------------------------------
LWMAP = {"Revenue_TW": "Revenue_LW", "Orders_TW": "Orders_LW", "Units_TW": "Units_LW"}


def E(name):
    assert name in B.T.index, ("unknown entity", name)
    return B.nonmeasure(name, "N", "product_digit", entity=name)


def NM(text, kind="week_label"):
    return B.nonmeasure(text, "N", kind)


def CNT(k, members=None, word=False):
    if members is not None:
        mem = list(B.T.index) if members == "ALL" else members
        assert len(mem) == k, ("count mismatch", k, len(mem))
        for e in mem:
            assert e in B.T.index, e
    if word:
        w = {v: k2 for k2, v in WORD_VAL.items()}[k]
        return B.nonmeasure(w, "CNT", "row_count_word", count=k, is_word=1)
    return B.nonmeasure(str(k), "CNT", "row_count", count=k)


def _ent(ents):
    return "TOTAL" if ents == "ALL" else "GROUP"


def R(e, per="TW", f="gbp2", **kw):
    return B.meas(X_cell(e, "Revenue_" + per), f, metric="revenue", period=per, entity=e, kind="money", **kw)


def U(e, per="TW", f="int", **kw):
    return B.meas(X_cell(e, "Units_" + per), f, metric="units", period=per, entity=e, kind="units", **kw)


def O(e, per="TW", f="int", **kw):
    return B.meas(X_cell(e, "Orders_" + per), f, metric="orders", period=per, entity=e, kind="orders", **kw)


def WOW(e, f="pct1", **kw):
    return B.meas(X_cell(e, "WoW_Pct"), f, metric="revenue", period="WoW", entity=e, kind="pct_change", **kw)


def YOY(e, f="pct1", **kw):
    return B.meas(X_cell(e, "YoY_Pct"), f, metric="revenue", period="YoY", entity=e, kind="pct_change", **kw)


def SH(e, f="pct1", **kw):
    return B.meas(X_cell(e, "Share_Pct"), f, metric="share", period="TW", entity=e, kind="share", **kw)


def RK(e, f="int", **kw):
    return B.meas(X_cell(e, "Rank"), f, metric="rank", period="TW", entity=e, kind="rank", **kw)


def SUM(col, ents, f=None, **kw):
    metric = col.split("_")[0].lower()
    per = col.split("_")[1]
    if f is None:
        f = "gbp2" if metric == "revenue" else "int"
    kind = "money" if metric == "revenue" else metric
    return B.meas(X_sum(col, ents), f, metric=metric, period=per, entity=_ent(ents), kind="group_" + kind, **kw)


def GSH(ents, f="pct1", **kw):
    return B.meas(X_shr(X_sum("Revenue_TW", ents), X_sum("Revenue_TW", "ALL")), f, metric="share",
                  period="TW", entity=_ent(ents), kind="group_share", **kw)


def GCHG(ents, a="Revenue_TW", b="Revenue_LW", f="pct1", **kw):
    per = {"Revenue_LW": "WoW", "Revenue_LY": "YoY"}[b]
    return B.meas(X_chg(X_sum(a, ents), X_sum(b, ents)), f, metric="revenue", period=per,
                  entity=_ent(ents), kind="group_pct_change", **kw)


def AVG(e, per="TW", f="gbp2", **kw):
    x = X_div(X_cell(e, "Revenue_" + per), X_cell(e, "Orders_" + per))
    return B.meas(x, f, metric="revenue_per_order", period=per, entity=e, kind="avg_order_value", **kw)


def UPO(e, per="TW", f="dec1", **kw):
    x = X_div(X_cell(e, "Units_" + per), X_cell(e, "Orders_" + per))
    return B.meas(x, f, metric="units_per_order", period=per, entity=e, kind="units_per_order", **kw)


def DIFF(e1, e2, col="Revenue_TW", f="gbp2", **kw):
    return B.meas(X_sub(X_cell(e1, col), X_cell(e2, col)), f, metric="revenue_gap", period="TW",
                  entity=f"{e1} | {e2}", kind="difference", **kw)


def ABSCHG(e, a="Revenue_TW", b="Revenue_LW", f="gbp2", **kw):
    return B.meas(X_sub(X_cell(e, a), X_cell(e, b)), f, metric="revenue", period="WoW", entity=e,
                  kind="abs_change", **kw)


def DCHG(e, a="Units_TW", b="Units_LW", f="pct1", **kw):
    metric = a.split("_")[0].lower()
    return B.meas(X_chg(X_cell(e, a), X_cell(e, b)), f, metric=metric, period="WoW", entity=e,
                  kind="derived_pct_change", **kw)


def RATIO(xa, xb, entity, f="x1", **kw):
    return B.meas(X_div(xa, xb), f, metric="ratio", period="TW", entity=entity, kind="ratio", **kw)


def MEAN(col, ents, f="gbp2", **kw):
    return B.meas(X_mean(col, ents), f, metric="revenue_mean", period=col.split("_")[1], entity=_ent(ents),
                  kind="group_mean", **kw)


def SHDIFF(e1, e2, f="dec1", **kw):
    return B.meas(X_sub(X_cell(e1, "Share_Pct"), X_cell(e2, "Share_Pct")), f, metric="share_gap",
                  period="TW", entity=f"{e1} | {e2}", kind="share_points", **kw)


def top(k): return list(B.T.index[:k])
def rest(k): return list(B.T.index[k:])
def fam(sub): return [e for e in B.T.index if sub in e]
def c(e, col): return X_cell(e, col)


# ----------------------------------------------------------------------------------------------
# notes
# ----------------------------------------------------------------------------------------------
def notes_small_country():
    tid = "small_country_2010-W21"
    UK, CI, IE, NL, CH, FR, DE = ("United Kingdom", "Channel Islands", "EIRE", "Netherlands",
                                  "Switzerland", "France", "Germany")

    B.start("SC-1", tid, "terse_kpi")
    B.line(f"{UK}: {R(UK)} TW vs {R(UK, 'LW')} LW, {WOW(UK, signed=True)} WoW, {SH(UK)} share")
    B.line(f"{CI}: {R(CI)} TW, {WOW(CI, signed=True, err=('row', FR))} WoW on {O(CI)} orders ({O(CI, 'LW')} LW)")
    B.line(f"{IE}: {R(IE, err=('col', 'Revenue_LW'))} TW, {WOW(IE, signed=True)} WoW, {U(IE)} units")
    B.line(f"{NL}: {R(NL)} TW from {U(NL)} units and {O(NL)} orders, new versus LW")
    B.line(f"{CH}: {R(CH)} TW vs {R(CH, 'LW')} LW, {WOW(CH, signed=True, err=('drift', -80.2))} WoW")
    B.line(f"{FR}: {R(FR)} TW, {WOW(FR, signed=True)} WoW; units {U(FR)} TW vs {U(FR, 'LW', err=('drift', 106))} LW")
    B.line(f"{DE}: {R(DE)} TW, {WOW(DE, signed=True)} WoW, {SH(DE, err='round')} share, {O(DE)} orders")
    B.line(f"All {CNT(7, 'ALL')} countries: {SUM('Revenue_TW', 'ALL')} TW vs "
           f"{SUM('Revenue_LW', 'ALL', err=('grpv', 154045.82))} LW, {GCHG('ALL', signed=True)} WoW")
    B.end()

    nonUK = [CI, IE, NL, CH, FR, DE]
    B.start("SC-2", tid, "narrative")
    B.line(f"The United Kingdom generated {R(UK, f='gbpk1')} in {NM('2010-W21')}, "
           f"{WOW(UK, verbs=('up', 'down'))} on LW and {SH(UK)} of TW revenue.")
    B.line(f"Channel Islands revenue rose to {R(CI)} TW from {R(CI, 'LW', err=('drift', 872.80))} LW, on {U(CI)} units.")
    B.line(f"Switzerland revenue {WOW(CH, verbs=('rose', 'fell'))} WoW to {R(CH)} TW, with orders down from "
           f"{O(CH, 'LW')} LW to {O(CH)} TW.")
    B.line(f"Germany was the smallest market at {R(DE)} TW, {WOW(DE, verbs=('up', 'down'))} WoW, on {O(DE)} orders.")
    B.line(f"EIRE revenue per order was {AVG(IE)} TW against {AVG(IE, 'LW', err=('row', DE))} LW.")
    B.line(f"Outside the United Kingdom, the other {CNT(6, nonUK)} countries took "
           f"{SUM('Revenue_TW', nonUK, err=('grp', [CI, IE, CH, FR, DE]))} TW, {GCHG(nonUK, verbs=('up', 'down'))} WoW.")
    B.line(f"France revenue {WOW(FR, verbs=('grew', 'fell'))} WoW to {R(FR)} TW on {O(FR, err=('row', DE))} orders.")
    B.line(f"UK units {DCHG(UK, verbs=('rose', 'fell'), err='dir')} WoW to {U(UK)} from {U(UK, 'LW')} LW.")
    B.end()

    B.start("SC-3", tid, "comparison")
    B.line(f"United Kingdom revenue of {R(UK)} TW was "
           f"{RATIO(c(UK, 'Revenue_TW'), c(CI, 'Revenue_TW'), f'{UK} / {CI}', err='round')} that of Channel Islands ({R(CI)})")
    B.line(f"Channel Islands moved up to {RK(CI, f='ord')} place, {DIFF(CI, IE, f='gbp0')} ahead of EIRE at "
           f"{R(CI, f='gbp0')} vs {R(IE, f='gbp0')} TW")
    B.line(f"Netherlands ({R(NL)}) and EIRE ({R(IE)}) each held {SH(NL)} of TW revenue")
    B.line(f"Switzerland lost {ABSCHG(CH, dirw='down')} of revenue WoW, the largest fall among the {CNT(7, 'ALL')} countries")
    B.line(f"France's {WOW(FR, signed=True)} WoW gain edged out Channel Islands' {WOW(CI, signed=True, err=('drift', 343.5))}")
    B.line(f"Germany and EIRE combined for {SUM('Revenue_TW', [DE, IE], err=('grp', [DE, FR]))} TW, down from "
           f"{SUM('Revenue_LW', [DE, IE])} LW")
    B.line(f"The top {CNT(3)} markets held {GSH(top(3))} of TW revenue; the remaining {CNT(4)} took "
           f"{SUM('Revenue_TW', rest(3))} TW")
    B.line(f"Revenue per order: United Kingdom {AVG(UK)} TW, Switzerland {AVG(CH, err=('col', LWMAP))} TW")
    B.end()

    B.start("SC-4", tid, "hedged_mixed")
    B.line(f"Week {NM('21')} of {NM('2010', 'year')}: United Kingdom revenue was about "
           f"{R(UK, f='gbpk0', q='approx')} TW, more than {SH(UK, f='pct0', q='over', show=90)} of the total")
    B.line(f"UK orders rose to {O(UK)} TW from {O(UK, 'LW')} LW, "
           f"{DCHG(UK, 'Orders_TW', 'Orders_LW', verbs=('up', 'down'), err=('drift', 17.9))}")
    B.line(f"Channel Islands units jumped to {U(CI)} TW from {U(CI, 'LW')} LW")
    B.line(f"Netherlands contributed roughly {R(NL, f='gbpk1', q='approx')} TW with {U(NL, err=('row', FR))} units "
           f"across {O(NL)} orders")
    B.line(f"EIRE revenue {WOW(IE, verbs=('rose', 'fell'), err='dir')} WoW to {R(IE)} TW, although units edged up to "
           f"{U(IE)} from {U(IE, 'LW')}")
    B.line(f"Switzerland units dropped to {U(CH)} TW from {U(CH, 'LW')} LW, {DCHG(CH, verbs=('up', 'down'), err='dir')}")
    B.line(f"France and Germany together brought in {SUM('Revenue_TW', [FR, DE])} TW, a {GSH([FR, DE], err='round')} share")
    B.line(f"Across all {CNT(7, 'ALL')} countries, TW revenue averaged "
           f"{MEAN('Revenue_TW', 'ALL', err=('drift', 22904.38))} per country, and orders totalled {SUM('Orders_TW', 'ALL')}")
    B.end()


def notes_small_product_w47():
    tid = "small_product_2011-W47"
    P50 = "PAPER CHAIN KIT 50'S CHRISTMAS"
    RNL = "RABBIT NIGHT LIGHT"
    CL = "CHILLI LIGHTS"
    RC = "REGENCY CAKESTAND 3 TIER"
    PV = "PAPER CHAIN KIT VINTAGE CHRISTMAS"
    RSA = "ROTATING SILVER ANGELS T-LIGHT HLDR"

    B.start("SP47-1", tid, "terse_kpi")
    B.line(f"{E(P50)}: {R(P50)} TW, {WOW(P50, signed=True)} WoW, {YOY(P50, signed=True, err=('drift', -6.0))} YoY, "
           f"{SH(P50)} share")
    B.line(f"{RNL}: {R(RNL)} TW vs {R(RNL, 'LW')} LW, {WOW(RNL, signed=True)} WoW")
    B.line(f"{CL}: {R(CL)} TW, {WOW(CL, signed=True, err=('col', 'YoY_Pct'))} WoW, {YOY(CL, signed=True)} YoY")
    B.line(f"{E(RC)}: {R(RC)} TW vs {R(RC, 'LY', err=('col', 'Revenue_LW'))} LY, {YOY(RC, signed=True)} YoY; "
           f"{U(RC)} units")
    B.line(f"{PV}: {R(PV)} TW, {O(PV)} orders (LW {O(PV, 'LW', err=('drift', 57))})")
    B.line(f"{RSA}: {R(RSA)} TW, {WOW(RSA, signed=True, err=('row', RC))} WoW, rank {RK(RSA)}")
    B.line(f"Total: {SUM('Revenue_TW', 'ALL')} TW vs {SUM('Revenue_LW', 'ALL', err=('grp', top(5)))} LW, "
           f"{GCHG('ALL', signed=True)} WoW")
    B.end()

    pcks = [P50, PV]
    B.start("SP47-2", tid, "narrative")
    B.line(f"Christmas lines led {NM('2011-W47')}, with {E(P50)} top of the table at {R(P50)} TW, {SH(P50)} of revenue.")
    B.line(f"The {CNT(2, pcks)} PAPER CHAIN KIT lines combined for {SUM('Revenue_TW', pcks)} TW, down from "
           f"{SUM('Revenue_LW', pcks, err=('grp', [P50, RSA]))} LW.")
    B.line(f"{RNL} revenue {WOW(RNL, verbs=('rose', 'fell'), err='dir')} WoW to {R(RNL)} TW, with units down to "
           f"{U(RNL)} from {U(RNL, 'LW')}.")
    B.line(f"{CL} revenue {WOW(CL, verbs=('rose', 'fell'))} WoW to {R(CL)} TW and was {YOY(CL, verbs=('up', 'down'))} YoY.")
    B.line(f"{E(RC)} {WOW(RC, verbs=('grew', 'declined'))} WoW to {R(RC)} TW but was "
           f"{YOY(RC, verbs=('up', 'down'), err=('drift', -54.9))} YoY.")
    B.line(f"{RSA} revenue was {R(RSA, err=('row', PV))} TW, {YOY(RSA, verbs=('up', 'down'))} YoY, from {O(RSA)} orders.")
    B.line(f"The top {CNT(3)} products accounted for {GSH(top(3), err='round')} of TW revenue.")
    B.end()

    B.start("SP47-3", tid, "comparison")
    B.line(f"{E(P50)} led {RNL} by {DIFF(P50, RNL)} TW ({R(P50)} vs {R(RNL)})")
    B.line(f"{RNL} sold the most units at {U(RNL)}, "
           f"{RATIO(c(RNL, 'Units_TW'), c(CL, 'Units_TW'), f'{RNL} / {CL} units', err=('drift', 2.6))} {CL}' {U(CL)}")
    B.line(f"{CL} averaged {AVG(CL, err='round')} per order TW versus {AVG(CL, 'LW')} LW")
    B.line(f"{E(RC)} averaged {AVG(RC, err=('col', LWMAP))} per order TW on {O(RC)} orders")
    B.line(f"{PV} ({R(PV)}) outsold {RSA} ({R(RSA)}) by {DIFF(PV, RSA, err=('drift', 149.96))} TW")
    B.line(f"Share gap: {E(P50)} {SH(P50)} vs {RNL} {SH(RNL)}, a {SHDIFF(P50, RNL)} pp difference")
    B.line(f"The bottom {CNT(3)} products combined for {SUM('Revenue_TW', rest(3), err=('grp', [RC, PV]))} TW, "
           f"{GSH(rest(3))} of the total")
    B.end()

    B.start("SP47-4", tid, "hedged_mixed")
    B.line(f"Nearly {SH(P50, f='pct0', q='approx')} of TW revenue came from {E(P50)}, at around "
           f"{R(P50, f='gbpk1', q='approx')}")
    B.line(f"{RNL} orders slipped to {O(RNL)} TW from {O(RNL, 'LW')} LW, while revenue {WOW(RNL, verbs=('rose', 'fell'))} WoW")
    B.line(f"{CL} units {DCHG(CL, verbs=('rose', 'fell'), err=('drift', 164.5))} WoW to {U(CL)}")
    B.line(f"Year on year, {CL} revenue was up {YOY(CL, dirw='up')} at {R(CL)} TW against "
           f"{R(CL, 'LY', err=('row', RSA))} LY")
    B.line(f"{E(RC)} held rank {RK(RC, err=('row', PV))} with {R(RC, f='gbp0')} TW, down from {R(RC, 'LY', f='gbp0')} LY")
    B.line(f"{PV}: {WOW(PV, signed=True)} WoW, {YOY(PV, signed=True)} YoY, {U(PV)} units TW")
    B.line(f"{RSA} units climbed to {U(RSA)} from {U(RSA, 'LW')} LW, while revenue was "
           f"{WOW(RSA, verbs=('up', 'down'), err='dir')}")
    B.line(f"Across the {CNT(6, 'ALL')} products, TW orders totalled {SUM('Orders_TW', 'ALL')} against "
           f"{SUM('Orders_LW', 'ALL', err=('grpv', 424))} LW")
    B.end()


def notes_small_product_w07():
    tid = "small_product_2010-W07"
    FM = "SMALL FAIRY CAKE FRIDGE MAGNETS"
    DR = "DOOR MAT RED SPOT"
    MUG = "BLACK AND WHITE PAISLEY FLOWER MUG"
    WH = "WHITE HANGING HEART T-LIGHT HOLDER"
    JB = "JUMBO BAG RED WHITE SPOTTY"
    DS = "DOOR MAT SPOTTY HOME SWEET HOME"

    B.start("SP07-1", tid, "terse_kpi")
    B.line(f"{FM}: {R(FM)} TW, {SH(FM)} share, {U(FM, err=('drift', 11690))} units from {O(FM)} orders; nil LW")
    B.line(f"{DR}: {R(DR)} TW vs {R(DR, 'LW')} LW, {WOW(DR, signed=True, err=('row', DS))} WoW")
    B.line(f"{MUG}: {R(MUG)} TW, {U(MUG)} units, {O(MUG, err=('row', FM))} orders")
    B.line(f"{WH}: {R(WH, err=('col', 'Revenue_LW'))} TW, {WOW(WH, signed=True)} WoW, {O(WH)} orders (LW {O(WH, 'LW')})")
    B.line(f"{JB}: {R(JB)} TW, {WOW(JB, signed=True)} WoW, {SH(JB, err='round')} share")
    B.line(f"{DS}: {R(DS)} TW, {WOW(DS, signed=True)} WoW, rank {RK(DS)}")
    B.line(f"Table total: {SUM('Revenue_TW', 'ALL')} TW vs {SUM('Revenue_LW', 'ALL')} LW; "
           f"{SUM('Units_TW', 'ALL', err=('grpv', 34476))} units")
    B.end()

    dms = [DR, DS]
    B.start("SP07-2", tid, "narrative")
    B.line(f"In week {NM('7')} of {NM('2010', 'year')}, {FM} led with {R(FM)} TW from just "
           f"{O(FM, err=('row', MUG))} orders, {SH(FM)} of revenue.")
    B.line(f"{MUG} shipped {U(MUG, err=('drift', 19615))} units TW, the most in the table, for {R(MUG)}.")
    B.line(f"{DR} revenue {WOW(DR, verbs=('rose', 'fell'))} WoW to {R(DR)} TW on flat orders of {O(DR)}.")
    B.line(f"{DS} posted the biggest WoW gain, up {WOW(DS, dirw='up')} to {R(DS)} TW from "
           f"{R(DS, 'LW', err=('drift', 332.50))} LW.")
    B.line(f"{JB} {WOW(JB, verbs=('grew', 'fell'))} WoW to {R(JB)}, with orders up from {O(JB, 'LW')} to {O(JB)}.")
    B.line(f"{WH} edged {WOW(WH, verbs=('up', 'down'), err='dir')} WoW to {R(WH)} TW.")
    B.line(f"The {CNT(2, dms)} DOOR MAT lines together took {SUM('Revenue_TW', dms)} TW, {GSH(dms)} share, against "
           f"{SUM('Revenue_LW', dms, err=('grpv', 998.94))} LW.")
    B.end()

    B.start("SP07-3", tid, "comparison")
    B.line(f"{FM} averaged {AVG(FM)} per order TW, versus {AVG(DR, err=('col', LWMAP))} for {DR}")
    B.line(f"{MUG} units ({U(MUG)}) were {RATIO(c(MUG, 'Units_TW'), c(FM, 'Units_TW'), f'{MUG} / {FM} units', err='round')} "
           f"those of {FM} ({U(FM)})")
    B.line(f"The top {CNT(3)} products took {GSH(top(3))} of TW revenue; the other {CNT(3)} took "
           f"{SUM('Revenue_TW', rest(3), err=('grp', [WH, JB]))}")
    B.line(f"{WH} and {JB} were {DIFF(WH, JB)} apart TW ({R(WH)} vs {R(JB)})")
    B.line(f"{DS} revenue rose by {ABSCHG(DS, dirw='up', err=('drift', 1354.85))} WoW, on {U(DS)} units vs {U(DS, 'LW')}")
    B.line(f"{JB} sold {UPO(JB)} units per order TW (LW {UPO(JB, 'LW', err=('row', DS))})")
    B.line(f"Total TW revenue of {SUM('Revenue_TW', 'ALL')} was about "
           f"{RATIO(X_sum('Revenue_TW', 'ALL'), X_sum('Revenue_LW', 'ALL'), 'TOTAL TW / LW', q='approx')} "
           f"LW's {SUM('Revenue_LW', 'ALL')}")
    B.line(f"{DR} and {DS} ranked {RK(DR)} and {RK(DS)}")
    B.end()

    B.start("SP07-4", tid, "hedged_mixed")
    B.line(f"{FM} brought in about {R(FM, f='gbpk1', q='approx')} TW, over {SH(FM, f='pct0', q='over', show=40)} of revenue")
    B.line(f"{MUG} sold more than {U(MUG, q='over', err=('drift', 20000))} units TW for roughly "
           f"{R(MUG, f='gbp0', q='approx')}")
    B.line(f"{DR}: {O(DR)} orders both TW and LW, revenue {WOW(DR, signed=True)} WoW")
    B.line(f"{WH} took {SH(WH, err=('row', MUG))} share with {U(WH)} units, up from {U(WH, 'LW')} LW")
    B.line(f"{JB} revenue was {R(JB)} TW, up from {R(JB, 'LW', err=('drift', 995.69))} LW")
    B.line(f"{DS} orders rose to {O(DS)} TW from {O(DS, 'LW')} LW, "
           f"{DCHG(DS, 'Orders_TW', 'Orders_LW', signed=True, err='round')}")
    B.line(f"The bottom {CNT(2)} products, {JB} and {DS}, made {SUM('Revenue_TW', [JB, DS])} TW combined, "
           f"{GSH([JB, DS])} of the table")
    B.line(f"Mean TW revenue across the {CNT(6, 'ALL')} products was {MEAN('Revenue_TW', 'ALL')}")
    B.end()


def notes_medium_product_w10():
    tid = "medium_product_2011-W10"
    RC = "REGENCY CAKESTAND 3 TIER"
    CL = "CHILLI LIGHTS"
    HW = "HEART OF WICKER SMALL"
    JBR = "JUMBO BAG RED RETROSPOT"
    PB = "PARTY BUNTING"
    WH = "WHITE HANGING HEART T-LIGHT HOLDER"
    DH = "DOORMAT HEARTS"
    SOM = "SOMBRERO"

    B.start("MP10-1", tid, "terse_kpi")
    B.line(f"{E(RC)}: {R(RC)} TW, {WOW(RC, signed=True, err='dir')} WoW, {SH(RC)} share")
    B.line(f"{CL}: {R(CL)} TW vs {R(CL, 'LW', err=('row', HW))} LW, {WOW(CL, signed=True)} WoW")
    B.line(f"{HW}: {R(HW)} TW, {WOW(HW, signed=True)} WoW, {U(HW, err=('col', 'Units_LW'))} units")
    B.line(f"{PB}: {R(PB)} TW, {WOW(PB, signed=True)} WoW, {YOY(PB, signed=True, err=('col', 'WoW_Pct'))} YoY")
    B.line(f"{WH}: {R(WH)} TW vs {R(WH, 'LY', err=('drift', 2356.84))} LY, {YOY(WH, signed=True)} YoY")
    B.line(f"{DH}: {R(DH)} TW, {YOY(DH, signed=True)} YoY from {R(DH, 'LY')} LY")
    B.line(f"{SOM}: {R(SOM)} TW, {U(SOM)} units, rank {RK(SOM)}")
    B.line(f"Table total: {SUM('Revenue_TW', 'ALL')} TW, {GCHG('ALL', signed=True, err='dir')} WoW across "
           f"{CNT(39, 'ALL')} products")
    B.end()

    jumbo7 = fam("JUMBO BAG") + fam("JUMBO  BAG")
    baroque = "JUMBO  BAG BAROQUE BLACK WHITE"
    dm5 = fam("DOORMAT")
    bg3 = fam("BABY GIFT SET")
    bunt3 = [PB, "VINTAGE UNION JACK BUNTING", "WOODEN UNION JACK BUNTING"]
    felt = ["FELTCRAFT CUSHION BUTTERFLY", "FELTCRAFT CUSHION RABBIT"]
    jj, jp = "JAM MAKING SET WITH JARS", "JAM MAKING SET PRINTED"
    ct = "SET OF 3 CAKE TINS PANTRY DESIGN"
    B.start("MP10-2", tid, "family_narrative")
    B.line(f"Across {NM('2011-W10')}, the {CNT(7, jumbo7)} JUMBO BAG lines combined for {SUM('Revenue_TW', jumbo7)} TW, "
           f"up {GCHG(jumbo7, dirw='up')} on "
           f"{SUM('Revenue_LW', jumbo7, err=('grp', [e for e in jumbo7 if e != baroque]))} LW.")
    B.line(f"{JBR} was the largest of them at {R(JBR)} TW, {WOW(JBR, verbs=('up', 'down'))} WoW.")
    B.line(f"The {CNT(5, dm5)} DOORMAT lines took {SUM('Revenue_TW', dm5)} TW, {GSH(dm5, err='round')} of the table, "
           f"up from {SUM('Revenue_LW', dm5)} LW.")
    B.line(f"The {CNT(3, bg3)} BABY GIFT SET lines, all new versus LW, totalled {SUM('Revenue_TW', bg3)} TW from "
           f"{SUM('Orders_TW', bg3, err=('grpv', 16))} orders.")
    B.line(f"The {CNT(3, bunt3, word=True)} bunting lines ({PB}, VINTAGE UNION JACK BUNTING and WOODEN UNION JACK BUNTING) "
           f"{GCHG(bunt3, verbs=('rose', 'fell'))} WoW to {SUM('Revenue_TW', bunt3)} TW.")
    B.line(f"FELTCRAFT CUSHION BUTTERFLY and FELTCRAFT CUSHION RABBIT together made {SUM('Revenue_TW', felt)} TW vs "
           f"{SUM('Revenue_LY', felt, err=('grpv', 353.46))} LY.")
    B.line(f"{jj} {WOW(jj, verbs=('rose', 'fell'))} WoW to {R(jj)}, while {jp} {WOW(jp, verbs=('rose', 'fell'))} to "
           f"{R(jp, err=('col', 'Revenue_LW'))}.")
    B.line(f"{E(ct)} held steady at {R(ct)} TW with {O(ct)} orders.")
    B.end()

    cs = "CHARLOTTE BAG SUKI DESIGN"
    vu = "VINTAGE UNION JACK BUNTING"
    mp = "MINI PAINT SET VINTAGE"
    sb, dg, cp = "SPACEBOY BABY GIFT SET", "DOLLY GIRL BABY GIFT SET", "CIRCUS PARADE BABY GIFT SET"
    vr = "VINTAGE RED KITCHEN CABINET"
    top4_share = X_shr(X_sum("Revenue_TW", top(4)), X_sum("Revenue_TW", "ALL"))
    B.start("MP10-3", tid, "comparison")
    B.line(f"The top {CNT(5)} products took {GSH(top(5), err=('grpx', top4_share))} of TW revenue, led by {E(RC)} at {R(RC)}")
    B.line(f"{E(RC)} out-earned {CL} by {DIFF(RC, CL, err=('drift', 1109.70))} TW")
    B.line(f"{cs} posted the steepest WoW rise at {WOW(cs, signed=True)}, reaching {R(cs)}")
    B.line(f"{vu} {WOW(vu, verbs=('rose', 'dropped'))} WoW to {R(vu, err=('col', 'Revenue_LW'))}, with orders down from "
           f"{O(vu, 'LW')} to {O(vu)}")
    B.line(f"{mp} sold {U(mp)} units TW, {RATIO(c(mp, 'Units_TW'), c(mp, 'Units_LW'), f'{mp} units TW / LW', err='round')} "
           f"its LW volume of {U(mp, 'LW')}")
    B.line(f"{sb} averaged {AVG(sb)} per order TW, below {dg}'s {AVG(dg, err=('row', cp))}")
    B.line(f"Outside the top {CNT(10)}, the remaining {CNT(29)} products averaged {MEAN('Revenue_TW', rest(10))} TW each")
    B.line(f"{vr} made {R(vr)} TW from just {U(vr)} units, or {AVG(vr)} per order")
    B.end()

    dfc = "DOORMAT FAIRY CAKE"
    jpv = "JUMBO BAG PINK VINTAGE PAISLEY"
    sp = "SMALL POPCORN HOLDER"
    hb, po = "HOT BATHS METAL SIGN", "PLEASE ONE PERSON METAL SIGN"
    ns = "NATURAL SLATE HEART CHALKBOARD"
    B.start("MP10-4", tid, "hedged_mixed")
    B.line(f"{E(RC)} revenue {WOW(RC, verbs=('rose', 'fell'))} WoW to about {R(RC, f='gbpk1', q='approx')}")
    B.line(f"{dfc} went from {R(dfc, 'LW')} LW to {R(dfc)} TW, up more than "
           f"{WOW(dfc, f='pct0', q='over', show=600, dirw='up')}")
    B.line(f"{jpv} was up {YOY(jpv, dirw='up', err=('drift', 543.2))} YoY at {R(jpv)} TW")
    B.line(f"{sp} units rose to {U(sp)} from {U(sp, 'LW')} LW; revenue {WOW(sp, verbs=('up', 'down'))}")
    B.line(f"{hb} and {po} took {SUM('Revenue_TW', [hb, po], err=('grpv', 1272.47))} TW combined, around "
           f"{GSH([hb, po], f='pct0', q='approx')} of the table")
    B.line(f"{ns} was below LY at {R(ns)} TW vs {R(ns, 'LY', err=('row', po))} LY, {YOY(ns, signed=True)} YoY")
    B.line(f"{baroque} (rank {RK(baroque)}) {WOW(baroque, verbs=('grew', 'fell'), err='round')} WoW to {R(baroque)}")
    B.line(f"{SOM} sold {U(SOM)} units TW with nothing LW, and was {YOY(SOM, verbs=('up', 'down'), err=('row', DH))} "
           f"on LY's {R(SOM, 'LY')}")
    B.end()


def notes_medium_product_w05():
    tid = "medium_product_2010-W05"
    WH = "WHITE HANGING HEART T-LIGHT HOLDER"
    CW = "COOK WITH WINE METAL SIGN"
    DRS = "DOOR MAT RED SPOT"
    DBF = "DOOR MAT BLACK FLOCK"
    RSC = "RETRO SPOT CAKE STAND"
    DWP = "DOOR MAT WELCOME PUPPIES"
    D3 = "DOOR MAT 3 SMILEY CATS"
    RH = "RED HANGING HEART T-LIGHT HOLDER"
    DSH = "DOOR MAT SPOTTY HOME SWEET HOME"
    JRW = "JUMBO BAG RED WHITE SPOTTY"
    DHE = "DOOR MAT HEARTS"
    KF = "KASHMIR FOLKART TUMBLERS"
    HSM = "HOME SWEET HOME METAL SIGN"
    BRF = "BLACK RECORD COVER FRAME"
    WCL = "WHITE CHERRY LIGHTS"
    PBF = "PINK BLUE FELT CRAFT TRINKET BOX"
    DUF = "DOOR MAT UNION FLAG"
    YCM = "YOU'RE CONFUSING ME METAL SIGN"
    CHC = "CREAM HEART CARD HOLDER"
    RSH = "RED SPOT HEART HOT WATER BOTTLE"
    POP = "PLEASE ONE PERSON METAL SIGN"
    SOM = "SOMBRERO"
    JPW = "JUMBO BAG PINK WITH WHITE SPOTS"
    LBB = "LOVE BUILDING BLOCK WORD"

    B.start("MP05-1", tid, "terse_kpi")
    B.line(f"{WH}: {R(WH)} TW vs {R(WH, 'LW')} LW, {WOW(WH, signed=True)} WoW, {SH(WH, err='round')} share")
    B.line(f"{CW}: {R(CW)} TW, {WOW(CW, signed=True)} WoW, {U(CW, err=('col', 'Units_LW'))} units")
    B.line(f"{E(D3)}: {R(D3)} TW, {WOW(D3, signed=True, err=('drift', 242.5))} WoW, {O(D3)} orders vs {O(D3, 'LW')} LW")
    B.line(f"{RH}: {R(RH)} TW, {WOW(RH, signed=True)} WoW")
    B.line(f"{KF}: {R(KF)} TW, {U(KF)} units, rank {RK(KF, err=('row', HSM))}")
    B.line(f"{YCM}: {R(YCM)} TW vs {R(YCM, 'LW', err=('drift', 36.87))} LW, {WOW(YCM, signed=True)} WoW")
    B.line(f"{LBB}: {R(LBB)} TW, {O(LBB)} orders, {SH(LBB)} share")
    B.line(f"Total: {SUM('Revenue_TW', 'ALL')} TW vs {SUM('Revenue_LW', 'ALL')} LW, "
           f"{GCHG('ALL', signed=True, err='dir')} WoW")
    B.end()

    dm7 = fam("DOOR MAT")
    ms5 = fam("METAL SIGN")
    hh2 = [WH, RH]
    jb2 = [JRW, JPW]
    B.start("MP05-2", tid, "family_narrative")
    B.line(f"In {NM('2010-W05')}, the {CNT(7, dm7)} DOOR MAT lines together took "
           f"{SUM('Revenue_TW', dm7, err=('grp', [e for e in dm7 if e != DUF]))} TW, {GSH(dm7)} of revenue, against "
           f"{SUM('Revenue_LW', dm7)} LW.")
    B.line(f"{CNT(3, [DRS, DSH, DHE])} of them ({DRS}, {DSH} and {DHE}) had no LW sales; {DRS} led at {R(DRS)} TW "
           f"and {DHE} made {R(DHE)}.")
    B.line(f"The {CNT(5, ms5)} METAL SIGN lines combined for {SUM('Revenue_TW', ms5)} TW, up from "
           f"{SUM('Revenue_LW', ms5, err=('grpv', 880.82))} LW, on {SUM('Units_TW', ms5)} units.")
    B.line(f"{POP} {WOW(POP, verbs=('jumped', 'fell'), err=('row', CW))} WoW to {R(POP)} TW.")
    B.line(f"The {CNT(2, hh2)} HANGING HEART T-LIGHT HOLDER lines fell to {SUM('Revenue_TW', hh2)} TW from "
           f"{SUM('Revenue_LW', hh2)} LW, {GCHG(hh2, verbs=('up', 'down'))}.")
    B.line(f"{DUF} {WOW(DUF, verbs=('rose', 'fell'))} WoW to {R(DUF)}, with orders down to {O(DUF)} from "
           f"{O(DUF, 'LW', err=('drift', 30))}.")
    B.line(f"{JRW} and {JPW} totalled {SUM('Revenue_TW', jb2)} TW, {GCHG(jb2, verbs=('up', 'down'), err='round')} WoW.")
    B.end()

    B.start("MP05-3", tid, "comparison")
    B.line(f"{WH} ({R(WH)}) stayed ahead of {CW} ({R(CW)}) by {DIFF(WH, CW, err=('drift', 530.75))} TW")
    B.line(f"{CW} had the steepest WoW rise at {WOW(CW, signed=True)}, from {R(CW, 'LW')} LW")
    B.line(f"{DRS} and {DBF} ranked {RK(DRS)} and {RK(DBF, err=('row', RSC))}, at {R(DRS)} and {R(DBF)} TW")
    B.line(f"{RSC} averaged {AVG(RSC)} per order TW vs {AVG(RSC, 'LW', err='round')} LW")
    B.line(f"{KF} sold {U(KF)} units, {RATIO(c(KF, 'Units_TW'), c(KF, 'Units_LW'), f'{KF} units TW / LW', err=('drift', 5.9))} "
           f"LW's {U(KF, 'LW')}")
    B.line(f"{PBF} rose {WOW(PBF, dirw='up')} WoW to {R(PBF)}, while {WCL} fell {WOW(WCL, dirw='down')} to "
           f"{R(WCL, err=('col', 'Revenue_LW'))}")
    B.line(f"The top {CNT(10)} products took {GSH(top(10))} of TW revenue; the other {CNT(16)} averaged "
           f"{MEAN('Revenue_TW', rest(10), err=('grp', rest(10)[:-1]))} each")
    B.line(f"{SOM} revenue of {R(SOM)} TW came from only {O(SOM)} orders, {AVG(SOM)} each")
    B.end()

    B.start("MP05-4", tid, "hedged_mixed")
    B.line(f"{WH} still ranked {RK(WH, f='ord')} at around {R(WH, f='gbpk1', q='approx')}, down "
           f"{WOW(WH, dirw='down', err=('drift', -43.6))} WoW")
    B.line(f"{CW} sold more than {U(CW, q='over', err=('drift', 900))} units TW versus {U(CW, 'LW')} LW")
    B.line(f"{DWP} revenue was up {WOW(DWP, dirw='up')} at {R(DWP)}, on {O(DWP, err=('col', 'Orders_LW'))} orders")
    B.line(f"{HSM} made {R(HSM)} TW from {O(HSM)} orders (LW {O(HSM, 'LW')})")
    B.line(f"{BRF} took about {SH(BRF, f='pct0', q='approx')} share with {R(BRF)}")
    B.line(f"{CHC} ({R(CHC)}) and {RSH} ({R(RSH, err=('col', 'Revenue_LW'))}) were almost level TW, "
           f"{DIFF(CHC, RSH)} apart")
    B.line(f"{SOM} units rose to {U(SOM)} from {U(SOM, 'LW')} LW, {DCHG(SOM, verbs=('up', 'down'))}")
    B.line(f"Units across all {CNT(26, 'ALL')} products totalled {SUM('Units_TW', 'ALL')} TW vs "
           f"{SUM('Units_LW', 'ALL', err=('grpv', 4372))} LW")
    B.end()


def notes_large_product_w11():
    tid = "large_product_2011-W11"
    RC = "REGENCY CAKESTAND 3 TIER"
    PB = "PARTY BUNTING"
    JBR = "JUMBO BAG RED RETROSPOT"
    WH = "WHITE HANGING HEART T-LIGHT HOLDER"
    SK = "PACK OF 12 SKULL TISSUES"
    LT = "PACK OF 12 LONDON TISSUES"
    PEN = "12 PENCILS SMALL TUBE RED RETROSPOT"
    HK = "3 HOOK PHOTO SHELF ANTIQUE WHITE"
    JSV = "JUMBO SHOPPER VINTAGE RED PAISLEY"

    B.start("LP11-1", tid, "terse_kpi")
    B.line(f"{E(RC)}: {R(RC)} TW, {WOW(RC, signed=True)} WoW, {YOY(RC, signed=True, err=('col', 'WoW_Pct'))} YoY, "
           f"{SH(RC)} share")
    B.line(f"{PB}: {R(PB)} TW vs {R(PB, 'LY', err=('drift', 1041.34))} LY, {YOY(PB, signed=True)} YoY")
    B.line(f"{JBR}: {R(JBR)} TW, {U(JBR)} units, {WOW(JBR, signed=True)} WoW")
    B.line(f"{WH}: {R(WH)} TW vs {R(WH, 'LY')} LY, {YOY(WH, signed=True, err='dir')} YoY")
    B.line(f"{E(SK)}: {R(SK)} TW from {R(SK, 'LW', err=('row', LT))} LW, {WOW(SK, signed=True)} WoW")
    B.line(f"{E(PEN)}: {R(PEN)} TW, {U(PEN, err=('drift', 980))} units, {O(PEN)} orders")
    B.line(f"{E(HK)}: {R(HK)} TW vs {R(HK, 'LY')} LY, {YOY(HK, signed=True)} YoY, rank {RK(HK, err=('row', JSV))}")
    B.line(f"Total: {SUM('Revenue_TW', 'ALL')} TW vs {SUM('Revenue_LW', 'ALL')} LW and "
           f"{SUM('Revenue_LY', 'ALL', err=('grpv', 23107.90))} LY")
    B.end()

    jb9 = [e for e in B.T.index if e.startswith("JUMBO BAG") or e.startswith("JUMBO  BAG")]
    toys = "JUMBO BAG TOYS"
    sc3 = fam("KITCHEN SCALES")
    bm3 = fam("BAKING MOULD")
    bg3 = fam("BABY GIFT SET")
    ms4 = fam("METAL SIGN")
    tc2 = fam("REGENCY TEACUP AND SAUCER")
    B.start("LP11-2", tid, "family_narrative")
    B.line(f"The {CNT(9, jb9)} JUMBO BAG lines, excluding the shopper and storage bag, made {SUM('Revenue_TW', jb9)} TW, "
           f"{GSH(jb9, err='round')} of revenue.")
    B.line(f"{toys} posted the group's biggest WoW rise at {WOW(toys, signed=True)}, to {R(toys)} from "
           f"{R(toys, 'LW', err=('col', 'Revenue_LY'))} LW.")
    B.line(f"The {CNT(3, sc3)} KITCHEN SCALES colours (IVORY, RED and BLACK) totalled {SUM('Revenue_TW', sc3)} TW vs "
           f"{SUM('Revenue_LW', sc3, err=('grp', ['IVORY KITCHEN SCALES', 'RED KITCHEN SCALES']))} LW.")
    B.line(f"The {CNT(3, bm3, word=True)} BAKING MOULD lines made {SUM('Revenue_TW', bm3)} TW from "
           f"{SUM('Orders_TW', bm3)} orders, up from {SUM('Revenue_LW', bm3, err=('grpv', 119.98))} LW.")
    B.line(f"The {CNT(3, bg3)} BABY GIFT SET lines {GCHG(bg3, verbs=('rose', 'fell'))} WoW to {SUM('Revenue_TW', bg3)}.")
    B.line(f"The {CNT(4, ms4)} METAL SIGN lines were almost flat at {SUM('Revenue_TW', ms4)} TW "
           f"({GCHG(ms4, signed=True, err='dir')} WoW) but {GCHG(ms4, 'Revenue_TW', 'Revenue_LY', verbs=('up', 'down'))} YoY.")
    B.line(f"ROSES REGENCY TEACUP AND SAUCER and GREEN REGENCY TEACUP AND SAUCER combined for {SUM('Revenue_TW', tc2)} TW, "
           f"{GCHG(tc2, verbs=('up', 'down'))} WoW.")
    B.line(f"{E(SK)} and {E(LT)} sold {SUM('Units_TW', [SK, LT])} units between them TW.")
    B.end()

    CL = "CHILLI LIGHTS"
    sbg, dolly = "SPACEBOY BABY GIFT SET", "DOLLY GIRL BABY GIFT SET"
    ct3 = "SET OF 3 CAKE TINS PANTRY DESIGN"
    sp6 = "SET OF 6 SPICE TINS PANTRY DESIGN"
    cc72 = "PACK OF 72 RETROSPOT CAKE CASES"
    bt = "BAKING MOULD TOFFEE CUP CHOCOLATE"
    r3 = rest(3)
    B.start("LP11-3", tid, "comparison")
    B.line(f"The top {CNT(3)} products ({E(RC)}, {PB} and {JBR}) held {GSH(top(3))} of TW revenue")
    B.line(f"The remaining {CNT(88)} products took {SUM('Revenue_TW', r3)} TW, an average of "
           f"{MEAN('Revenue_TW', r3, err=('grpv', ev_float(X_sum('Revenue_TW', r3), B.T) / 91))} each")
    B.line(f"{E(RC)} led {PB} by {DIFF(RC, PB)} TW")
    B.line(f"{CL} {WOW(CL, verbs=('rose', 'fell'))} WoW to {R(CL)}, just above {sbg} at {R(sbg, err=('row', dolly))}")
    B.line(f"{E(ct3)} averaged {AVG(ct3)} per order versus {AVG(sp6, err=('col', LWMAP))} for {E(sp6)}")
    B.line(f"{E(cc72)} sold {U(cc72)} units TW across {O(cc72)} orders, {UPO(cc72)} units per order")
    B.line(f"{bt} had the highest WoW growth in the table at {WOW(bt, signed=True)}")
    B.line(f"{WH} LY revenue of {R(WH, 'LY')} was "
           f"{RATIO(c(WH, 'Revenue_LY'), c(WH, 'Revenue_TW'), f'{WH} LY / TW', err='round')} its TW level of {R(WH)}")
    B.end()

    jpp = "JUMBO BAG PINK POLKADOT"
    ep = "EDWARDIAN PARASOL NATURAL"
    wd2 = "WOOD 2 DRAWER CABINET WHITE FINISH"
    sl10 = "SET 10 LIGHTS NIGHT OWL"
    gt = "GIN + TONIC DIET METAL SIGN"
    lbrr = "LUNCH BAG RED RETROSPOT"
    tea11 = "RETROSPOT TEA SET CERAMIC 11 PC"
    nap20 = "SET/20 RED RETROSPOT PAPER NAPKINS"
    B.start("LP11-4", tid, "hedged_mixed")
    B.line(f"Rank {RK(RC)}: {E(RC)} at {R(RC)} TW, up {WOW(RC, dirw='up', err=('drift', 34.0))} on LW's {R(RC, 'LW')}")
    B.line(f"{PB} units climbed to {U(PB)} TW from {U(PB, 'LW')} LW; revenue {WOW(PB, verbs=('up', 'down'))}")
    B.line(f"{jpp}: roughly {R(jpp, f='gbpk1', q='approx')} TW, up over {WOW(jpp, f='pct0', q='over', show=170, dirw='up')} WoW")
    B.line(f"{ep} was {YOY(ep, verbs=('up', 'down'), err=('col', 'WoW_Pct'))} on LY at {R(ep)} TW, from {O(ep)} orders")
    B.line(f"{E(wd2)} made {R(wd2)} TW, {WOW(wd2, signed=True, err='round')} WoW, on {U(wd2)} units")
    B.line(f"{E(sl10)} revenue rose {WOW(sl10, dirw='up')} WoW and {YOY(sl10, dirw='up', err=('drift', 1168.1))} YoY "
           f"to {R(sl10)}")
    B.line(f"{gt} fell {YOY(gt, dirw='down')} YoY to {R(gt, err=('row', lbrr))} TW despite a {WOW(gt, dirw='up')} WoW gain")
    B.line(f"{E(tea11)} and {E(nap20)} took {SUM('Revenue_TW', [tea11, nap20])} TW combined, about "
           f"{GSH([tea11, nap20], q='approx')} of the table")
    B.end()


def notes_large_product_w16():
    tid = "large_product_2010-W16"
    RC = "REGENCY CAKESTAND 3 TIER"
    PB = "PARTY BUNTING"
    WH = "WHITE HANGING HEART T-LIGHT HOLDER"
    PLQ = "PINK AND LILAC QUILTED THROW"
    ERQ = "ENGLISH ROSE DESIGN QUILTED THROW"
    GBS = "GIANT BLACK SUNGLASSES"
    DUF = "DOOR MAT UNION FLAG"
    CMB = "CLASSIC METAL BIRDCAGE PLANT HOLDER"
    RRJ = "RED RETROSPOT JUMBO BAG"
    JBS = "JUMBO BAG STRAWBERRY"
    PMD = "PINK 3 PIECE MINI DOTS CUTLERY SET"
    GMD = "GREEN 3 PIECE MINI DOTS CUTLERY SET"
    RMD = "RED 3 PIECE MINI DOTS CUTLERY SET"
    BMD = "BLUE 3 PIECE MINI DOTS CUTLERY SET"
    POP = "PLEASE ONE PERSON METAL SIGN"
    TTC = "TEA TIME CAKE STAND IN GIFT BOX"
    TF60 = "60 TEATIME FAIRY CAKE CASES"

    B.start("LP16-1", tid, "terse_kpi")
    B.line(f"{E(RC)}: {R(RC)} TW, {WOW(RC, signed=True)} WoW, {SH(RC, err='round')} share, {O(RC)} orders")
    B.line(f"{PLQ}: {R(PLQ)} TW vs {R(PLQ, 'LW')} LW, {WOW(PLQ, signed=True, err=('drift', 3783.7))} WoW")
    B.line(f"{GBS}: {R(GBS)} TW from {O(GBS, err=('col', 'Orders_LW'))} orders, {U(GBS)} units, {WOW(GBS, signed=True)} WoW")
    B.line(f"{CMB}: {R(CMB)} TW, new versus nil LW, {U(CMB)} units")
    B.line(f"{E(PMD)}: {R(PMD)} TW, {WOW(PMD, signed=True, err=('drift', 962.8))} WoW")
    B.line(f"{POP}: {R(POP)} TW vs {R(POP, 'LW', err=('row', TTC))} LW, {WOW(POP, signed=True)} WoW")
    B.line(f"{E(TF60)}: {R(TF60)} TW, {U(TF60)} units vs {U(TF60, 'LW', err=('drift', 1049))} LW")
    B.line(f"Total across {CNT(115, 'ALL')} products: {SUM('Revenue_TW', 'ALL')} TW, {GCHG('ALL', signed=True)} WoW")
    B.end()

    cut4 = fam("MINI DOTS CUTLERY SET")
    qt2 = fam("QUILTED THROW")
    dm6 = fam("DOOR MAT")
    gs3 = fam("GARDEN SET")
    lb3 = fam("LUNCH BAG")
    tg2 = fam("ANTIQUE SILVER TEA GLASS")
    mug5 = [e for e in B.T.index if e.endswith(" MUG")]
    glam = "GLAMOROUS  MUG"
    B.start("LP16-2", tid, "family_narrative")
    B.line(f"In {NM('2010-W16')}, the {CNT(4, cut4)} MINI DOTS CUTLERY SET colours made {SUM('Revenue_TW', cut4)} TW, "
           f"up from {SUM('Revenue_LW', cut4, err=('grp', [e for e in cut4 if e != BMD]))} LW.")
    B.line(f"{E(PMD)} was the largest at {R(PMD)}, ahead of {E(GMD)} at {R(GMD, err=('row', RMD))}.")
    B.line(f"The {CNT(2, qt2)} QUILTED THROW lines totalled {SUM('Revenue_TW', qt2)} TW, "
           f"{GSH(qt2, err=('drift', 6.4))} share, against {SUM('Revenue_LW', qt2)} LW.")
    B.line(f"The {CNT(6, dm6)} DOOR MAT lines {GCHG(dm6, verbs=('rose', 'fell'), err='dir')} WoW to {SUM('Revenue_TW', dm6)} TW.")
    B.line(f"The {CNT(3, gs3)} wooden garden sets (SKITTLES, CROQUET and ROUNDERS) combined for {SUM('Revenue_TW', gs3)} TW, "
           f"down from {SUM('Revenue_LW', gs3)} LW.")
    B.line(f"The {CNT(3, lb3)} LUNCH BAG lines fell to {SUM('Revenue_TW', lb3)} TW from {SUM('Revenue_LW', lb3)} LW, "
           f"on {SUM('Orders_TW', lb3)} orders.")
    B.line(f"The {CNT(2, tg2, word=True)} ANTIQUE SILVER TEA GLASS lines {GCHG(tg2, verbs=('rose', 'fell'))} WoW to "
           f"{SUM('Revenue_TW', tg2)}.")
    B.line(f"The {CNT(5, mug5)} MUG lines sold {SUM('Units_TW', mug5, err=('grp', [e for e in mug5 if e != glam]))} units "
           f"TW for {SUM('Revenue_TW', mug5)} revenue.")
    B.end()

    top9_share = X_shr(X_sum("Revenue_TW", top(9)), X_sum("Revenue_TW", "ALL"))
    wr, wc = "WOODEN ROUNDERS GARDEN SET", "WOODEN CROQUET GARDEN SET"
    B.start("LP16-3", tid, "comparison")
    B.line(f"{E(RC)} earned {RATIO(c(RC, 'Revenue_TW'), c(PB, 'Revenue_TW'), f'{RC} / {PB}', err=('drift', 1.6))} "
           f"{PB}'s {R(PB)} TW")
    B.line(f"The top {CNT(5)} products took {GSH(top(5))} of TW revenue and the top {CNT(10)} "
           f"{GSH(top(10), err=('grpx', top9_share))}")
    B.line(f"{GBS} generated {AVG(GBS)} from its {O(GBS)} order TW")
    B.line(f"{WH} {WOW(WH, verbs=('rose', 'fell'), err='dir')} WoW to {R(WH)} but kept {RK(WH, f='ord')} place")
    B.line(f"{RRJ} ({R(RRJ)}) stayed {DIFF(RRJ, JBS)} ahead of {JBS} ({R(JBS)})")
    B.line(f"LUNCH BAG  BLACK SKULL. and {POP} both fell {WOW(POP, dirw='down')} WoW")
    B.line(f"{wr} averaged {AVG(wr)} per order TW, versus {AVG(wc, err=('col', LWMAP))} for {wc}")
    B.line(f"Below the top {CNT(10)}, the other {CNT(105)} products averaged {MEAN('Revenue_TW', rest(10))} TW")
    B.end()

    cc72 = "PACK OF 72 RETRO SPOT CAKE CASES"
    wpf = "WOODEN PICTURE FRAME WHITE FINISH"
    psd = "PIZZA SLICE DISH"
    doil = "SET OF 36 PAISLEY FLOWER DOILIES"
    lovebird = "CAKE STAND LOVEBIRD 2 TIER WHITE"
    B.start("LP16-4", tid, "hedged_mixed")
    B.line(f"{PB} revenue {WOW(PB, verbs=('rose', 'fell'))} WoW to about {R(PB, f='gbpk1', q='approx')}")
    B.line(f"{ERQ} sold {U(ERQ, err=('drift', 35))} units across {O(ERQ)} orders for {R(ERQ)}, with nothing LW")
    B.line(f"{DUF} was the top door mat at {R(DUF, err=('col', 'Revenue_LW'))} TW, {WOW(DUF, verbs=('up', 'down'))} WoW, "
           f"rank {RK(DUF)}")
    B.line(f"{E(cc72)} units fell to {U(cc72)} from {U(cc72, 'LW')} LW, {DCHG(cc72, verbs=('up', 'down'), err='round')}")
    B.line(f"{wpf} had more than {O(wpf, q='over', err=('drift', 50))} orders TW, though revenue slipped "
           f"{WOW(wpf, dirw='down')} WoW")
    B.line(f"{psd} jumped to {R(psd)} TW from {R(psd, 'LW', err=('drift', 150.00))} LW")
    B.line(f"{E(doil)} {WOW(doil, verbs=('rose', 'fell'))} WoW to {R(doil)} on {U(doil)} units")
    B.line(f"{E(lovebird)} averaged roughly {AVG(lovebird, f='gbp0', q='approx')} per order across its "
           f"{O(lovebird)} orders")
    B.end()


# ----------------------------------------------------------------------------------------------
# independent verification (csv + Decimal, no pandas)
# ----------------------------------------------------------------------------------------------
def load_dec(tid):
    with open(TABLE_DIR / f"{tid}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["Entity"]: r for r in rows}, [r["Entity"] for r in rows]


def ev_dec(x, tab, order):
    op = x["op"]
    if op == "cell":
        s = tab[x["ent"]][x["col"]]
        assert s != "", ("blank", x)
        return Decimal(s)
    if op in ("sum", "mean"):
        ents = order if x["ents"] == "ALL" else x["ents"]
        vals = [Decimal(tab[e][x["col"]]) for e in ents]
        s = sum(vals, Decimal(0))
        return s if op == "sum" else s / Decimal(len(vals))
    a, b = ev_dec(x["a"], tab, order), ev_dec(x["b"], tab, order)
    return {"sub": a - b, "div": a / b, "chg": Decimal(100) * (a / b - 1), "shr": Decimal(100) * a / b}[op]


def parse_value_text(vt):
    s = vt
    explicit = None
    if s[0] in "+-":
        explicit, s = s[0], s[1:]
    s = s.replace("£", "")
    m = re.fullmatch(r"([\d,]+)(?:\.(\d+))?(k|x|%|st|nd|rd|th)?", s)
    assert m, ("unparseable", vt)
    intpart, dec, suf = m.group(1), m.group(2) or "", m.group(3) or ""
    num = Decimal(intpart.replace(",", "") + ("." + dec if dec else ""))
    scale = Decimal(1000) if suf == "k" else Decimal(1)
    unit = scale * Decimal(1).scaleb(-len(dec))
    val = num * scale
    if explicit == "-":
        val = -val
    return val, unit, explicit


def half_up(v, unit):
    return (v / unit).quantize(Decimal(1), rounding=ROUND_HALF_UP) * unit


def verify(rows):
    tabs = {t: load_dec(t) for t in TABLES}
    problems = []
    for r in rows:
        if r["label"] in ("N", "CNT"):
            if r["is_word"]:
                assert r["value_text"].lower() in WORD_VAL
            else:
                assert r["value_text"].isdigit()
            if r["label"] == "CNT" and r["count"] != "":
                v = WORD_VAL.get(r["value_text"].lower(), None) if r["is_word"] else int(r["value_text"])
                assert v == int(r["count"]), r
            continue
        tab, order = tabs[r["table_id"]]
        expr = json.loads(r["expr"])
        true = ev_dec(expr, tab, order)
        assert abs(true - Dd(float(r["true_value"]))) < Decimal("1e-6"), ("pandas vs Decimal truth", r)
        cited, unit, explicit = parse_value_text(r["value_text"])
        # directional reading
        dt = r["direction_text"]
        dir_ok = True
        if dt in ("up", "down"):
            dir_ok = (true > 0) if dt == "up" else (true < 0)
            cmp_true = abs(true)
        elif explicit is not None:
            cmp_true = true
        else:
            assert true >= 0, ("negative truth shown unsigned", r)
            cmp_true = true
        q = r["qualifier"]
        if q in ("none", "approx"):
            ratio = abs(cmp_true) / unit
            frac = ratio - ratio.to_integral_value(rounding=ROUND_FLOOR)
            assert abs(frac - Decimal("0.5")) > Decimal("1e-7"), ("half-way tie, ambiguous", r)
            value_ok = half_up(cmp_true, unit) == cited
        elif q == "over":
            value_ok = cmp_true > cited
        elif q == "under":
            value_ok = cmp_true < cited
        else:
            raise ValueError(q)
        ok = dir_ok and value_ok
        # double-rounding guard: a percent cell cited coarser than the cell's 0.1 must agree with raw data
        if expr["op"] == "cell" and expr["col"] in ("WoW_Pct", "YoY_Pct", "Share_Pct") and unit > Decimal("0.1") \
                and q in ("none", "approx"):
            e_, row = expr["ent"], tab[expr["ent"]]
            if expr["col"] == "Share_Pct":
                raw = Decimal(100) * Decimal(row["Revenue_TW"]) / sum(Decimal(tab[x]["Revenue_TW"]) for x in order)
            else:
                den = Decimal(row["Revenue_LW" if expr["col"] == "WoW_Pct" else "Revenue_LY"])
                raw = Decimal(100) * (Decimal(row["Revenue_TW"]) / den - 1)
            rr = abs(raw) if dt in ("up", "down") else raw
            assert half_up(rr, unit) == half_up(cmp_true, unit), ("double rounding disagreement", r)
        if (r["label"] == "C") != ok:
            problems.append((r["note_id"], r["line"], r["value_text"], r["label"], r["error_type"], str(true)))
        if r["label"] == "W":
            assert r["error_type"], r
    assert not problems, problems
    return True


def main():
    for fn in (notes_small_country, notes_small_product_w47, notes_small_product_w07, notes_medium_product_w10,
               notes_medium_product_w05, notes_large_product_w11, notes_large_product_w16):
        fn()
    rows = B.rows
    verify(rows)
    # per-table and global checks
    per_table = {}
    for n in B.notes:
        per_table[n["table_id"]] = per_table.get(n["table_id"], 0) + 1
    assert set(per_table) == set(TABLES) and min(per_table.values()) >= 3
    assert len(B.notes) >= 24
    digit_rows = [r for r in rows if not r["is_word"]]
    assert len(digit_rows) >= 550, len(digit_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "notes.jsonl", "w") as f:
        for n in B.notes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    cols = ["note_id", "table_id", "line", "idx", "start", "end", "value_text", "label", "kind", "entity", "metric",
            "period", "qualifier", "direction_text", "true_value", "cited_value", "precision", "error_type",
            "error_src", "is_word", "count", "expr"]
    with open(OUT / "truth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    from collections import Counter
    lab = Counter(r["label"] for r in digit_rows)
    et = Counter(r["error_type"] for r in digit_rows if r["label"] == "W")
    meas = lab["C"] + lab["W"]
    print(f"notes={len(B.notes)} lines={sum(n['n_lines'] for n in B.notes)} digit_mentions={len(digit_rows)} "
          f"word_mentions={len(rows) - len(digit_rows)}")
    print("labels:", dict(lab), f"W share of measurements={lab['W'] / meas:.3f}")
    print("error types:", dict(et))
    print("per table notes:", per_table)
    print("mentions per table:", dict(Counter(r["table_id"] for r in digit_rows)))


if __name__ == "__main__":
    main()
