"""Stress test of number_gate.gate_level1 on varied, LLM-style executive notes.

Every cited number is registered with a ground truth computed from the table with pandas:
  C    correct number (should be Supported_cell / Supported_derived)
  W    deliberately wrong number (should be Unsupported)
  N    not a measurement (digit inside a product name, 'top 3', '10 markets') -> should be Skipped
  CNT  a count-of-rows claim ('69 of the 135 lines'), skipped by design (method.md 2.2)
  WORD a number written in words ('more than doubled'); not extracted by design (method.md 7)
The truth of every C and W item is asserted with an independent rounding check before the gate runs.

Run from anywhere:  <project>/.venv/bin/python stress_level1.py
Read-only with respect to the project; writes results next to this script.
"""
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = str(Path(__file__).resolve().parents[2])  # project root
sys.path.insert(0, os.path.join(ROOT, "src"))
import number_gate as ng  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))


def load(tid):
    return pd.read_csv(os.path.join(ROOT, "data", "tables", f"{tid}.csv"))


TA, TB, TC, TD = ("small_country_2010-W50", "medium_product_2011-W42",
                  "large_product_2011-W47", "medium_product_2010-W26")
A, B, Cq, D = load(TA), load(TB), load(TC), load(TD)


def v(df, ent, col):
    return float(df.set_index("Entity").loc[ent, col])


def tot(df, col):
    return float(df[col].sum())


def pctc(a, b):
    return (a - b) / b * 100


def share_tw(df, ent):
    return v(df, ent, "Revenue_TW") / tot(df, "Revenue_TW") * 100


def share_lw(df, ent):
    return v(df, ent, "Revenue_LW") / tot(df, "Revenue_LW") * 100


def topk(df, k):
    s = df.sort_values("Revenue_TW", ascending=False)["Revenue_TW"]
    return float(s.iloc[:k].sum()), float(s.iloc[:k].sum() / s.sum() * 100)


def upct(df, ent, m="Units"):
    return pctc(v(df, ent, f"{m}_TW"), v(df, ent, f"{m}_LW"))

# ------------------------------------------------------------------ item registry

ITEMS = []


def _reg(**kw):
    ITEMS.append(kw)
    return f"\x00{len(ITEMS) - 1}\x00"


def C(text, tv, why="", sign=0, kind="std"):
    """Correct number. sign = direction the note asserts (+1 up, -1 down, 0 none)."""
    return _reg(truth="C", text=text, tv=tv, why=why, sign=sign, kind=kind)


def W(text, tv, kind, why="", sign=0):
    """Wrong number. tv = true value of the quantity the sentence claims."""
    return _reg(truth="W", text=text, tv=tv, why=why, sign=sign, kind=kind)


def N(text, why="not a measurement"):
    return _reg(truth="N", text=text, tv=None, why=why, sign=0, kind="non_measure")


def CNT(text, tv, correct, why="count of rows"):
    return _reg(truth="CNT_C" if correct else "CNT_W", text=text, tv=tv, why=why, sign=0, kind="count_claim")


def WORD(text, correct, why=""):
    return _reg(truth="WORD_C" if correct else "WORD_W", text=text, tv=None, why=why, sign=0, kind="word_number")


def my_parse(text):
    t = text.replace(",", "")
    m = re.search(r"\d+(?:\.\d+)?", t)
    num = m.group(0)
    d = len(num.split(".")[1]) if "." in num else 0
    rest = t[m.end():].strip().lower()
    scale = 1e3 if rest.startswith("k") else 1e6 if rest.startswith("m") else 1.0
    return float(num), d, scale


def rounds_to(tv, text):
    cited, d, scale = my_parse(text)
    return abs(abs(tv) / scale - cited) <= 0.5 * 10 ** (-d) + 1e-9


def check_truth(it):
    if it["truth"] == "C":
        if it["kind"] == "truncated":
            cited, d, scale = my_parse(it["text"])
            assert int(abs(it["tv"]) / scale * 10 ** d) / 10 ** d == cited, it
        else:
            assert rounds_to(it["tv"], it["text"]), f"C item does not round: {it}"
        if it["sign"]:
            assert it["sign"] * it["tv"] > 0, f"C item direction wrong: {it}"
    elif it["truth"] == "W":
        if it["kind"] == "direction":
            assert rounds_to(it["tv"], it["text"]) and it["sign"] * it["tv"] < 0, f"bad direction W: {it}"
        else:
            assert not rounds_to(it["tv"], it["text"]), f"W item is actually correct: {it}"
    elif it["truth"] in ("CNT_C", "CNT_W"):
        assert (float(it["text"]) == it["tv"]) == (it["truth"] == "CNT_C"), it

# ------------------------------------------------------------------ notes

NOTES = []


def note(nid, table_id, df, lines):
    NOTES.append({"note_id": nid, "table_id": table_id, "df": df, "lines": lines})


# ---------- Table A: small_country_2010-W50 (10 rows, has Revenue_LY)
a_tw, a_lw, a_ly = tot(A, "Revenue_TW"), tot(A, "Revenue_LW"), tot(A, "Revenue_LY")
a_wow, a_yoy = pctc(a_tw, a_lw), pctc(a_tw, a_ly)
UK = "United Kingdom"
exuk_tw, exuk_lw = a_tw - v(A, UK, "Revenue_TW"), a_lw - v(A, UK, "Revenue_LW")

note("A1_terse", TA, A, [
    f"Total revenue was {C('£208.7k', a_tw)} TW, down by {C('27.7%', a_wow, sign=-1)} WoW from {C('£288.8k', a_lw)} LW, "
    f"and {C('9.9%', a_yoy, 'below is not a gate direction word; unsigned', sign=-1)} below the same week LY ({C('£231.8k', a_ly)}).",
    f"United Kingdom delivered {C('£193.4k', v(A, UK, 'Revenue_TW'))} TW, {C('92.6%', v(A, UK, 'Share_Pct'))} of the total, "
    f"a {W('31.3%', v(A, UK, 'WoW_Pct'), 'drift', 'true -30.3', sign=-1)} drop WoW.",
    f"Germany ranked {C('#2', 2)} with {C('£4,525.14', v(A, 'Germany', 'Revenue_TW'))} TW, up {C('16.2%', v(A, 'Germany', 'WoW_Pct'), sign=1)} WoW "
    f"and {C('42.6%', v(A, 'Germany', 'YoY_Pct'), sign=1)} YoY.",
    f"Sweden booked {C('£3,834.30', v(A, 'Sweden', 'Revenue_TW'))} TW against nil LW, up {C('1,244.0%', v(A, 'Sweden', 'YoY_Pct'), sign=1)} YoY "
    f"from {C('£285.30', v(A, 'Sweden', 'Revenue_LY'))} LY.",
    f"EIRE revenue rose {W('24.9%', v(A, 'EIRE', 'WoW_Pct'), 'direction', 'EIRE fell 24.9%', sign=1)} WoW to {C('£2,392.28', v(A, 'EIRE', 'Revenue_TW'))}.",
    f"France's revenue dropped to just {C('£515.12', v(A, 'France', 'Revenue_TW'), 'level after dropped-to-just')} "
    f"({C('−87.3%', v(A, 'France', 'WoW_Pct'), sign=-1)} WoW) on {C('4', v(A, 'France', 'Orders_TW'))} orders versus "
    f"{W('6', v(A, 'France', 'Orders_LW'), 'drift', 'France Orders_LW is 8')} LW.",
    f"Spain units climbed to {C('395', v(A, 'Spain', 'Units_TW'))} from {C('67', v(A, 'Spain', 'Units_LW'))} LW while orders "
    f"{WORD('doubled', True, 'orders 1 -> 2')} to {C('2', v(A, 'Spain', 'Orders_TW'))}.",
    f"Australia revenue was {C('£415.70', v(A, 'Australia', 'Revenue_TW'))} TW, up {C('60.6%', v(A, 'Australia', 'WoW_Pct'), sign=1)} WoW, "
    f"on {W('214', v(A, 'Australia', 'Units_TW'), 'period', '214 is Units_LW; TW is 146')} units TW.",
])

note("A2_verbose_british", TA, A, [
    f"Group revenue fell to {C('£0.21 million', a_tw)} this week from {C('£0.29 million', a_lw)} last week, "
    f"a decline of {C('27.7 per cent', a_wow, sign=-1)}.",
    f"The United Kingdom remains dominant at {C('92.6 per cent', v(A, UK, 'Share_Pct'))} of sales, although its revenue of "
    f"{C('£193,356', v(A, UK, 'Revenue_TW'))} was {C('30.3 per cent', v(A, UK, 'WoW_Pct'), sign=-1)} lower than last week's "
    f"{W('£277,828', v(A, UK, 'Revenue_LW'), 'drift', 'transposed digits; true 277,288')}.",
    f"UK order count dropped by {C('80', v(A, UK, 'Orders_TW') - v(A, UK, 'Orders_LW'), sign=-1)} to {C('413', v(A, UK, 'Orders_TW'))}, "
    f"a {C('16.2 per cent', upct(A, UK, 'Orders'), 'UK orders pct change', sign=-1)} fall.",
    f"Germany grew {C('16.2 per cent', v(A, 'Germany', 'WoW_Pct'), sign=1)} week-on-week to {C('£4,525', v(A, 'Germany', 'Revenue_TW'))}, "
    f"its units up {W('8.6 per cent', upct(A, 'Germany'), 'drift', 'true units +6.8%', sign=1)}.",
    f"Spain's revenue rose more than {WORD('fivefold', True, '5.2x')}, to {C('£909', v(A, 'Spain', 'Revenue_TW'))}, while Belgium's rose to "
    f"{C('£541', v(A, 'Belgium', 'Revenue_TW'))} from {C('£8.50', v(A, 'Belgium', 'Revenue_LY'))} a year ago.",
    f"France's revenue fell {C('73.3 per cent', v(A, 'France', 'YoY_Pct'), sign=-1)} year-on-year, to {C('£515', v(A, 'France', 'Revenue_TW'))}.",
    f"Cyprus added {W('£1,690.82', v(A, 'Cyprus', 'Revenue_TW'), 'drift', 'true 1,590.82', sign=1)} in its first week of trading, and Finland took "
    f"{C('£652', v(A, 'Finland', 'Revenue_TW'), 'truncated 652.80', kind='truncated')}, up {C('39.2 per cent', v(A, 'Finland', 'YoY_Pct'), sign=1)} "
    f"on last year, on {W('1,284', v(A, 'Finland', 'Units_TW'), 'drift', 'true 1,248')} units.",
    f"EIRE revenue fell {W('26.5 per cent', v(A, 'EIRE', 'WoW_Pct'), 'period', '26.5 is EIRE YoY; WoW is -24.9', sign=-1)} week-on-week to "
    f"{C('£2,392', v(A, 'EIRE', 'Revenue_TW'))}.",
])

note("A3_colon_parenthetical", TA, A, [
    f"UK: {C('£193.4k', v(A, UK, 'Revenue_TW'))} TW ({C('−30.3%', v(A, UK, 'WoW_Pct'), sign=-1)} WoW, {C('−13.0%', v(A, UK, 'YoY_Pct'), sign=-1)} YoY); "
    f"share {C('92.6%', v(A, UK, 'Share_Pct'))}, down {C('3.4pp', share_tw(A, UK) - share_lw(A, UK), 'share TW minus share LW', sign=-1, kind='outside_library')} vs LW.",
    f"Germany: {C('£4.5k', v(A, 'Germany', 'Revenue_TW'))} TW ({C('+16.2%', v(A, 'Germany', 'WoW_Pct'), sign=1)} WoW; "
    f"{C('+42.6%', v(A, 'Germany', 'YoY_Pct'), sign=1)} YoY), {C('10', v(A, 'Germany', 'Orders_TW'))} orders vs {C('7', v(A, 'Germany', 'Orders_LW'))} LW.",
    f"EIRE: {C('£2.4k', v(A, 'EIRE', 'Revenue_TW'))} TW ({C('–24.9%', v(A, 'EIRE', 'WoW_Pct'), 'en dash as minus', sign=-1)} WoW; "
    f"YoY {W('–24.9%', v(A, 'EIRE', 'YoY_Pct'), 'period', 'WoW value cited as YoY; YoY is -26.5', sign=-1)}).",
    f"Spain: {C('£909.01', v(A, 'Spain', 'Revenue_TW'))} TW vs. {C('£174.72', v(A, 'Spain', 'Revenue_LW'))} LW "
    f"({C('5.2x', v(A, 'Spain', 'Revenue_TW') / v(A, 'Spain', 'Revenue_LW'))}).",
    f"France: {C('£515.12', v(A, 'France', 'Revenue_TW'))} TW ({W('–78.3%', v(A, 'France', 'WoW_Pct'), 'drift', 'true -87.3', sign=-1)} WoW); "
    f"Australia: {C('£415.70', v(A, 'Australia', 'Revenue_TW'))} TW ({W('–60.6%', v(A, 'Australia', 'WoW_Pct'), 'direction', 'en dash minus; Australia rose', sign=-1)} WoW).",
    f"Cyprus and Finland were new this week (no LW sales), contributing {C('£1.6k', v(A, 'Cyprus', 'Revenue_TW'))} and "
    f"{C('£0.7k', v(A, 'Finland', 'Revenue_TW'))} respectively.",
    f"Units: {C('106,817', tot(A, 'Units_TW'))} TW, {C('−6.6%', pctc(tot(A, 'Units_TW'), tot(A, 'Units_LW')), sign=-1)} WoW; "
    f"orders: {W('445', tot(A, 'Orders_TW'), 'drift', 'true 440')}, {C('−14.6%', pctc(tot(A, 'Orders_TW'), tot(A, 'Orders_LW')), sign=-1)}.",
    f"Belgium: {C('£541.16', v(A, 'Belgium', 'Revenue_TW'))} TW, {W('#7', v(A, 'Belgium', 'Rank'), 'other_row', 'Belgium is #8; #7 is Finland')}.",
])

note("A4_rounded_headline", TA, A, [
    f"Revenue totalled c.{C('£209k', a_tw)}, {C('28pc', a_wow, sign=-1)} lower week-on-week.",
    f"The UK alone brought in {C('£193k', v(A, UK, 'Revenue_TW'))}, roughly {C('93pc', v(A, UK, 'Share_Pct'))} of sales, while "
    f"{WORD('Seven', True, '7 markets grew')} of the {N('10')} markets grew WoW.",
    f"The top five markets generated {C('98.5%', topk(A, 5)[1])} of revenue, and {CNT('4', int((A.Revenue_TW < A.Revenue_LW).sum()), False, 'true 3 decliners')} "
    f"of the {N('10')} markets posted lower revenue WoW.",
    f"Germany's revenue per order was {C('£452.51', v(A, 'Germany', 'Revenue_TW') / v(A, 'Germany', 'Orders_TW'), 'AOV', kind='outside_library')}, "
    f"versus {C('£468.17', v(A, UK, 'Revenue_TW') / v(A, UK, 'Orders_TW'), 'AOV', kind='outside_library')} for the UK.",
    f"Spain's units rose almost {WORD('six-fold', True, '5.9x')}, to {C('395', v(A, 'Spain', 'Units_TW'))}, and Australia posted a "
    f"{W('60.6%', v(A, 'Australia', 'WoW_Pct'), 'direction', 'Australia rose 60.6%', sign=-1)} week-on-week decline.",
    f"Non-UK markets contributed {C('£15.4k', exuk_tw, 'sum ex-UK', kind='outside_library')} combined, up "
    f"{C(f'{pctc(exuk_tw, exuk_lw):.1f}%', pctc(exuk_tw, exuk_lw), 'pct change ex-UK', sign=1, kind='outside_library')} WoW.",
    f"France's share slipped to {C('0.2%', v(A, 'France', 'Share_Pct'))} from {C('1.4%', share_lw(A, 'France'))} LW.",
    f"Germany's LY revenue of {W('£3.9k', v(A, 'Germany', 'Revenue_LY'), 'period', '3.9k is Germany LW; LY is 3,173')} compares with "
    f"{C('£4.5k', v(A, 'Germany', 'Revenue_TW'))} TW; Sweden entered at {C('#3', 3)} with "
    f"{W('£38.3k', v(A, 'Sweden', 'Revenue_TW'), 'drift', 'decimal slip x10; true 3.8k')}.",
])

# ---------- Table B: medium_product_2011-W42 (33 rows, has Revenue_LY)
b_tw, b_lw, b_ly = tot(B, "Revenue_TW"), tot(B, "Revenue_LW"), tot(B, "Revenue_LY")
RC = "REGENCY CAKESTAND 3 TIER"
PCK50 = "PAPER CHAIN KIT 50'S CHRISTMAS"
JB = "JUMBO BAG RED RETROSPOT"

note("B1_terse", TB, B, [
    f"Total revenue was {C('£58.3k', b_tw)} TW, up {C('150.8%', pctc(b_tw, b_lw), sign=1)} WoW from {C('£23.3k', b_lw)} LW and "
    f"{C('180.6%', pctc(b_tw, b_ly), sign=1)} above LY ({C('£20.8k', b_ly)}).",
    f"{WORD('Six', True, '6 Landmark lines with nil LW')} new LANDMARK FRAME lines (all nil LW) took the top six spots, together "
    f"{C('£21.7k', topk(B, 6)[0])} or {C('37.1%', topk(B, 6)[1])} of revenue.",
    f"LANDMARK FRAME COVENT GARDEN led with {C('£3,766.40', v(B, 'LANDMARK FRAME COVENT GARDEN', 'Revenue_TW'))} TW on "
    f"{W('304', v(B, 'LANDMARK FRAME COVENT GARDEN', 'Units_TW'), 'drift', 'true 340 units')} units and "
    f"{C('12', v(B, 'LANDMARK FRAME COVENT GARDEN', 'Orders_TW'))} orders.",
    f"REGENCY CAKESTAND {N('3')} TIER was {C('#7', v(B, RC, 'Rank'))} at {C('£3,205.71', v(B, RC, 'Revenue_TW'))}, up "
    f"{C('17.5%', v(B, RC, 'WoW_Pct'), sign=1)} WoW but down {C('33.8%', v(B, RC, 'YoY_Pct'), sign=-1)} YoY.",
    f"GINGHAM HEART DECORATION units jumped to {C('2,130', v(B, 'GINGHAM HEART DECORATION', 'Units_TW'))} from "
    f"{C('80', v(B, 'GINGHAM HEART DECORATION', 'Units_LW'))} LW, lifting revenue "
    f"{C('2,054.1%', v(B, 'GINGHAM HEART DECORATION', 'WoW_Pct'), sign=1)} WoW.",
    f"PAPER CHAIN KIT {N('50')}'S CHRISTMAS slipped {C('5.1%', v(B, PCK50, 'WoW_Pct'), sign=-1)} WoW to {C('£2,222.96', v(B, PCK50, 'Revenue_TW'))} "
    f"but was {C('72.3%', v(B, PCK50, 'YoY_Pct'), sign=1)} ahead of LY.",
    f"BOX OF {N('6')} CHRISTMAS CAKE DECORATIONS rose {W('232.6%', v(B, 'BOX OF 6 CHRISTMAS CAKE DECORATIONS', 'WoW_Pct'), 'drift', 'true 332.6', sign=1)} "
    f"WoW to {C('£1,316.72', v(B, 'BOX OF 6 CHRISTMAS CAKE DECORATIONS', 'Revenue_TW'))}, while DOORMAT KEEP CALM AND COME IN grew "
    f"{W('34.4%', v(B, 'DOORMAT KEEP CALM AND COME IN', 'WoW_Pct'), 'direction', 'doormat fell 34.4%', sign=1)}.",
    f"JUMBO BAG RED RETROSPOT orders rose to {W('47', v(B, JB, 'Orders_TW'), 'period', 'TW/LW swapped: TW is 41')} from "
    f"{W('41', v(B, JB, 'Orders_LW'), 'period', 'TW/LW swapped: LW is 47')} LW.",
])

note("B2_gpt_mixed_units", TB, B, [
    f"Revenue more than {WORD('doubled', True, '2.5x')} week-on-week ({C('+150.8%', pctc(b_tw, b_lw), sign=1)}) to {C('GBP 58,323', b_tw)}, "
    f"with units up {C('104.5%', pctc(tot(B, 'Units_TW'), tot(B, 'Units_LW')), sign=1)} and orders up "
    f"{W('32.7%', pctc(tot(B, 'Orders_TW'), tot(B, 'Orders_LW')), 'drift', 'true +23.7%', sign=1)}.",
    f"The Regency Cakestand {N('3')} Tier sold {C('268', v(B, RC, 'Units_TW'))} units (vs. {C('214', v(B, RC, 'Units_LW'))} LW), a "
    f"{C('25.2%', upct(B, RC), sign=1)} increase, but orders fell {C('24%', upct(B, RC, 'Orders'), sign=-1)} to {C('38', v(B, RC, 'Orders_TW'))}.",
    f"Botanical Gardens Wall Clock, new this week, generated {C('GBP 2,928.80', v(B, 'BOTANICAL GARDENS WALL CLOCK', 'Revenue_TW'))} from "
    f"{C('8', v(B, 'BOTANICAL GARDENS WALL CLOCK', 'Orders_TW'))} orders, an average of "
    f"{C('GBP 366.10', v(B, 'BOTANICAL GARDENS WALL CLOCK', 'Revenue_TW') / 8, 'AOV', kind='outside_library')} per order.",
    f"Gingham Heart Decoration's revenue grew {C('21.5x', v(B, 'GINGHAM HEART DECORATION', 'Revenue_TW') / v(B, 'GINGHAM HEART DECORATION', 'Revenue_LW'), sign=1)} "
    f"week-on-week to {C('GBP 1,565.58', v(B, 'GINGHAM HEART DECORATION', 'Revenue_TW'))}, while Chocolate Hot Water Bottle added "
    f"{C('+£0.9k', v(B, 'CHOCOLATE HOT WATER BOTTLE', 'Revenue_TW') - v(B, 'CHOCOLATE HOT WATER BOTTLE', 'Revenue_LW'), sign=1)} "
    f"({C('+134.4%', v(B, 'CHOCOLATE HOT WATER BOTTLE', 'WoW_Pct'), sign=1)}).",
    f"Assorted Colour Bird Ornament revenue was down {W('49.3%', v(B, 'ASSORTED COLOUR BIRD ORNAMENT', 'WoW_Pct'), 'period', '-49.3 is YoY; WoW is +86.6', sign=-1)} "
    f"week-on-week at {C('GBP 1,324.21', v(B, 'ASSORTED COLOUR BIRD ORNAMENT', 'Revenue_TW'))}.",
    f"The {N('3')}-tier Regency cakestand remains the top non-Landmark line with a {C('5.5%', v(B, RC, 'Share_Pct'))} share.",
    f"Round Snack Boxes Set of {N('4')} Woodland jumped {W('1,060%', v(B, 'ROUND SNACK BOXES SET OF4 WOODLAND', 'YoY_Pct'), 'drift', 'true 1,006.0%', sign=1)} "
    f"YoY to {C('GBP 1,044.04', v(B, 'ROUND SNACK BOXES SET OF4 WOODLAND', 'Revenue_TW'))}.",
    f"Retrospot Tea Set Ceramic {N('11')}pc rose {C('33.6%', v(B, 'RETROSPOT TEA SET CERAMIC 11 PC', 'WoW_Pct'), sign=1)} WoW to "
    f"{C('£1,014.24', v(B, 'RETROSPOT TEA SET CERAMIC 11 PC', 'Revenue_TW'))}.",
])

PCKV = "PAPER CHAIN KIT VINTAGE CHRISTMAS"
note("B3_shares_pp_endash", TB, B, [
    f"Share of revenue for REGENCY CAKESTAND {N('3')} TIER fell to {C('5.5%', v(B, RC, 'Share_Pct'))} (LW: {C('11.7%', share_lw(B, RC))}), a "
    f"{C('6.2pp', share_tw(B, RC) - share_lw(B, RC), 'share change', sign=-1, kind='outside_library')} drop.",
    f"PAPER CHAIN KIT {N('50')}'S CHRISTMAS share: {C('3.8%', v(B, PCK50, 'Share_Pct'))} TW vs {C('10.1%', share_lw(B, PCK50))} LW – down "
    f"{C('6.3 points', share_tw(B, PCK50) - share_lw(B, PCK50), 'share change', sign=-1, kind='outside_library')}.",
    f"JUMBO BAG RED RETROSPOT: {C('£2,021.85', v(B, JB, 'Revenue_TW'))} TW – {C('+7.2%', v(B, JB, 'WoW_Pct'), sign=1)} WoW, "
    f"{W('+42.3%', v(B, JB, 'YoY_Pct'), 'drift', 'near miss; true +42.9%', sign=1)} YoY.",
    f"CHRISTMAS LIGHTS {N('10')} VINTAGE BAUBLES: {C('£896.84', v(B, 'CHRISTMAS LIGHTS 10 VINTAGE BAUBLES', 'Revenue_TW'))} TW, "
    f"{C('–54.3%', v(B, 'CHRISTMAS LIGHTS 10 VINTAGE BAUBLES', 'YoY_Pct'), sign=-1)} YoY (LY: {C('£1,960.94', v(B, 'CHRISTMAS LIGHTS 10 VINTAGE BAUBLES', 'Revenue_LY'))}).",
    f"PAPER CHAIN KIT VINTAGE CHRISTMAS revenue fell to roughly {C('£935', v(B, PCKV, 'Revenue_TW'), 'level after fell-to-roughly')} from "
    f"{C('£1,283', v(B, PCKV, 'Revenue_LW'))} LW; PARTY BUNTING was {W('−70.6%', v(B, 'PARTY BUNTING', 'WoW_Pct'), 'direction', 'party bunting rose 70.6%', sign=-1)} WoW.",
    f"WALL ART BIG LOVE: {C('£858.88', v(B, 'WALL ART BIG LOVE', 'Revenue_TW'))} TW, "
    f"{C('3.4x', v(B, 'WALL ART BIG LOVE', 'Revenue_TW') / v(B, 'WALL ART BIG LOVE', 'Revenue_LW'))} LW "
    f"({W('£215.12', v(B, 'WALL ART BIG LOVE', 'Revenue_LW'), 'drift', 'true 251.12')}).",
    f"RED HARMONICA IN BOX: YoY {C('+26,576.5%', v(B, 'RED  HARMONICA IN BOX', 'YoY_Pct'), sign=1)} off a "
    f"{C('£3.75', v(B, 'RED  HARMONICA IN BOX', 'Revenue_LY'))} base LY.",
    f"Units across the range: {C('14,739', tot(B, 'Units_TW'))} TW vs {C('7,209', tot(B, 'Units_LW'))} LW "
    f"({C('2.0x', tot(B, 'Units_TW') / tot(B, 'Units_LW'))}); SET OF {N('3')} REGENCY CAKE TINS orders were flat at "
    f"{C('31', v(B, 'SET OF 3 REGENCY CAKE TINS', 'Orders_TW'))}, revenue up "
    f"{W('38.3%', v(B, 'SET OF 3 REGENCY CAKE TINS', 'WoW_Pct'), 'drift', 'true +83.3%', sign=1)} WoW.",
])

# ---------- Table C: large_product_2011-W47 (135 rows, has Revenue_LY)
c_tw, c_lw, c_ly = tot(Cq, "Revenue_TW"), tot(Cq, "Revenue_LW"), tot(Cq, "Revenue_LY")
WHH = "WHITE HANGING HEART T-LIGHT HOLDER"
RSA = "ROTATING SILVER ANGELS T-LIGHT HLDR"
J50 = "JUMBO BAG 50'S CHRISTMAS"

note("C1_terse", TC, Cq, [
    f"Total revenue was {W('£135.4k', c_tw, 'drift_small', 'true 135.25k -> 135.3k; 0.11% off')} TW, down "
    f"{C('15.0%', pctc(c_tw, c_lw), sign=-1)} WoW ({C('£159.1k', c_lw)} LW) but up {C('57.9%', pctc(c_tw, c_ly), sign=1)} YoY.",
    f"PAPER CHAIN KIT {N('50')}'S CHRISTMAS remained the No.{C('1', 1, 'rank written No.1')} line at {C('£5,701.16', v(Cq, PCK50, 'Revenue_TW'))} TW, "
    f"down {C('39.9%', v(Cq, PCK50, 'WoW_Pct'), sign=-1)} WoW and {C('0.6%', v(Cq, PCK50, 'YoY_Pct'), sign=-1)} YoY.",
    f"RABBIT NIGHT LIGHT more than {WORD('halved', True, '-52.4%')}: {C('£5,084.81', v(Cq, 'RABBIT NIGHT LIGHT', 'Revenue_TW'))} TW vs "
    f"{C('£10,687.69', v(Cq, 'RABBIT NIGHT LIGHT', 'Revenue_LW'))} LW ({C('-52.4%', v(Cq, 'RABBIT NIGHT LIGHT', 'WoW_Pct'), sign=-1)}).",
    f"CHILLI LIGHTS more than {WORD('doubled', True, '+115%')} to {C('£4,960.25', v(Cq, 'CHILLI LIGHTS', 'Revenue_TW'))} "
    f"({C('+115.0%', v(Cq, 'CHILLI LIGHTS', 'WoW_Pct'), sign=1)} WoW) and was up {C('229.3%', v(Cq, 'CHILLI LIGHTS', 'YoY_Pct'), sign=1)} YoY; "
    f"{WORD('two of the top three', True, 'rank 1 and 2 fell, rank 3 rose')} lines fell WoW.",
    f"The top {N('3')} products took {C('11.6%', topk(Cq, 3)[1])} of revenue ({C('£15.7k', topk(Cq, 3)[0])}).",
    f"REGENCY CAKESTAND {N('3')} TIER revenue rose {C('17.3%', v(Cq, RC, 'WoW_Pct'), sign=1)} WoW to {C('£3,543.60', v(Cq, RC, 'Revenue_TW'))} but fell "
    f"{C('45.9%', v(Cq, RC, 'YoY_Pct'), sign=-1)} YoY; units rose {W('11.6%', upct(Cq, RC), 'drift', 'true units +11.1%', sign=1)} to "
    f"{C('261', v(Cq, RC, 'Units_TW'))}.",
    f"WHITE HANGING HEART T-LIGHT HOLDER fell {C('35.0%', v(Cq, WHH, 'WoW_Pct'), sign=-1)} WoW to {C('£2,475.48', v(Cq, WHH, 'Revenue_TW'))}, with units "
    f"down {C('31.8%', upct(Cq, WHH), sign=-1)} to {C('904', v(Cq, WHH, 'Units_TW'))} and YoY down "
    f"{W('45.9%', v(Cq, WHH, 'YoY_Pct'), 'other_row', '-45.9 is Regency YoY; WHH YoY is -58.7', sign=-1)}.",
    f"DOORMAT BLACK FLOCK had the biggest WoW jump, up {W('1,454.7%', v(Cq, 'DOORMAT BLACK FLOCK', 'WoW_Pct'), 'drift', 'true 1,544.7%', sign=1)} to "
    f"{C('£542.76', v(Cq, 'DOORMAT BLACK FLOCK', 'Revenue_TW'))}.",
])

note("C2_gpt_verbose", TC, Cq, [
    f"Weekly sales came in at {C('£0.14m', c_tw)}, {C('15pc', pctc(c_tw, c_lw), sign=-1)} down on the prior week's {C('£0.16m', c_lw)}.",
    f"Against the same week last year, revenue is up {C('57.9 per cent', pctc(c_tw, c_ly), sign=1)}, from {C('£85.7k', c_ly)}.",
    f"Units fell {W('11.9 per cent', pctc(tot(Cq, 'Units_TW'), tot(Cq, 'Units_LW')), 'arithmetic', 'computed on TW base; true -10.7%', sign=-1)} to "
    f"{C('46,342', tot(Cq, 'Units_TW'))} and orders {C('6.1 per cent', pctc(tot(Cq, 'Orders_TW'), tot(Cq, 'Orders_LW')), sign=-1)} to "
    f"{C('4,027', tot(Cq, 'Orders_TW'))}.",
    f"Average revenue per product was {C('£1,001.88', Cq.Revenue_TW.mean())}; the median line took "
    f"{W('£702.39', float(Cq.Revenue_TW.median()), 'drift', 'true median 720.39')}.",
    f"Rabbit Night Light units dropped {C('45.5 per cent', upct(Cq, 'RABBIT NIGHT LIGHT'), sign=-1)} to {C('2,331', v(Cq, 'RABBIT NIGHT LIGHT', 'Units_TW'))}, "
    f"while Popcorn Holder units rose {W('6.0 per cent', upct(Cq, 'POPCORN HOLDER'), 'direction', 'popcorn units fell 6.0%', sign=1)} to "
    f"{C('2,955', v(Cq, 'POPCORN HOLDER', 'Units_TW'))}.",
    f"Hot Water Bottle Keep Calm revenue was broadly flat at {C('£2,401.04', v(Cq, 'HOT WATER BOTTLE KEEP CALM', 'Revenue_TW'))}, "
    f"{W('2.8 per cent', v(Cq, 'HOT WATER BOTTLE KEEP CALM', 'WoW_Pct'), 'direction', 'it fell 2.8%', sign=1)} up on last week.",
    f"Jumbo Bag Red Retrospot revenue {WORD('halved', True, '-49%')}, down {C('49.0 per cent', v(Cq, JB, 'WoW_Pct'), sign=-1)} to "
    f"{C('£1,581.84', v(Cq, JB, 'Revenue_TW'))}, versus {W('£2,221.19', v(Cq, JB, 'Revenue_LW'), 'period', '2,221.19 is LY; LW is 3,102.01')} in the prior week.",
    f"Coloured Glass Star T-Light Holder increased "
    f"{C('11.5x', v(Cq, 'COLOURED GLASS STAR T-LIGHT HOLDER', 'Revenue_TW') / v(Cq, 'COLOURED GLASS STAR T-LIGHT HOLDER', 'Revenue_LW'), sign=1)} "
    f"WoW to {C('£1,278.75', v(Cq, 'COLOURED GLASS STAR T-LIGHT HOLDER', 'Revenue_TW'))}.",
])

G15 = "15CM CHRISTMAS GLASS BALL 20 LIGHTS"
H3 = "3 HEARTS HANGING DECORATION RUSTIC"
V3 = "SET/3 VANILLA SCENTED CANDLE IN BOX"
note("C3_names_with_digits", TC, Cq, [
    f"{N('15')}cm Christmas Glass Ball {N('20')} Lights: {C('£1.0k', v(Cq, G15, 'Revenue_TW'))} TW, {C('+110.9%', v(Cq, G15, 'WoW_Pct'), sign=1)} WoW, "
    f"{C('−16.1%', v(Cq, G15, 'YoY_Pct'), sign=-1)} YoY.",
    f"{N('3')} Hearts Hanging Decoration Rustic: {C('£788.05', v(Cq, H3, 'Revenue_TW'))} TW, {C('−73.8%', v(Cq, H3, 'WoW_Pct'), sign=-1)} WoW; units "
    f"{C('271', v(Cq, H3, 'Units_TW'))} vs {W('508', v(Cq, H3, 'Units_LW'), 'drift', 'true 580')} LW.",
    f"Set/{N('3')} Vanilla Scented Candle in Box: {C('£804', v(Cq, V3, 'Revenue_TW'))} TW vs {C('£59', v(Cq, V3, 'Revenue_LW'))} LW "
    f"({C('13.6x', v(Cq, V3, 'Revenue_TW') / v(Cq, V3, 'Revenue_LW'))}).",
    f"{N('60')} Cake Cases Vintage Christmas: {C('£492.55', v(Cq, '60 CAKE CASES VINTAGE CHRISTMAS', 'Revenue_TW'))} TW, "
    f"{C('−48.3%', v(Cq, '60 CAKE CASES VINTAGE CHRISTMAS', 'WoW_Pct'), sign=-1)} WoW, ranked {C('127th', v(Cq, '60 CAKE CASES VINTAGE CHRISTMAS', 'Rank'))}.",
    f"Set {N('7')} Babushka Nesting Boxes: {C('75', v(Cq, 'SET 7 BABUSHKA NESTING BOXES', 'Units_TW'))} units TW, orders "
    f"{C('−14.3%', upct(Cq, 'SET 7 BABUSHKA NESTING BOXES', 'Orders'), sign=-1)} WoW.",
    f"Wood {N('2')} Drawer Cabinet White Finish: {C('£642.84', v(Cq, 'WOOD 2 DRAWER CABINET WHITE FINISH', 'Revenue_TW'))} TW, "
    f"{W('+161.6%', v(Cq, 'WOOD 2 DRAWER CABINET WHITE FINISH', 'WoW_Pct'), 'drift', 'true +116.6%', sign=1)} WoW.",
    f"Medium Ceramic Top Storage Jar: {C('£473.07', v(Cq, 'MEDIUM CERAMIC TOP STORAGE JAR', 'Revenue_TW'))} TW, the smallest line "
    f"({W('#135', v(Cq, 'MEDIUM CERAMIC TOP STORAGE JAR', 'Rank'), 'other_row', 'it is #134; #135 is Plasters in Tin')}), up "
    f"{C('872.8%', v(Cq, 'MEDIUM CERAMIC TOP STORAGE JAR', 'WoW_Pct'), sign=1)} WoW.",
    f"Gin + Tonic Diet Metal Sign: {C('£894.40', v(Cq, 'GIN + TONIC DIET METAL SIGN', 'Revenue_TW'))} TW "
    f"({C('+1.1%', v(Cq, 'GIN + TONIC DIET METAL SIGN', 'WoW_Pct'), sign=1)} WoW), "
    f"{W('+131.7%', v(Cq, 'GIN + TONIC DIET METAL SIGN', 'YoY_Pct'), 'drift', 'true +113.7%', sign=1)} YoY; Vintage Christmas Bunting "
    f"{C('−69.7%', v(Cq, 'VINTAGE CHRISTMAS BUNTING', 'WoW_Pct'), sign=-1)} WoW to {C('£828.82', v(Cq, 'VINTAGE CHRISTMAS BUNTING', 'Revenue_TW'))}, "
    f"from {C('£2,734.70', v(Cq, 'VINTAGE CHRISTMAS BUNTING', 'Revenue_LW'))}.",
])

note("C4_error_heavy", TC, Cq, [
    f"Revenue fell by {C('£23.8k', c_tw - c_lw, sign=-1)} WoW, with {CNT('69', int((Cq.Revenue_TW < Cq.Revenue_LW).sum()), True)} of the "
    f"{N('135')} lines declining, including {CNT('2', 2, True, 'ranks 1 and 2 fell')} of the top {N('3')}.",
    f"RABBIT NIGHT LIGHT was the second-largest line at {C('£5.1k', v(Cq, 'RABBIT NIGHT LIGHT', 'Revenue_TW'))}, though revenue nearly halved from "
    f"{C('£10.7k', v(Cq, 'RABBIT NIGHT LIGHT', 'Revenue_LW'))} LW.",
    f"CHILLI LIGHTS revenue {WORD('nearly tripled', False, 'true 3.29x: more than tripled')} YoY to {C('£4,960.25', v(Cq, 'CHILLI LIGHTS', 'Revenue_TW'))} "
    f"from {C('£1,506.40', v(Cq, 'CHILLI LIGHTS', 'Revenue_LY'))} LY.",
    f"PAPER CHAIN KIT VINTAGE CHRISTMAS revenue was {W('£3,304.33', v(Cq, PCKV, 'Revenue_TW'), 'drift', 'transposed; true 3,034.33')} TW "
    f"({C('−32.7%', v(Cq, PCKV, 'WoW_Pct'), sign=-1)} WoW).",
    f"PARTY BUNTING more than doubled WoW ({C('+106.7%', v(Cq, 'PARTY BUNTING', 'WoW_Pct'), sign=1)}) to {C('£2,182.49', v(Cq, 'PARTY BUNTING', 'Revenue_TW'))}, "
    f"and was up {W('364.2%', v(Cq, 'PARTY BUNTING', 'YoY_Pct'), 'drift', 'true +463.2%', sign=1)} YoY.",
    f"ROTATING SILVER ANGELS T-LIGHT HLDR units rose {C('59.2%', upct(Cq, RSA), sign=1)} to {C('1,170', v(Cq, RSA, 'Units_TW'))}, but revenue only rose "
    f"{C('10.6%', v(Cq, RSA, 'WoW_Pct'), sign=1)} as orders fell {C('11.9%', upct(Cq, RSA, 'Orders'), sign=-1)} to "
    f"{W('42', v(Cq, RSA, 'Orders_TW'), 'period', '42 is Orders_LW; TW is 37')}.",
    f"JUMBO BAG {N('50')}'S CHRISTMAS revenue fell {W('96.0%', v(Cq, J50, 'WoW_Pct'), 'drift', 'true -69.0%', sign=-1)} "
    f"WoW to {C('£1,151.45', v(Cq, J50, 'Revenue_TW'))}.",
    f"Top-{N('10')} lines generated {W('28.5%', topk(Cq, 10)[1], 'drift', 'true 25.8%')} of revenue ({C('£34.9k', topk(Cq, 10)[0])}).",
])

# ---------- Table D: medium_product_2010-W26 (26 rows, NO Revenue_LY)
d_tw, d_lw = tot(D, "Revenue_TW"), tot(D, "Revenue_LW")
note("D1_terse", TD, D, [
    f"Total revenue was {C('£36.9k', d_tw)} TW, up {C('120.6%', pctc(d_tw, d_lw), sign=1)} WoW from {C('£16.7k', d_lw)} LW.",
    f"REGENCY CAKESTAND {N('3')} TIER led with {C('£4,554.09', v(D, RC, 'Revenue_TW'))} ({C('12.3%', v(D, RC, 'Share_Pct'))} share), up "
    f"{C('85.4%', v(D, RC, 'WoW_Pct'), sign=1)} WoW on {W('319', v(D, RC, 'Units_TW'), 'drift', 'true 391')} units.",
    f"WHITE HANGING HEART T-LIGHT HOLDER was {C('#2', v(D, WHH, 'Rank'))} at {C('£3,215', v(D, WHH, 'Revenue_TW'))} TW, up "
    f"{C('47.9%', v(D, WHH, 'WoW_Pct'), sign=1)} WoW.",
    f"PAPER CHAIN KIT SKULLS surged {C('3,983.9%', v(D, 'PAPER CHAIN KIT SKULLS', 'WoW_Pct'), sign=1)} WoW to "
    f"{C('£1,445.70', v(D, 'PAPER CHAIN KIT SKULLS', 'Revenue_TW'))}, from just {C('£35.40', v(D, 'PAPER CHAIN KIT SKULLS', 'Revenue_LW'))} LW.",
    f"ASSORTED COLOURS SILK FAN was the only decliner, down {C('5.0%', v(D, 'ASSORTED COLOURS SILK FAN', 'WoW_Pct'), sign=-1)} WoW to "
    f"{C('£960.95', v(D, 'ASSORTED COLOURS SILK FAN', 'Revenue_TW'))}.",
    f"LARGE RED SPOT WINDMILL was new this week with {C('£1,332.00', v(D, 'LARGE RED SPOT WINDMILL', 'Revenue_TW'))} from a single order "
    f"({C('720', v(D, 'LARGE RED SPOT WINDMILL', 'Units_TW'))} units).",
    f"JUMBO BAG RED RETROSPOT rose {W('87.1%', v(D, JB, 'WoW_Pct'), 'metric', '87.1% is its units growth; revenue WoW is 78.8%', sign=1)} WoW to "
    f"{C('£2,489.35', v(D, JB, 'Revenue_TW'))} on {C('1,405', v(D, JB, 'Units_TW'))} units.",
    f"PICNIC BASKET WICKER LARGE orders increased to {W('24', v(D, 'PICNIC BASKET WICKER LARGE', 'Orders_TW'), 'period', 'swapped; TW is 19')} from "
    f"{W('19', v(D, 'PICNIC BASKET WICKER LARGE', 'Orders_LW'), 'period', 'swapped; LW is 24')}.",
])

note("D2_mixed", TD, D, [
    f"Revenue rose by {C('£20,170', d_tw - d_lw, sign=1)} week-on-week, to {C('£36,893', d_tw)}.",
    f"Units more than doubled ({C('+106.5%', pctc(tot(D, 'Units_TW'), tot(D, 'Units_LW')), sign=1)}) to {C('14,987', tot(D, 'Units_TW'))}, while orders grew "
    f"{C('20.4 per cent', pctc(tot(D, 'Orders_TW'), tot(D, 'Orders_LW')), sign=1)} to {C('626', tot(D, 'Orders_TW'))}.",
    f"The top five products generated {W('41.5%', topk(D, 5)[1], 'drift', 'true 38.5%')} of revenue, or {C('£14.2k', topk(D, 5)[0])}.",
    f"REGENCY CAKESTAND {N('3')} TIER share slipped to {C('12.3%', v(D, RC, 'Share_Pct'))} from {C('14.7%', share_lw(D, RC))} LW, down "
    f"{C('2.3pp', share_tw(D, RC) - share_lw(D, RC), 'share change', sign=-1, kind='outside_library')}.",
    f"PAPER BUNTING RETRO SPOTS revenue rose {C('160.0%', v(D, 'PAPER BUNTING RETRO SPOTS', 'WoW_Pct'), sign=1)} WoW, with units up "
    f"{W('241.5%', upct(D, 'PAPER BUNTING RETRO SPOTS'), 'drift', 'true +214.5%', sign=1)}.",
    f"PAPER CHAIN KIT RETRO SPOT: {C('8.1x', v(D, 'PAPER CHAIN KIT RETRO SPOT', 'Revenue_TW') / v(D, 'PAPER CHAIN KIT RETRO SPOT', 'Revenue_LW'))} LW revenue "
    f"({C('£1,573.81', v(D, 'PAPER CHAIN KIT RETRO SPOT', 'Revenue_TW'))} vs {C('£194.74', v(D, 'PAPER CHAIN KIT RETRO SPOT', 'Revenue_LW'))}).",
    f"CHILDS BREAKFAST SET DOLLY GIRL grew {C('9,039.0%', v(D, 'CHILDS BREAKFAST SET DOLLY GIRL', 'WoW_Pct'), sign=1)} WoW off "
    f"{C('£9.95', v(D, 'CHILDS BREAKFAST SET DOLLY GIRL', 'Revenue_LW'))}; CHILDS BREAKFAST SET SPACEBOY grew "
    f"{C('3,144.7%', v(D, 'CHILDS BREAKFAST SET SPACEBOY', 'WoW_Pct'), sign=1)} to "
    f"{W('£986.55', v(D, 'CHILDS BREAKFAST SET SPACEBOY', 'Revenue_TW'), 'drift', 'true 968.55')}.",
    f"LUNCH BAG RED SPOTTY was flat ({C('+0.3%', v(D, 'LUNCH BAG RED SPOTTY', 'WoW_Pct'), sign=1)} WoW) at {C('£876.99', v(D, 'LUNCH BAG RED SPOTTY', 'Revenue_TW'))}, "
    f"ranked {W('25th', v(D, 'LUNCH BAG RED SPOTTY', 'Rank'), 'other_row', 'it is 26th')}; VINTAGE UNION JACK BUNTING sold "
    f"{C('198', v(D, 'VINTAGE UNION JACK BUNTING', 'Units_TW'))} units at an average "
    f"{C('£8.53', v(D, 'VINTAGE UNION JACK BUNTING', 'Revenue_TW') / v(D, 'VINTAGE UNION JACK BUNTING', 'Units_TW'), 'avg price', kind='outside_library')}, down from "
    f"{C('£8.87', v(D, 'VINTAGE UNION JACK BUNTING', 'Revenue_LW') / v(D, 'VINTAGE UNION JACK BUNTING', 'Units_LW'), 'avg price LW', kind='outside_library')} LW.",
])

note("D3_error_heavy", TD, D, [
    f"Revenue was up {W('102.6%', pctc(d_tw, d_lw), 'drift', 'true +120.6%', sign=1)} on last week at {C('£36.9k', d_tw)}.",
    f"Orders rose {W('26pc', pctc(tot(D, 'Orders_TW'), tot(D, 'Orders_LW')), 'drift', 'true +20.4%', sign=1)} WoW to {C('626', tot(D, 'Orders_TW'))}.",
    f"JUMBO BAG STRAWBERRY fell {W('101.2%', v(D, 'JUMBO BAG STRAWBERRY', 'WoW_Pct'), 'direction', 'it rose 101.2%', sign=-1)} WoW to "
    f"{C('£1,544.16', v(D, 'JUMBO BAG STRAWBERRY', 'Revenue_TW'))}.",
    f"SET/{N('5')} RED SPOTTY LID GLASS BOWLS rose {C('302.4%', v(D, 'SET/5 RED SPOTTY LID GLASS BOWLS', 'WoW_Pct'), sign=1)} to "
    f"{W('£943.08', v(D, 'SET/5 RED SPOTTY LID GLASS BOWLS', 'Revenue_TW'), 'drift', 'true 934.08')}.",
    f"RETRO SPORT PARTY BAG + STICKER SET revenue rose {C('812.2%', v(D, 'RETRO SPORT PARTY BAG + STICKER SET', 'WoW_Pct'), sign=1)} WoW to "
    f"{C('£997.20', v(D, 'RETRO SPORT PARTY BAG + STICKER SET', 'Revenue_TW'))} from "
    f"{W('£190.32', v(D, 'RETRO SPORT PARTY BAG + STICKER SET', 'Revenue_LW'), 'drift', 'true 109.32')}.",
    f"JUMBO BAG PINK WITH WHITE SPOTS saw revenue decline week on week by "
    f"{W('25.6%', v(D, 'JUMBO BAG PINK WITH WHITE SPOTS', 'WoW_Pct'), 'direction', 'it rose 25.6%; direction word 5 tokens back', sign=-1)} to "
    f"{C('£1,085.14', v(D, 'JUMBO BAG PINK WITH WHITE SPOTS', 'Revenue_TW'))}.",
    f"ASSORTED COLOUR BIRD ORNAMENT units grew {W('138.2%', upct(D, 'ASSORTED COLOUR BIRD ORNAMENT'), 'metric', '138.2 is revenue WoW; units +174.6%', sign=1)} to "
    f"{C('692', v(D, 'ASSORTED COLOUR BIRD ORNAMENT', 'Units_TW'))}.",
    f"Jumbo Bag Baroque Black White revenue rose {C('139.0%', v(D, 'JUMBO  BAG BAROQUE BLACK WHITE', 'WoW_Pct'), sign=1)} to "
    f"{C('£1,131.77', v(D, 'JUMBO  BAG BAROQUE BLACK WHITE', 'Revenue_TW'))}.",
])

# ---------- extra error-heavy notes (probe whether the failure classes generalise)
note("A5_error_heavy", TA, A, [
    f"Total revenue slipped to {W('£209.5k', a_tw, 'drift_small', 'true 208.7k; 0.37% off')} TW, "
    f"{W('19.9%', a_yoy, 'drift', 'true -9.9% YoY', sign=-1)} down on the same week LY.",
    f"UK revenue rose {W('£83.9k', v(A, UK, 'Revenue_TW') - v(A, UK, 'Revenue_LW'), 'direction', 'UK fell by 83.9k', sign=1)} WoW to "
    f"{C('£193.4k', v(A, UK, 'Revenue_TW'))}.",
    f"Germany took {W('4pc', share_tw(A, 'Germany'), 'drift', 'true 2.2% share')} of revenue, up from {C('1.3%', share_lw(A, 'Germany'))} LW.",
    f"Spain revenue was {W('6.2x', v(A, 'Spain', 'Revenue_TW') / v(A, 'Spain', 'Revenue_LW'), 'drift', 'true 5.2x')} LW at "
    f"{C('£909.01', v(A, 'Spain', 'Revenue_TW'))}.",
    f"France units dropped to only {C('533', v(A, 'France', 'Units_TW'), 'level after dropped-to-only')}, from {C('2,280', v(A, 'France', 'Units_LW'))} LW.",
    f"EIRE was the No.{W('3', v(A, 'EIRE', 'Rank'), 'other_row', 'EIRE is #4; #3 is Sweden')} market with {C('£2,392.28', v(A, 'EIRE', 'Revenue_TW'))}, "
    f"{W('–26.5%', v(A, 'EIRE', 'WoW_Pct'), 'period', 'YoY value cited as WoW', sign=-1)} WoW.",
    f"Finland posted a {W('39.2%', v(A, 'Finland', 'YoY_Pct'), 'direction', 'Finland rose 39.2% YoY', sign=-1)} year-on-year decline.",
    f"Orders fell {C('14.6%', pctc(tot(A, 'Orders_TW'), tot(A, 'Orders_LW')), sign=-1)} WoW to {W('404', tot(A, 'Orders_TW'), 'drift', 'true 440')}, "
    f"with units down {W('7.6%', pctc(tot(A, 'Units_TW'), tot(A, 'Units_LW')), 'drift', 'true -6.6%', sign=-1)}.",
])

LFOS = "LANDMARK FRAME OXFORD STREET"
note("B4_error_heavy", TB, B, [
    f"Total revenue came to {W('GBP 53,823', b_tw, 'drift', 'true 58,323')}, {C('+150.8%', pctc(b_tw, b_lw), sign=1)} WoW.",
    f"LANDMARK FRAME OXFORD STREET generated {W('£3,735.90', v(B, LFOS, 'Revenue_TW'), 'drift', 'transposed; true 3,753.90')} from "
    f"{C('339', v(B, LFOS, 'Units_TW'))} units.",
    f"JUMBO BAG RED RETROSPOT revenue was up {C('7.2%', v(B, JB, 'WoW_Pct'), sign=1)} WoW but units were "
    f"{W('10.3%', upct(B, JB), 'direction', 'units rose 10.3%', sign=-1)} lower.",
    f"CHOCOLATE HOT WATER BOTTLE orders rose {W('27.3%', upct(B, 'CHOCOLATE HOT WATER BOTTLE', 'Orders'), 'drift', 'true +37.5%', sign=1)} to "
    f"{C('33', v(B, 'CHOCOLATE HOT WATER BOTTLE', 'Orders_TW'))}.",
    f"DOORMAT KEEP CALM AND COME IN: {C('£1,356.79', v(B, 'DOORMAT KEEP CALM AND COME IN', 'Revenue_TW'))} TW, "
    f"{W('–43.4%', v(B, 'DOORMAT KEEP CALM AND COME IN', 'WoW_Pct'), 'drift', 'true -34.4%', sign=-1)} WoW.",
    f"PARTY BUNTING revenue was {W('58.5%', v(B, 'PARTY BUNTING', 'WoW_Pct'), 'period', '58.5 is YoY; WoW is +70.6', sign=1)} higher week-on-week at "
    f"{C('£1,183.85', v(B, 'PARTY BUNTING', 'Revenue_TW'))}.",
    f"ZINC FOLKART SLEIGH BELLS revenue grew {C('88.1%', v(B, 'ZINC FOLKART SLEIGH BELLS', 'WoW_Pct'), sign=1)} WoW, while POPCORN HOLDER units grew "
    f"{W('37.1%', upct(B, 'POPCORN HOLDER'), 'drift', 'true +31.7%', sign=1)}.",
    f"WOODEN ADVENT CALENDAR CREAM revenue was down {W('34.8%', v(B, 'WOODEN ADVENT CALENDAR CREAM', 'YoY_Pct'), 'direction', 'YoY is +34.8', sign=-1)} YoY at "
    f"{C('£1,029.22', v(B, 'WOODEN ADVENT CALENDAR CREAM', 'Revenue_TW'))}.",
])

CL10 = "CHRISTMAS LIGHTS 10 REINDEER"
WACR = "WOODEN ADVENT CALENDAR RED"
note("C5_error_heavy", TC, Cq, [
    f"Total revenue was down {W('£25.8k', c_tw - c_lw, 'drift', 'true -23.8k', sign=-1)} WoW, at {C('£135.3k', c_tw)}.",
    f"CHILLI LIGHTS orders rose to {W('35', v(Cq, 'CHILLI LIGHTS', 'Orders_TW'), 'drift', 'true 31')} from {C('27', v(Cq, 'CHILLI LIGHTS', 'Orders_LW'))} LW.",
    f"POPCORN HOLDER revenue fell {W('27.3%', v(Cq, 'POPCORN HOLDER', 'WoW_Pct'), 'drift', 'true -37.3%', sign=-1)} WoW to "
    f"{C('£2,575.75', v(Cq, 'POPCORN HOLDER', 'Revenue_TW'))}.",
    f"HAND WARMER OWL DESIGN revenue fell {C('48.1%', v(Cq, 'HAND WARMER OWL DESIGN', 'WoW_Pct'), sign=-1)} WoW and "
    f"{W('26.1%', v(Cq, 'HAND WARMER OWL DESIGN', 'YoY_Pct'), 'drift', 'true -62.1% YoY', sign=-1)} YoY.",
    f"SPOTTY BUNTING revenue jumped {W('148.0%', v(Cq, 'SPOTTY BUNTING', 'WoW_Pct'), 'drift', 'true +184.0%', sign=1)} WoW to "
    f"{C('£2,080.54', v(Cq, 'SPOTTY BUNTING', 'Revenue_TW'))}.",
    f"LUNCH BAG RED RETROSPOT units were {W('56.9%', upct(Cq, 'LUNCH BAG RED RETROSPOT'), 'direction', 'units fell 56.9%', sign=1)} higher WoW.",
    f"CHRISTMAS LIGHTS {N('10')} REINDEER: {C('£624.19', v(Cq, CL10, 'Revenue_TW'))} TW, {W('–45.6%', v(Cq, CL10, 'WoW_Pct'), 'drift', 'true -54.6%', sign=-1)} WoW, "
    f"{C('–39.3%', v(Cq, CL10, 'YoY_Pct'), sign=-1)} YoY.",
    f"WOODEN ADVENT CALENDAR RED orders fell to {W('17', v(Cq, WACR, 'Orders_TW'), 'drift', 'true 7')} from {C('23', v(Cq, WACR, 'Orders_LW'))}; revenue "
    f"{C('–5.8%', v(Cq, WACR, 'WoW_Pct'), sign=-1)} WoW.",
])


# ------------------------------------------------------------------ build, verify, run


def build(n):
    """Replace markers with text; record each item's (line_idx, start, end) in the gate's line coordinates."""
    lines_out = []
    for li, raw in enumerate(n["lines"]):
        line = "- "
        for piece in re.split(r"(\x00\d+\x00)", raw):
            mm = re.fullmatch(r"\x00(\d+)\x00", piece)
            if mm:
                it = ITEMS[int(mm.group(1))]
                it.update(note_id=n["note_id"], table_id=n["table_id"], line_idx=li,
                          start=len(line), end=len(line) + len(it["text"]))
                line += it["text"]
            else:
                line += piece
        lines_out.append(line)
    for it in ITEMS:
        if it.get("note_id") == n["note_id"]:
            it["line"] = lines_out[it["line_idx"]]
    return "\n".join(lines_out)


def outcome(it, recs):
    t = it["truth"]
    if not recs:
        label = "NotExtracted"
    else:
        label = recs[0]["label"]
    supported = label in ("Supported_cell", "Supported_derived")
    if t == "C":
        return label, ("ok" if supported else "FALSE_ALARM" if label == "Unsupported"
                       else "C_SKIPPED" if label == "Skipped" else "C_NOT_EXTRACTED")
    if t == "W":
        return label, ("caught" if label == "Unsupported" else "MISS" if supported
                       else "W_SKIPPED" if label == "Skipped" else "W_NOT_EXTRACTED")
    if t == "N":
        return label, ("ok" if label in ("Skipped", "NotExtracted") else
                       "N_FLAGGED" if label == "Unsupported" else "N_checked_supported")
    if t.startswith("CNT"):
        return label, f"count_{'correct' if t == 'CNT_C' else 'wrong'}_{label}"
    return label, f"word_{'correct' if t == 'WORD_C' else 'wrong'}_{label}"


def main(suffix=""):
    rows, extras = [], []
    for n in NOTES:
        text = build(n)
        n["text"] = text
        recs = ng.gate_level1(text, n["df"], note_id=n["note_id"])
        items = [it for it in ITEMS if it.get("note_id") == n["note_id"]]
        used = set()
        for it in items:
            hits = [(i, r) for i, r in enumerate(recs) if r["line_idx"] == it["line_idx"]
                    and r["position"] < it["end"] and r["position"] + len(r["raw_text"]) > it["start"]]
            used |= {i for i, _ in hits}
            label, res = outcome(it, [r for _, r in hits])
            r0 = hits[0][1] if hits else {}
            rows.append({"note_id": n["note_id"], "table_id": n["table_id"], "line_idx": it["line_idx"],
                         "text": it["text"], "truth": it["truth"], "kind": it["kind"], "why": it["why"],
                         "true_value": it["tv"], "gate_label": label, "result": res,
                         "gate_raw": r0.get("raw_text", ""), "gate_value": r0.get("value", ""),
                         "gate_unit": r0.get("unit", ""), "gate_direction": r0.get("direction", ""),
                         "evidence": r0.get("evidence", ""), "n_gate_records": len(hits),
                         "line": it["line"]})
        for i, r in enumerate(recs):
            if i not in used:
                extras.append({"note_id": n["note_id"], **{k: r[k] for k in ("line_idx", "raw_text", "label", "evidence")}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"stress_level1_items{suffix}.csv"), index=False)
    with open(os.path.join(OUT, "stress_level1_notes.txt"), "w") as f:
        for n in NOTES:
            f.write(f"### {n['note_id']}  ({n['table_id']})\n{n['text']}\n\n")
    return df, extras


if __name__ == "__main__":
    for it in ITEMS:
        check_truth(it)
    df, extras = main()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    pd.set_option("display.max_rows", 500)
    print(f"notes={len(NOTES)}  items={len(df)}")
    print(df.groupby(["truth", "result"]).size().to_string())
    print("\nnumbers per note (C+W):")
    print(df[df.truth.isin(["C", "W"])].groupby("note_id").size().to_string())
    print("\nextra gate records not matched to any annotation:", extras)
    cols = ["note_id", "text", "truth", "kind", "gate_label", "gate_unit", "gate_direction", "gate_value", "evidence", "why"]
    for res in ["FALSE_ALARM", "C_SKIPPED", "C_NOT_EXTRACTED", "MISS", "W_SKIPPED", "W_NOT_EXTRACTED", "N_FLAGGED", "N_checked_supported"]:
        sub = df[df.result == res]
        if len(sub):
            print(f"\n==== {res} ({len(sub)})")
            print(sub[cols].to_string())
    print("\n==== count / word items")
    print(df[df.truth.str.startswith(("CNT", "WORD"))][cols].to_string())
    print("\n==== caught wrong items")
    print(df[df.result == "caught"][cols].to_string())
