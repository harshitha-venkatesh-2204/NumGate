"""Generator for the Level 2 held-out set (heldout2/L2).

Writes, into this folder:
  notes.json  - 18 notes: text plus the claim JSON a careful extractor following
                prompts/extract_system.txt would emit (the gate's input)
  truth.csv   - one row per claim: truth label, error mode, true value, source of
                a wrong value, independent half-up rounding check

Written BEFORE the gate was run or its source read. Every truth is asserted
from pathlib import Path
from the table with pandas at the cited precision (half-up); the script stops
on any assertion failure. It refuses to overwrite once FROZEN.json exists.
"""
import csv
import json
import os
import re
import sys
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

ROOT = str(Path(__file__).resolve().parents[3])
OUT = os.path.join(ROOT, "paper", "heldout2", "L2")
if os.path.exists(os.path.join(OUT, "FROZEN.json")) and "--force-dry" not in sys.argv:
    sys.exit("FROZEN.json exists: notes and truth are frozen, refusing to regenerate.")
DRY = "--force-dry" in sys.argv

A = "small_country_2010-W27"
B = "small_country_2010-W33"
C = "small_product_2010-W20"
D = "medium_product_2011-W17"
E = "medium_product_2010-W14"
F = "medium_product_2011-W41"
G = "large_product_2011-W22"
H = "large_product_2010-W40"
TABLES = {t: pd.read_csv(os.path.join(ROOT, "data", "tables", t + ".csv")) for t in [A, B, C, D, E, F, G, H]}

# ---------------------------------------------------------------- references
def cell(ent, col):
    return ("cell", ent, col)

def der(ent, kind):          # aov, ratio, diff, uwow
    return ("der", ent, kind)

def grp(ents, kind, col=None):   # kind: sum (col), share, wow
    return ("grp", tuple(ents), kind, col)

def fam(substr, kind, col=None):  # every row whose Entity contains substr
    return ("fam", substr, kind, col)

def tot(kind, col=None):
    return ("tot", kind, col)

def topk(k, kind, col=None):
    return ("topk", k, kind, col)

def rest(k, kind, col=None):
    return ("rest", k, kind, col)


def K(truth, ent, metric, period, vt, ref, src=None, d="none", q="none", mode="", why=""):
    return dict(truth=truth, entity=ent, metric=metric, period=period, value_text=vt,
                ref=ref, src=src, direction=d, qualifier=q, mode=mode, why=why)

def c(*a, **k):  return K("Correct", *a, **k)
def we(*a, **k): return K("Wrong_entity", *a, **{"mode": "other_row", **k})
def wm(*a, **k): return K("Wrong_metric", *a, **k)
def wv(*a, **k): return K("Wrong_value", *a, **{"mode": "drift", **k})
def wd(*a, **k): return K("Wrong_direction", *a, **{"mode": "direction", **k})
def gc(*a, **k): return K("Group_correct", *a, **k)
def gw(*a, **k): return K("Group_wrong", *a, **{"mode": "group_total", **k})
def uv(ent, metric, period, vt, why, d="none", q="none"):
    return K("Unverifiable", ent, metric, period, vt, None, d=d, q=q, mode="not_in_table", why=why)

UK, IE, DE, FR, NL = "United Kingdom", "EIRE", "Germany", "France", "Netherlands"

NOTES = []
def note(nid, table, style, lines):
    NOTES.append(dict(note_id=nid, table_id=table, style=style, lines=lines))

# =============================================================== the notes
note("n01", A, "terse KPI lines", [
 ("- UK: £118,122.66 TW vs £139,889.90 LW (-15.6% WoW), 78.3% share.", [
   c("UK", "revenue", "TW", "£118,122.66", cell(UK, "Revenue_TW")),
   c("UK", "revenue", "LW", "£139,889.90", cell(UK, "Revenue_LW")),
   c("UK", "revenue", "WoW", "-15.6%", cell(UK, "WoW_Pct")),
   c("UK", "share", "TW", "78.3%", cell(UK, "Share_Pct"))]),
 ("- EIRE: £23,853.27 TW on £2,091.43 LW, +1,040.5% WoW, 8 orders TW.", [
   c("EIRE", "revenue", "TW", "£23,853.27", cell(IE, "Revenue_TW")),
   c("EIRE", "revenue", "LW", "£2,091.43", cell(IE, "Revenue_LW")),
   c("EIRE", "revenue", "WoW", "+1,040.5%", cell(IE, "WoW_Pct")),
   c("EIRE", "orders", "TW", "8", cell(IE, "Orders_TW"))]),
 ("- Germany: £3,891.20 TW (-52.2% WoW) on 12 orders TW.", [
   c("Germany", "revenue", "TW", "£3,891.20", cell(DE, "Revenue_TW")),
   c("Germany", "revenue", "WoW", "-52.2%", cell(DE, "WoW_Pct")),
   wm("Germany", "orders", "TW", "12", cell(DE, "Orders_TW"), src=cell(DE, "Orders_LW"), mode="other_period")]),
 ("- France: £1,816.12 TW, -8.6% WoW, 1.2% share.", [
   c("France", "revenue", "TW", "£1,816.12", cell(FR, "Revenue_TW")),
   wv("France", "revenue", "WoW", "-8.6%", cell(FR, "WoW_Pct"), why="digits of -6.8 transposed"),
   c("France", "share", "TW", "1.2%", cell(FR, "Share_Pct"))]),
 ("- Netherlands: £451.45 TW vs £15,223.88 LW (-97.0% WoW), 9,380 units TW.", [
   c("Netherlands", "revenue", "TW", "£451.45", cell(NL, "Revenue_TW")),
   c("Netherlands", "revenue", "LW", "£15,223.88", cell(NL, "Revenue_LW")),
   c("Netherlands", "revenue", "WoW", "-97.0%", cell(NL, "WoW_Pct")),
   wm("Netherlands", "units", "TW", "9,380", cell(NL, "Units_TW"), src=cell(NL, "Units_LW"), mode="other_period")]),
 ("- Thailand: £1,113.70 TW from zero LW, 542 units TW, rank 5.", [
   c("Thailand", "revenue", "TW", "£1,113.70", cell("Thailand", "Revenue_TW")),
   wv("Thailand", "units", "TW", "542", cell("Thailand", "Units_TW"), why="524 transposed"),
   c("Thailand", "rank", "TW", "5", cell("Thailand", "Rank"))]),
 ("- Belgium and Portugal: -77.6% and -77.0% WoW respectively.", [
   we("Belgium", "revenue", "WoW", "-77.6%", cell("Belgium", "WoW_Pct"), src=cell("Portugal", "WoW_Pct"), why="Belgium/Portugal values swapped"),
   we("Portugal", "revenue", "WoW", "-77.0%", cell("Portugal", "WoW_Pct"), src=cell("Belgium", "WoW_Pct"), why="Belgium/Portugal values swapped")]),
 ("- All 10 countries: £150,839.41 TW.", [
   gc("TOTAL", "revenue", "TW", "£150,839.41", tot("sum", "Revenue_TW"))]),
])

note("n02", A, "narrative with demonyms and hedges", [
 ("- The UK generated £118.2k TW, down 15.6% WoW, and held a 78.3% share of revenue.", [
   wv("UK", "revenue", "TW", "£118.2k", cell(UK, "Revenue_TW"), mode="rounding", why="118.12k rounds to 118.1k"),
   c("UK", "revenue", "WoW", "15.6%", cell(UK, "WoW_Pct"), d="down"),
   c("UK", "share", "TW", "78.3%", cell(UK, "Share_Pct"))]),
 ("- Irish revenue jumped to £23,853.27 TW from £2,019.43 LW, more than 11x the LW level.", [
   c("Irish", "revenue", "TW", "£23,853.27", cell(IE, "Revenue_TW")),
   wv("Irish", "revenue", "LW", "£2,019.43", cell(IE, "Revenue_LW"), why="2,091.43 transposed"),
   c("Irish", "revenue", "WoW", "11x", der(IE, "ratio"), d="up", q="over")]),
 ("- German revenue rose 52.2% WoW to £3,891.20 TW on 4,501 units TW.", [
   wd("German", "revenue", "WoW", "52.2%", cell(DE, "WoW_Pct"), d="up"),
   c("German", "revenue", "TW", "£3,891.20", cell(DE, "Revenue_TW")),
   wm("German", "units", "TW", "4,501", cell(DE, "Units_TW"), src=cell(DE, "Units_LW"), mode="other_period")]),
 ("- Dutch revenue fell 87.3% WoW to £451.45 TW, from £15,223.88 LW.", [
   we("Dutch", "revenue", "WoW", "87.3%", cell(NL, "WoW_Pct"), src=cell("Australia", "WoW_Pct"), d="down"),
   c("Dutch", "revenue", "TW", "£451.45", cell(NL, "Revenue_TW")),
   c("Dutch", "revenue", "LW", "£15,223.88", cell(NL, "Revenue_LW"))]),
 ("- Danish sales were £555.66 TW from 1 order TW, about 0.4% of the total.", [
   c("Danish", "revenue", "TW", "£555.66", cell("Denmark", "Revenue_TW")),
   c("Danish", "orders", "TW", "1", cell("Denmark", "Orders_TW")),
   c("Danish", "share", "TW", "0.4%", cell("Denmark", "Share_Pct"), q="approx")]),
 ("- Australian units dropped to 48 TW from 1,618 LW while revenue fell to £232.10 TW.", [
   c("Australian", "units", "TW", "48", cell("Australia", "Units_TW")),
   c("Australian", "units", "LW", "1,618", cell("Australia", "Units_LW")),
   we("Australian", "revenue", "TW", "£232.10", cell("Australia", "Revenue_TW"), src=cell("Portugal", "Revenue_TW"), why="next row (Portugal)")]),
 ("- The top three countries took 96.7% of revenue TW, led by the UK and EIRE.", [
   gc("GROUP", "share", "TW", "96.7%", topk(3, "share"))]),
 ("- The seven smaller markets together brought in £4,927.28 TW.", [
   gw("GROUP", "revenue", "TW", "£4,927.28", rest(3, "sum", "Revenue_TW"), why="true 4,972.28, digits transposed")]),
])

note("n03", B, "narrative sentences", [
 ("- UK revenue slipped 3.0% WoW to £133,081.75 TW, an 82.0% share of the week.", [
   c("UK", "revenue", "WoW", "3.0%", cell(UK, "WoW_Pct"), d="down"),
   c("UK", "revenue", "TW", "£133,081.75", cell(UK, "Revenue_TW")),
   c("UK", "share", "TW", "82.0%", cell(UK, "Share_Pct"))]),
 ("- UK units rose to 84,187 TW from 79,042 LW, while orders eased to 304 TW, about 5% below LY.", [
   c("UK", "units", "TW", "84,187", cell(UK, "Units_TW")),
   c("UK", "units", "LW", "79,042", cell(UK, "Units_LW")),
   c("UK", "orders", "TW", "304", cell(UK, "Orders_TW")),
   uv("UK", "orders", "YoY", "5%", "orders LY not in table (and no LY for 2010)", d="down", q="approx")]),
 ("- EIRE revenue grew 395.9% WoW to £16,455.16 TW across 5 orders TW.", [
   c("EIRE", "revenue", "WoW", "395.9%", cell(IE, "WoW_Pct"), d="up"),
   c("EIRE", "revenue", "TW", "£16,455.16", cell(IE, "Revenue_TW")),
   wm("EIRE", "orders", "TW", "5", cell(IE, "Orders_TW"), src=cell(IE, "Orders_LW"), mode="other_period")]),
 ("- Netherlands revenue fell 64.8% WoW to £6,062.68 TW, with units down to 4,244 TW from 12,076 LW.", [
   wm("Netherlands", "revenue", "WoW", "64.8%", cell(NL, "WoW_Pct"), src=der(NL, "uwow"), d="down", mode="other_metric", why="units change reported as revenue change"),
   c("Netherlands", "revenue", "TW", "£6,062.68", cell(NL, "Revenue_TW")),
   c("Netherlands", "units", "TW", "4,244", cell(NL, "Units_TW")),
   wv("Netherlands", "units", "LW", "12,076", cell(NL, "Units_LW"), why="12,067 transposed")]),
 ("- German revenue rose 596.0% WoW to £2,873.23 TW.", [
   c("German", "revenue", "WoW", "596.0%", cell(DE, "WoW_Pct"), d="up"),
   wv("German", "revenue", "TW", "£2,873.23", cell(DE, "Revenue_TW"), why="2,837.23 transposed")]),
 ("- The UAE contributed £2,047.28 TW from a single order, 1.7% of revenue.", [
   c("UAE", "revenue", "TW", "£2,047.28", cell("United Arab Emirates", "Revenue_TW")),
   we("UAE", "share", "TW", "1.7%", cell("United Arab Emirates", "Share_Pct"), src=cell(DE, "Share_Pct"))]),
 ("- Swedish revenue rose 309.4% WoW to £1,858.91 TW from £407.67 LW.", [
   c("Swedish", "revenue", "WoW", "309.4%", cell("Sweden", "WoW_Pct"), d="up"),
   c("Swedish", "revenue", "TW", "£1,858.91", cell("Sweden", "Revenue_TW")),
   we("Swedish", "revenue", "LW", "£407.67", cell("Sweden", "Revenue_LW"), src=cell(DE, "Revenue_LW"))]),
 ("- All six countries brought in £162,343.01 TW, up 3.1% WoW.", [
   gc("TOTAL", "revenue", "TW", "£162,343.01", tot("sum", "Revenue_TW")),
   gw("TOTAL", "revenue", "WoW", "3.1%", tot("wow"), d="up", why="true +2.1%")]),
])

note("n04", B, "terse KPI with k suffixes", [
 ("- UK £133.1k TW (£137.2k LW), -2.9% WoW, 1st by revenue.", [
   c("UK", "revenue", "TW", "£133.1k", cell(UK, "Revenue_TW")),
   c("UK", "revenue", "LW", "£137.2k", cell(UK, "Revenue_LW")),
   wv("UK", "revenue", "WoW", "-2.9%", cell(UK, "WoW_Pct"), mode="rounding", why="-2.994 truncated instead of rounded to -3.0"),
   c("UK", "rank", "TW", "1st", cell(UK, "Rank"))]),
 ("- EIRE £16.4k TW vs £3.2k LW, 10.1% share TW.", [
   wv("EIRE", "revenue", "TW", "£16.4k", cell(IE, "Revenue_TW"), mode="rounding", why="16.455k rounds to 16.5k"),
   wv("EIRE", "revenue", "LW", "£3.2k", cell(IE, "Revenue_LW"), why="true 3.3k"),
   c("EIRE", "share", "TW", "10.1%", cell(IE, "Share_Pct"))]),
 ("- Netherlands £6.1k TW, +65.5% WoW, 2 orders TW vs 4 LW.", [
   c("Netherlands", "revenue", "TW", "£6.1k", cell(NL, "Revenue_TW")),
   wd("Netherlands", "revenue", "WoW", "+65.5%", cell(NL, "WoW_Pct")),
   c("Netherlands", "orders", "TW", "2", cell(NL, "Orders_TW")),
   c("Netherlands", "orders", "LW", "4", cell(NL, "Orders_LW"))]),
 ("- Germany: 1,989 units TW vs 167 LW, revenue +595.0% WoW.", [
   c("Germany", "units", "TW", "1,989", cell(DE, "Units_TW")),
   c("Germany", "units", "LW", "167", cell(DE, "Units_LW")),
   wv("Germany", "revenue", "WoW", "+595.0%", cell(DE, "WoW_Pct"), mode="rounding", why="595.96 truncated")]),
 ("- Sweden £1.9k TW on 1,364 units TW, 4.1x LW revenue.", [
   c("Sweden", "revenue", "TW", "£1.9k", cell("Sweden", "Revenue_TW")),
   we("Sweden", "units", "TW", "1,364", cell("Sweden", "Units_TW"), src=cell("United Arab Emirates", "Units_TW")),
   c("Sweden", "revenue", "WoW", "4.1x", der("Sweden", "ratio"))]),
 ("- UAE £2.0k TW, 1.3% share, units up 18% YoY.", [
   c("UAE", "revenue", "TW", "£2.0k", cell("United Arab Emirates", "Revenue_TW")),
   c("UAE", "share", "TW", "1.3%", cell("United Arab Emirates", "Share_Pct")),
   uv("UAE", "units", "YoY", "18%", "units LY not in table", d="up")]),
 ("- UK and EIRE combined: 92.1% of revenue TW.", [
   gc("GROUP", "share", "TW", "92.1%", grp([UK, IE], "share"))]),
 ("- Netherlands, Germany, UAE and Sweden together: £12.8k TW.", [
   gc("GROUP", "revenue", "TW", "£12.8k", grp([NL, DE, "United Arab Emirates", "Sweden"], "sum", "Revenue_TW"))]),
])

WHH = "WHITE HANGING HEART T-LIGHT HOLDER"
SUKI, HEARTS, PSPOT = "PACK OF 12 SUKI TISSUES", "PACK OF 12 HEARTS DESIGN TISSUES", "PACK OF 12 PINK SPOT TISSUES"
SKIT, CROQ = "WOODEN SKITTLES GARDEN SET", "WOODEN CROQUET GARDEN SET"
PB = "PARTY BUNTING"
note("n05", C, "narrative, product names with digits, family totals", [
 ("- White Hanging Heart T-Light Holder led with £3,388.78 TW, up 14.4% WoW, a 20.2% share.", [
   c("White Hanging Heart T-Light Holder", "revenue", "TW", "£3,388.78", cell(WHH, "Revenue_TW")),
   c("White Hanging Heart T-Light Holder", "revenue", "WoW", "14.4%", cell(WHH, "WoW_Pct"), d="up"),
   c("White Hanging Heart T-Light Holder", "share", "TW", "20.2%", cell(WHH, "Share_Pct"))]),
 ("- Wooden Skittles Garden Set revenue rose 591.9% WoW to £2,650.90 TW on 230 units TW.", [
   we("Wooden Skittles Garden Set", "revenue", "WoW", "591.9%", cell(SKIT, "WoW_Pct"), src=cell(CROQ, "WoW_Pct"), d="up", why="similar name: Croquet set"),
   c("Wooden Skittles Garden Set", "revenue", "TW", "£2,650.90", cell(SKIT, "Revenue_TW")),
   c("Wooden Skittles Garden Set", "units", "TW", "230", cell(SKIT, "Units_TW"))]),
 ("- Pack of 12 Suki Tissues sold 5,007 units TW, up from 147 LW.", [
   we("Pack of 12 Suki Tissues", "units", "TW", "5,007", cell(SUKI, "Units_TW"), src=cell(HEARTS, "Units_TW"), why="similar name: Hearts Design tissues"),
   c("Pack of 12 Suki Tissues", "units", "LW", "147", cell(SUKI, "Units_LW"))]),
 ("- Pack of 12 Hearts Design Tissues revenue was £2,504.11 TW vs £42.63 LW, up 6,087.6% WoW.", [
   c("Pack of 12 Hearts Design Tissues", "revenue", "TW", "£2,504.11", cell(HEARTS, "Revenue_TW")),
   we("Pack of 12 Hearts Design Tissues", "revenue", "LW", "£42.63", cell(HEARTS, "Revenue_LW"), src=cell(SUKI, "Revenue_LW"), why="similar name: Suki tissues"),
   c("Pack of 12 Hearts Design Tissues", "revenue", "WoW", "6,087.6%", cell(HEARTS, "WoW_Pct"), d="up")]),
 ("- Wooden Croquet Garden Set took £2,232.33 TW from 25 orders TW, ranking 5th.", [
   c("Wooden Croquet Garden Set", "revenue", "TW", "£2,232.33", cell(CROQ, "Revenue_TW")),
   c("Wooden Croquet Garden Set", "orders", "TW", "25", cell(CROQ, "Orders_TW")),
   c("Wooden Croquet Garden Set", "rank", "TW", "5th", cell(CROQ, "Rank"))]),
 ("- Party Bunting rose 12.4% WoW (£233.64) to £2,110.88 TW across 36 orders TW.", [
   c("Party Bunting", "revenue", "WoW", "12.4%", cell(PB, "WoW_Pct"), d="up"),
   c("Party Bunting", "revenue", "WoW", "£233.64", der(PB, "diff"), d="up"),
   c("Party Bunting", "revenue", "TW", "£2,110.88", cell(PB, "Revenue_TW")),
   wm("Party Bunting", "orders", "TW", "36", cell(PB, "Orders_TW"), src=cell(PB, "Orders_LW"), mode="other_period")]),
 ("- The three Pack of 12 tissue lines combined took £6,357.34 TW, 39.0% of revenue.", [
   gc("GROUP", "revenue", "TW", "£6,357.34", fam("TISSUES", "sum", "Revenue_TW")),
   gw("GROUP", "share", "TW", "39.0%", fam("TISSUES", "share"), why="true 38.0%")]),
 ("- Pack of 12 Pink Spot Tissues revenue grew 2,942.9% WoW to £1,297.52 TW.", [
   c("Pack of 12 Pink Spot Tissues", "revenue", "WoW", "2,942.9%", cell(PSPOT, "WoW_Pct"), d="up"),
   wv("Pack of 12 Pink Spot Tissues", "revenue", "TW", "£1,297.52", cell(PSPOT, "Revenue_TW"), why="1,279.52 transposed")]),
])

note("n06", C, "terse KPI, averages per order, hedges", [
 ("- WHITE HANGING HEART T-LIGHT HOLDER: £3,388.78 TW, 1,105 units, 62 orders, about £54.66 per order.", [
   c("WHITE HANGING HEART T-LIGHT HOLDER", "revenue", "TW", "£3,388.78", cell(WHH, "Revenue_TW")),
   wm("WHITE HANGING HEART T-LIGHT HOLDER", "units", "TW", "1,105", cell(WHH, "Units_TW"), src=cell(WHH, "Units_LW"), mode="other_period"),
   c("WHITE HANGING HEART T-LIGHT HOLDER", "orders", "TW", "62", cell(WHH, "Orders_TW")),
   c("WHITE HANGING HEART T-LIGHT HOLDER", "revenue", "TW", "£54.66", der(WHH, "aov"), q="approx", why="average per order")]),
 ("- Wooden Skittles Garden Set: £2,650.90 TW vs £468.85 LW, 15.4% share.", [
   c("Wooden Skittles Garden Set", "revenue", "TW", "£2,650.90", cell(SKIT, "Revenue_TW")),
   c("Wooden Skittles Garden Set", "revenue", "LW", "£468.85", cell(SKIT, "Revenue_LW")),
   we("Wooden Skittles Garden Set", "share", "TW", "15.4%", cell(SKIT, "Share_Pct"), src=cell(SUKI, "Share_Pct"), why="next row")]),
 ("- Suki Tissues: £2,573.71 TW, more than 60x LW revenue of £42.63.", [
   c("Suki Tissues", "revenue", "TW", "£2,573.71", cell(SUKI, "Revenue_TW")),
   c("Suki Tissues", "revenue", "WoW", "60x", der(SUKI, "ratio"), q="over"),
   c("Suki Tissues", "revenue", "LW", "£42.63", cell(SUKI, "Revenue_LW"))]),
 ("- Hearts Design Tissues: 13 orders TW vs 6 LW, averaging £500.82 per order TW.", [
   we("Hearts Design Tissues", "orders", "TW", "13", cell(HEARTS, "Orders_TW"), src=cell(SUKI, "Orders_TW"), why="similar name: Suki tissues"),
   c("Hearts Design Tissues", "orders", "LW", "6", cell(HEARTS, "Orders_LW")),
   c("Hearts Design Tissues", "revenue", "TW", "£500.82", der(HEARTS, "aov"), why="average per order")]),
 ("- Wooden Croquet Garden Set: £2,322.33 TW, +591.9% WoW, 13.3% share.", [
   wv("Wooden Croquet Garden Set", "revenue", "TW", "£2,322.33", cell(CROQ, "Revenue_TW"), why="2,232.33 transposed"),
   c("Wooden Croquet Garden Set", "revenue", "WoW", "+591.9%", cell(CROQ, "WoW_Pct")),
   c("Wooden Croquet Garden Set", "share", "TW", "13.3%", cell(CROQ, "Share_Pct"))]),
 ("- Party Bunting: £2,110.88 TW, -12.4% WoW, 442 units TW.", [
   c("Party Bunting", "revenue", "TW", "£2,110.88", cell(PB, "Revenue_TW")),
   wd("Party Bunting", "revenue", "WoW", "-12.4%", cell(PB, "WoW_Pct")),
   wm("Party Bunting", "units", "TW", "442", cell(PB, "Units_TW"), src=cell(PB, "Units_LW"), mode="other_period")]),
 ("- Pink Spot Tissues: £1,279.52 TW, 7.6% share, units up 1,693.1% WoW.", [
   c("Pink Spot Tissues", "revenue", "TW", "£1,279.52", cell(PSPOT, "Revenue_TW")),
   c("Pink Spot Tissues", "share", "TW", "7.6%", cell(PSPOT, "Share_Pct")),
   c("Pink Spot Tissues", "units", "WoW", "1,693.1%", der(PSPOT, "uwow"), d="up", why="units change, computable from Units_TW/LW")]),
 ("- Top five lines: 79.7% of revenue TW.", [
   gc("GROUP", "share", "TW", "79.7%", topk(5, "share"))]),
])

RCS = "REGENCY CAKESTAND 3 TIER"
note("n07", D, "narrative with YoY", [
 ("- Regency Cakestand 3 Tier led with £3,804.27 TW, up 52.1% WoW and 6.3% YoY, an 18.6% share.", [
   c("Regency Cakestand 3 Tier", "revenue", "TW", "£3,804.27", cell(RCS, "Revenue_TW")),
   c("Regency Cakestand 3 Tier", "revenue", "WoW", "52.1%", cell(RCS, "WoW_Pct"), d="up"),
   c("Regency Cakestand 3 Tier", "revenue", "YoY", "6.3%", cell(RCS, "YoY_Pct"), d="up"),
   c("Regency Cakestand 3 Tier", "share", "TW", "18.6%", cell(RCS, "Share_Pct"))]),
 ("- Party Bunting fell 47.2% WoW to £1,840.12 TW, and was up 13.0% YoY.", [
   c("Party Bunting", "revenue", "WoW", "47.2%", cell(PB, "WoW_Pct"), d="down"),
   c("Party Bunting", "revenue", "TW", "£1,840.12", cell(PB, "Revenue_TW")),
   wd("Party Bunting", "revenue", "YoY", "13.0%", cell(PB, "YoY_Pct"), d="up")]),
 ("- Paper Chain Kit Empire rose 16.3% WoW to £1,266.57 TW, up 279.4% YoY from £333.81 LY, with units up 40% YoY.", [
   c("Paper Chain Kit Empire", "revenue", "WoW", "16.3%", cell("PAPER CHAIN KIT EMPIRE", "WoW_Pct"), d="up"),
   c("Paper Chain Kit Empire", "revenue", "TW", "£1,266.57", cell("PAPER CHAIN KIT EMPIRE", "Revenue_TW")),
   c("Paper Chain Kit Empire", "revenue", "YoY", "279.4%", cell("PAPER CHAIN KIT EMPIRE", "YoY_Pct"), d="up"),
   c("Paper Chain Kit Empire", "revenue", "LY", "£333.81", cell("PAPER CHAIN KIT EMPIRE", "Revenue_LY")),
   uv("Paper Chain Kit Empire", "units", "YoY", "40%", "units LY not in table", d="up")]),
 ("- White Hanging Heart T-Light Holder dropped 83.1% WoW to £1,028.01 TW from £3,074.72 LW.", [
   c("White Hanging Heart T-Light Holder", "revenue", "WoW", "83.1%", cell(WHH, "WoW_Pct"), d="down"),
   c("White Hanging Heart T-Light Holder", "revenue", "TW", "£1,028.01", cell(WHH, "Revenue_TW")),
   wm("White Hanging Heart T-Light Holder", "revenue", "LW", "£3,074.72", cell(WHH, "Revenue_LW"), src=cell(WHH, "Revenue_LY"), mode="other_period")]),
 ("- Union Stripe with Fringe Hammock grew 729.0% WoW to £724.97 TW.", [
   wm("Union Stripe with Fringe Hammock", "revenue", "WoW", "729.0%", cell("UNION STRIPE", "WoW_Pct"), src=cell("UNION STRIPE", "YoY_Pct"), d="up", mode="other_period", why="YoY value given as WoW"),
   c("Union Stripe with Fringe Hammock", "revenue", "TW", "£724.97", cell("UNION STRIPE", "Revenue_TW"))]),
 ("- World War 2 Gliders sold 2,592 units TW, with revenue up 29.7% WoW to £536.52 TW.", [
   c("World War 2 Gliders", "units", "TW", "2,592", cell("WORLD WAR 2 GLIDERS", "Units_TW")),
   c("World War 2 Gliders", "revenue", "WoW", "29.7%", cell("WORLD WAR 2 GLIDERS", "WoW_Pct"), d="up"),
   wv("World War 2 Gliders", "revenue", "TW", "£536.52", cell("WORLD WAR 2 GLIDERS", "Revenue_TW"), why="563.52 transposed")]),
 ("- Balloon Water Bomb Pack of 35 revenue rose 1,373.1% WoW and 583.6% YoY to £464.03 TW.", [
   c("Balloon Water Bomb Pack of 35", "revenue", "WoW", "1,373.1%", cell("BALLOON WATER BOMB", "WoW_Pct"), d="up"),
   wv("Balloon Water Bomb Pack of 35", "revenue", "YoY", "583.6%", cell("BALLOON WATER BOMB", "YoY_Pct"), d="up", why="538.6 transposed"),
   c("Balloon Water Bomb Pack of 35", "revenue", "TW", "£464.03", cell("BALLOON WATER BOMB", "Revenue_TW"))]),
 ("- Assorted Colour Bird Ornament rose 53.8% WoW to £440.72 TW.", [
   wd("Assorted Colour Bird Ornament", "revenue", "WoW", "53.8%", cell("ASSORTED COLOUR BIRD ORNAMENT", "WoW_Pct"), d="up"),
   c("Assorted Colour Bird Ornament", "revenue", "TW", "£440.72", cell("ASSORTED COLOUR BIRD ORNAMENT", "Revenue_TW"))]),
])

JRR, JAP, JPP = "JUMBO BAG RED RETROSPOT", "JUMBO BAG APPLES", "JUMBO BAG PINK POLKADOT"
note("n08", D, "terse KPI, family and top-k shares", [
 ("- Cakestand (3 tier): £3.8k TW, 325 units, 25 orders TW vs 42 LW.", [
   c("Cakestand (3 tier)", "revenue", "TW", "£3.8k", cell(RCS, "Revenue_TW")),
   c("Cakestand (3 tier)", "units", "TW", "325", cell(RCS, "Units_TW")),
   c("Cakestand (3 tier)", "orders", "TW", "25", cell(RCS, "Orders_TW")),
   c("Cakestand (3 tier)", "orders", "LW", "42", cell(RCS, "Orders_LW"))]),
 ("- Party Bunting: 276 units TW vs 684 LW; units down 12% YoY.", [
   c("Party Bunting", "units", "TW", "276", cell(PB, "Units_TW")),
   wv("Party Bunting", "units", "LW", "684", cell(PB, "Units_LW"), why="648 transposed"),
   uv("Party Bunting", "units", "YoY", "12%", "units LY not in table", d="down")]),
 ("- Ceramic Strawberry Money Box: £973.43 TW vs £8.26 LW, 4.6% share.", [
   wv("Ceramic Strawberry Money Box", "revenue", "TW", "£973.43", cell("CERAMIC STRAWBERRY MONEY BOX", "Revenue_TW"), why="937.43 transposed"),
   c("Ceramic Strawberry Money Box", "revenue", "LW", "£8.26", cell("CERAMIC STRAWBERRY MONEY BOX", "Revenue_LW")),
   c("Ceramic Strawberry Money Box", "share", "TW", "4.6%", cell("CERAMIC STRAWBERRY MONEY BOX", "Share_Pct"))]),
 ("- Jumbo Bag Red Retrospot: £817.65 TW, +27.8% WoW; Jumbo Bag Pink Polkadot: £611.47 TW, +69.8% WoW.", [
   c("Jumbo Bag Red Retrospot", "revenue", "TW", "£817.65", cell(JRR, "Revenue_TW")),
   wd("Jumbo Bag Red Retrospot", "revenue", "WoW", "+27.8%", cell(JRR, "WoW_Pct")),
   we("Jumbo Bag Pink Polkadot", "revenue", "TW", "£611.47", cell(JPP, "Revenue_TW"), src=cell(JAP, "Revenue_TW"), why="similar name: Jumbo Bag Apples"),
   c("Jumbo Bag Pink Polkadot", "revenue", "WoW", "+69.8%", cell(JPP, "WoW_Pct"))]),
 ("- The five Jumbo Bag lines combined: £3,004.97 TW.", [
   gc("GROUP", "revenue", "TW", "£3,004.97", fam("JUMBO BAG", "sum", "Revenue_TW"))]),
 ("- Lunch Bag Woodland: £421.66 TW, +110.5% WoW, -3.2% YoY.", [
   wm("Lunch Bag Woodland", "revenue", "TW", "£421.66", cell("LUNCH BAG WOODLAND", "Revenue_TW"), src=cell("LUNCH BAG WOODLAND", "Revenue_LY"), mode="other_period"),
   c("Lunch Bag Woodland", "revenue", "WoW", "+110.5%", cell("LUNCH BAG WOODLAND", "WoW_Pct")),
   c("Lunch Bag Woodland", "revenue", "YoY", "-3.2%", cell("LUNCH BAG WOODLAND", "YoY_Pct"))]),
 ("- Top three lines: 33.8% of TW revenue; top ten: 63.1%.", [
   gc("GROUP", "share", "TW", "33.8%", topk(3, "share")),
   gc("GROUP", "share", "TW", "63.1%", topk(10, "share"), why="sum of rounded Share_Pct is 63.0; exact share 63.07")]),
 ("- Vintage Red Kitchen Cabinet: £500.00 TW from 4 units TW, 2.4% share.", [
   c("Vintage Red Kitchen Cabinet", "revenue", "TW", "£500.00", cell("VINTAGE RED KITCHEN CABINET", "Revenue_TW")),
   c("Vintage Red Kitchen Cabinet", "units", "TW", "4", cell("VINTAGE RED KITCHEN CABINET", "Units_TW")),
   c("Vintage Red Kitchen Cabinet", "share", "TW", "2.4%", cell("VINTAGE RED KITCHEN CABINET", "Share_Pct"))]),
])

ERQ = "ENGLISH ROSE DESIGN QUILTED THROW"
HBB, HBP = "BLUE HAPPY BIRTHDAY BUNTING", "PINK HAPPY BIRTHDAY BUNTING"
MCM, MVC = "MAGNETS PACK OF 4 CHILDHOOD MEMORY", "MAGNETS PACK OF 4 VINTAGE COLLAGE"
note("n09", E, "narrative, parenthetical colour variants", [
 ("- English Rose Design Quilted Throw led with £4,577.17 TW, up 590.6% WoW from £662.74 LW, a 10.9% share.", [
   c("English Rose Design Quilted Throw", "revenue", "TW", "£4,577.17", cell(ERQ, "Revenue_TW")),
   c("English Rose Design Quilted Throw", "revenue", "WoW", "590.6%", cell(ERQ, "WoW_Pct"), d="up"),
   c("English Rose Design Quilted Throw", "revenue", "LW", "£662.74", cell(ERQ, "Revenue_LW")),
   we("English Rose Design Quilted Throw", "share", "TW", "10.9%", cell(ERQ, "Share_Pct"), src=cell(RCS, "Share_Pct"), why="next row")]),
 ("- Regency Cakestand 3 Tier rose 9.9% WoW to £2,799.04 TW on 44 orders TW.", [
   wd("Regency Cakestand 3 Tier", "revenue", "WoW", "9.9%", cell(RCS, "WoW_Pct"), d="up"),
   c("Regency Cakestand 3 Tier", "revenue", "TW", "£2,799.04", cell(RCS, "Revenue_TW")),
   wm("Regency Cakestand 3 Tier", "orders", "TW", "44", cell(RCS, "Orders_TW"), src=cell(RCS, "Orders_LW"), mode="other_period")]),
 ("- White Hanging Heart T-Light Holder fell 20.3% WoW to £2,420.03 TW, though units edged up to 909 TW from 900 LW.", [
   c("White Hanging Heart T-Light Holder", "revenue", "WoW", "20.3%", cell(WHH, "WoW_Pct"), d="down"),
   c("White Hanging Heart T-Light Holder", "revenue", "TW", "£2,420.03", cell(WHH, "Revenue_TW")),
   c("White Hanging Heart T-Light Holder", "units", "TW", "909", cell(WHH, "Units_TW")),
   c("White Hanging Heart T-Light Holder", "units", "LW", "900", cell(WHH, "Units_LW"))]),
 ("- Red Retrospot Cake Stand revenue rose 1,269.8% WoW to £1,499.39 TW.", [
   c("Red Retrospot Cake Stand", "revenue", "WoW", "1,269.8%", cell("RED RETROSPOT CAKE STAND", "WoW_Pct"), d="up"),
   wv("Red Retrospot Cake Stand", "revenue", "TW", "£1,499.39", cell("RED RETROSPOT CAKE STAND", "Revenue_TW"), why="pence transposed (1,499.93)")]),
 ("- Edwardian Parasol (Natural) took £1,458.21 TW on 282 units TW, up 1,051.9% WoW.", [
   c("Edwardian Parasol (Natural)", "revenue", "TW", "£1,458.21", cell("EDWARDIAN PARASOL NATURAL", "Revenue_TW")),
   c("Edwardian Parasol (Natural)", "units", "TW", "282", cell("EDWARDIAN PARASOL NATURAL", "Units_TW")),
   c("Edwardian Parasol (Natural)", "revenue", "WoW", "1,051.9%", cell("EDWARDIAN PARASOL NATURAL", "WoW_Pct"), d="up")]),
 ("- Happy Birthday Bunting (Blue) fell 61.5% WoW to £771.52 TW.", [
   we("Happy Birthday Bunting (Blue)", "revenue", "WoW", "61.5%", cell(HBB, "WoW_Pct"), src=cell(HBP, "WoW_Pct"), d="down", why="similar name: Pink variant"),
   c("Happy Birthday Bunting (Blue)", "revenue", "TW", "£771.52", cell(HBB, "Revenue_TW"))]),
 ("- Magnets Pack of 4 (Childhood Memory) revenue rose 364.0% WoW to £863.86 TW on 771 units TW.", [
   c("Magnets Pack of 4 (Childhood Memory)", "revenue", "WoW", "364.0%", cell(MCM, "WoW_Pct"), d="up"),
   c("Magnets Pack of 4 (Childhood Memory)", "revenue", "TW", "£863.86", cell(MCM, "Revenue_TW")),
   we("Magnets Pack of 4 (Childhood Memory)", "units", "TW", "771", cell(MCM, "Units_TW"), src=cell(MVC, "Units_TW"), why="similar name: Vintage Collage variant")]),
 ("- The two Happy Birthday Bunting lines together fell to £1,532.02 TW from £3,827.72 LW.", [
   gc("GROUP", "revenue", "TW", "£1,532.02", grp([HBB, HBP], "sum", "Revenue_TW")),
   gw("GROUP", "revenue", "LW", "£3,827.72", grp([HBB, HBP], "sum", "Revenue_LW"), why="true 3,872.72 transposed")]),
])

note("n10", E, "terse KPI, averages per order, rest of table", [
 ("- English Rose Quilted Throw: £4,577.17 TW, 5 orders, around £915 per order.", [
   c("English Rose Quilted Throw", "revenue", "TW", "£4,577.17", cell(ERQ, "Revenue_TW")),
   c("English Rose Quilted Throw", "orders", "TW", "5", cell(ERQ, "Orders_TW")),
   c("English Rose Quilted Throw", "revenue", "TW", "£915", der(ERQ, "aov"), q="approx", why="average per order")]),
 ("- Blue Cushion Cover with Flower: £985.95 TW from 161 orders TW, nothing LW.", [
   wv("Blue Cushion Cover with Flower", "revenue", "TW", "£985.95", cell("BLUE CUSHION COVER", "Revenue_TW"), why="958.95 transposed"),
   wm("Blue Cushion Cover with Flower", "orders", "TW", "161", cell("BLUE CUSHION COVER", "Orders_TW"), src=cell("BLUE CUSHION COVER", "Units_TW"), mode="other_metric", why="units given as orders")]),
 ("- Chilli Lights: £960.90 TW, +46.1% WoW, 216 units TW vs 405 LW.", [
   c("Chilli Lights", "revenue", "TW", "£960.90", cell("CHILLI LIGHTS", "Revenue_TW")),
   wd("Chilli Lights", "revenue", "WoW", "+46.1%", cell("CHILLI LIGHTS", "WoW_Pct")),
   c("Chilli Lights", "units", "TW", "216", cell("CHILLI LIGHTS", "Units_TW")),
   wv("Chilli Lights", "units", "LW", "405", cell("CHILLI LIGHTS", "Units_LW"), why="true 450")]),
 ("- Camouflage Design Teddy: £811.50 TW, +6,340.5% WoW, 3.2% share.", [
   c("Camouflage Design Teddy", "revenue", "TW", "£811.50", cell("CAMOFLAGE DESIGN TEDDY", "Revenue_TW"), why="table spells CAMOFLAGE"),
   c("Camouflage Design Teddy", "revenue", "WoW", "+6,340.5%", cell("CAMOFLAGE DESIGN TEDDY", "WoW_Pct")),
   c("Camouflage Design Teddy", "share", "TW", "3.2%", cell("CAMOFLAGE DESIGN TEDDY", "Share_Pct"))]),
 ("- Party Bunting: £885.31 TW, -29.6% WoW, rank 9.", [
   c("Party Bunting", "revenue", "TW", "£885.31", cell(PB, "Revenue_TW")),
   wv("Party Bunting", "revenue", "WoW", "-29.6%", cell(PB, "WoW_Pct"), why="-26.9 transposed"),
   c("Party Bunting", "rank", "TW", "9", cell(PB, "Rank"))]),
 ("- Door Mat Union Flag: 28 orders TW vs 16 LW, £858.46 TW.", [
   c("Door Mat Union Flag", "orders", "TW", "28", cell("DOOR MAT UNION FLAG", "Orders_TW")),
   c("Door Mat Union Flag", "orders", "LW", "16", cell("DOOR MAT UNION FLAG", "Orders_LW")),
   we("Door Mat Union Flag", "revenue", "TW", "£858.46", cell("DOOR MAT UNION FLAG", "Revenue_TW"), src=cell("VINTAGE UNION JACK BUNTING", "Revenue_TW"), why="next row")]),
 ("- Top five products: more than 49% of revenue TW.", [
   gc("GROUP", "share", "TW", "49%", topk(5, "share"), q="over")]),
 ("- The remaining 15 products shared £12,861.89 TW.", [
   gw("GROUP", "revenue", "TW", "£12,861.89", rest(5, "sum", "Revenue_TW"), why="true 12,816.89 transposed")]),
])

DKC = "DOORMAT KEEP CALM AND COME IN"
note("n11", F, "narrative with YoY, family totals", [
 ("- Regency Cakestand 3 Tier stayed top at £2,729.22 TW, down 44.1% WoW and 36.4% YoY.", [
   c("Regency Cakestand 3 Tier", "revenue", "TW", "£2,729.22", cell(RCS, "Revenue_TW")),
   c("Regency Cakestand 3 Tier", "revenue", "WoW", "44.1%", cell(RCS, "WoW_Pct"), d="down"),
   c("Regency Cakestand 3 Tier", "revenue", "YoY", "36.4%", cell(RCS, "YoY_Pct"), d="down")]),
 ("- Paper Chain Kit 50's Christmas fell 13.5% WoW to £2,432.01 TW but was up 56.4% YoY.", [
   c("Paper Chain Kit 50's Christmas", "revenue", "WoW", "13.5%", cell("PAPER CHAIN KIT 50'S CHRISTMAS", "WoW_Pct"), d="down"),
   wv("Paper Chain Kit 50's Christmas", "revenue", "TW", "£2,432.01", cell("PAPER CHAIN KIT 50'S CHRISTMAS", "Revenue_TW"), why="2,342.01 transposed"),
   c("Paper Chain Kit 50's Christmas", "revenue", "YoY", "56.4%", cell("PAPER CHAIN KIT 50'S CHRISTMAS", "YoY_Pct"), d="up")]),
 ("- Feltcraft Doll Molly revenue of £2,289.28 TW was up 934.6% YoY from £204.16 LY, with units up more than 300% YoY.", [
   c("Feltcraft Doll Molly", "revenue", "TW", "£2,289.28", cell("FELTCRAFT DOLL MOLLY", "Revenue_TW")),
   c("Feltcraft Doll Molly", "revenue", "YoY", "934.6%", cell("FELTCRAFT DOLL MOLLY", "YoY_Pct"), d="up"),
   we("Feltcraft Doll Molly", "revenue", "LY", "£204.16", cell("FELTCRAFT DOLL MOLLY", "Revenue_LY"), src=cell("FELTCRAFT BUTTERFLY HEARTS", "Revenue_LY"), why="Feltcraft family swap"),
   uv("Feltcraft Doll Molly", "units", "YoY", "300%", "units LY not in table", d="up", q="over")]),
 ("- Doormat Keep Calm and Come In dropped 62.3% WoW to £2,068.84 TW from £6,105.84 LW.", [
   we("Doormat Keep Calm and Come In", "revenue", "WoW", "62.3%", cell(DKC, "WoW_Pct"), src=cell(JRR, "WoW_Pct"), d="down"),
   c("Doormat Keep Calm and Come In", "revenue", "TW", "£2,068.84", cell(DKC, "Revenue_TW")),
   c("Doormat Keep Calm and Come In", "revenue", "LW", "£6,105.84", cell(DKC, "Revenue_LW"))]),
 ("- White Hanging Heart T-Light Holder rose 275.8% WoW to £1,893.01 TW, yet was up 31.1% YoY.", [
   c("White Hanging Heart T-Light Holder", "revenue", "WoW", "275.8%", cell(WHH, "WoW_Pct"), d="up"),
   c("White Hanging Heart T-Light Holder", "revenue", "TW", "£1,893.01", cell(WHH, "Revenue_TW")),
   wd("White Hanging Heart T-Light Holder", "revenue", "YoY", "31.1%", cell(WHH, "YoY_Pct"), d="up")]),
 ("- The three hot water bottle lines (Tea and Sympathy, Keep Calm, Love) totalled £2,729.24 TW.", [
   gc("GROUP", "revenue", "TW", "£2,729.24", grp(["HOT WATER BOTTLE TEA AND SYMPATHY", "HOT WATER BOTTLE KEEP CALM", "LOVE HOT WATER BOTTLE"], "sum", "Revenue_TW"))]),
 ("- The three Feltcraft cushions (Owl, Rabbit, Butterfly) together fell to £2,564.52 TW from £3,727.96 LW.", [
   gc("GROUP", "revenue", "TW", "£2,564.52", fam("FELTCRAFT CUSHION", "sum", "Revenue_TW")),
   gw("GROUP", "revenue", "LW", "£3,727.96", fam("FELTCRAFT CUSHION", "sum", "Revenue_LW"), why="true 3,772.96 transposed")]),
 ("- Jam Making Set with Jars grew 60.8% WoW to £1,057.78 TW on 23 orders TW.", [
   c("Jam Making Set with Jars", "revenue", "WoW", "60.8%", cell("JAM MAKING SET WITH JARS", "WoW_Pct"), d="up"),
   c("Jam Making Set with Jars", "revenue", "TW", "£1,057.78", cell("JAM MAKING SET WITH JARS", "Revenue_TW")),
   wm("Jam Making Set with Jars", "orders", "TW", "23", cell("JAM MAKING SET WITH JARS", "Orders_TW"), src=cell("JAM MAKING SET WITH JARS", "Orders_LW"), mode="other_period")]),
])

FCO, FCR = "FELTCRAFT CUSHION OWL", "FELTCRAFT CUSHION RABBIT"
note("n12", F, "terse KPI, similar-name swaps", [
 ("- Paper Chain Kit Vintage Christmas: £1,456.87 TW, -11.9% WoW, -47.5% YoY.", [
   wm("Paper Chain Kit Vintage Christmas", "revenue", "TW", "£1,456.87", cell("PAPER CHAIN KIT VINTAGE CHRISTMAS", "Revenue_TW"), src=cell("PAPER CHAIN KIT VINTAGE CHRISTMAS", "Revenue_LW"), mode="other_period"),
   c("Paper Chain Kit Vintage Christmas", "revenue", "WoW", "-11.9%", cell("PAPER CHAIN KIT VINTAGE CHRISTMAS", "WoW_Pct")),
   c("Paper Chain Kit Vintage Christmas", "revenue", "YoY", "-47.5%", cell("PAPER CHAIN KIT VINTAGE CHRISTMAS", "YoY_Pct"))]),
 ("- Feltcraft Cushion Owl: £888.60 TW, 264 units TW.", [
   we("Feltcraft Cushion Owl", "revenue", "TW", "£888.60", cell(FCO, "Revenue_TW"), src=cell(FCR, "Revenue_TW"), why="similar name: Rabbit cushion"),
   c("Feltcraft Cushion Owl", "units", "TW", "264", cell(FCO, "Units_TW"))]),
 ("- Feltcraft Cushion Rabbit: £888.60 TW, 6 orders TW (15 LW), about £148 per order.", [
   c("Feltcraft Cushion Rabbit", "revenue", "TW", "£888.60", cell(FCR, "Revenue_TW")),
   c("Feltcraft Cushion Rabbit", "orders", "TW", "6", cell(FCR, "Orders_TW")),
   c("Feltcraft Cushion Rabbit", "orders", "LW", "15", cell(FCR, "Orders_LW")),
   c("Feltcraft Cushion Rabbit", "revenue", "TW", "£148", der(FCR, "aov"), q="approx", why="average per order")]),
 ("- Red Woolly Hottie: £1,213.15 TW, +388.8% WoW, +173.7% YoY.", [
   c("Red Woolly Hottie", "revenue", "TW", "£1,213.15", cell("RED WOOLLY HOTTIE", "Revenue_TW")),
   c("Red Woolly Hottie", "revenue", "WoW", "+388.8%", cell("RED WOOLLY HOTTIE", "WoW_Pct")),
   wv("Red Woolly Hottie", "revenue", "YoY", "+173.7%", cell("RED WOOLLY HOTTIE", "YoY_Pct"), why="137.7 transposed")]),
 ("- Rotating Silver Angels T-Light: £1,089.49 TW, 2.5% share; orders down 30% YoY.", [
   c("Rotating Silver Angels T-Light", "revenue", "TW", "£1,089.49", cell("ROTATING SILVER ANGELS", "Revenue_TW")),
   c("Rotating Silver Angels T-Light", "share", "TW", "2.5%", cell("ROTATING SILVER ANGELS", "Share_Pct")),
   uv("Rotating Silver Angels T-Light", "orders", "YoY", "30%", "orders LY not in table", d="down")]),
 ("- Jam Making Set Printed: £701.09 TW vs £702.81 LW, +0.2% WoW.", [
   c("Jam Making Set Printed", "revenue", "TW", "£701.09", cell("JAM MAKING SET PRINTED", "Revenue_TW")),
   c("Jam Making Set Printed", "revenue", "LW", "£702.81", cell("JAM MAKING SET PRINTED", "Revenue_LW")),
   wd("Jam Making Set Printed", "revenue", "WoW", "+0.2%", cell("JAM MAKING SET PRINTED", "WoW_Pct"))]),
 ("- Top ten lines: 42.7% of revenue TW; top five about 26%.", [
   gc("GROUP", "share", "TW", "42.7%", topk(10, "share"), why="sum of rounded Share_Pct is 42.9; exact share 42.71"),
   gc("GROUP", "share", "TW", "26%", topk(5, "share"), q="approx")]),
 ("- Party Bunting: £691.53 TW, 183 units TW, down 54.2% YoY from £1,514.39 LY.", [
   we("Party Bunting", "revenue", "TW", "£691.53", cell(PB, "Revenue_TW"), src=cell("WALL ART STOP FOR TEA", "Revenue_TW"), why="next row"),
   wm("Party Bunting", "units", "TW", "183", cell(PB, "Units_TW"), src=cell(PB, "Units_LW"), mode="other_period"),
   c("Party Bunting", "revenue", "YoY", "54.2%", cell(PB, "YoY_Pct"), d="down"),
   c("Party Bunting", "revenue", "LY", "£1,514.39", cell(PB, "Revenue_LY"))]),
])

HWS, HWL = "HEART OF WICKER SMALL", "HEART OF WICKER LARGE"
note("n13", G, "narrative, similar-name swaps", [
 ("- Regency Cakestand 3 Tier led with £3,041.70 TW, up 43.7% WoW and 137.5% YoY, a 5.4% share.", [
   c("Regency Cakestand 3 Tier", "revenue", "TW", "£3,041.70", cell(RCS, "Revenue_TW")),
   c("Regency Cakestand 3 Tier", "revenue", "WoW", "43.7%", cell(RCS, "WoW_Pct"), d="up"),
   wv("Regency Cakestand 3 Tier", "revenue", "YoY", "137.5%", cell(RCS, "YoY_Pct"), d="up", why="135.7 transposed"),
   c("Regency Cakestand 3 Tier", "share", "TW", "5.4%", cell(RCS, "Share_Pct"))]),
 ("- Party Bunting halved to £2,327.02 TW from £4,652.41 LW, down 50.0% WoW though up 60.0% YoY; its LY share was 7.3%.", [
   c("Party Bunting", "revenue", "TW", "£2,327.02", cell(PB, "Revenue_TW")),
   c("Party Bunting", "revenue", "LW", "£4,652.41", cell(PB, "Revenue_LW")),
   c("Party Bunting", "revenue", "WoW", "50.0%", cell(PB, "WoW_Pct"), d="down"),
   c("Party Bunting", "revenue", "YoY", "60.0%", cell(PB, "YoY_Pct"), d="up"),
   uv("Party Bunting", "share", "LY", "7.3%", "share is only given for TW")]),
 ("- Black Record Cover Frame took £1,708.56 TW with no sales LW, up 764.6% YoY.", [
   c("Black Record Cover Frame", "revenue", "TW", "£1,708.56", cell("BLACK RECORD COVER FRAME", "Revenue_TW")),
   c("Black Record Cover Frame", "revenue", "YoY", "764.6%", cell("BLACK RECORD COVER FRAME", "YoY_Pct"), d="up")]),
 ("- Heart of Wicker (Small) fell 75.1% WoW to £736.41 TW.", [
   we("Heart of Wicker (Small)", "revenue", "WoW", "75.1%", cell(HWS, "WoW_Pct"), src=cell(HWL, "WoW_Pct"), d="down", why="similar name: Large variant"),
   c("Heart of Wicker (Small)", "revenue", "TW", "£736.41", cell(HWS, "Revenue_TW"))]),
 ("- Heart of Wicker (Large) dropped to £637.65 TW from £2,562.85 LW, on 895 units TW.", [
   c("Heart of Wicker (Large)", "revenue", "TW", "£637.65", cell(HWL, "Revenue_TW")),
   c("Heart of Wicker (Large)", "revenue", "LW", "£2,562.85", cell(HWL, "Revenue_LW")),
   wm("Heart of Wicker (Large)", "units", "TW", "895", cell(HWL, "Units_TW"), src=cell(HWL, "Units_LW"), mode="other_period")]),
 ("- Set/4 White Retro Storage Cubes grew 1,306.2% WoW to £1,132.52 TW from just 30 units TW.", [
   c("Set/4 White Retro Storage Cubes", "revenue", "WoW", "1,306.2%", cell("SET/4 WHITE RETRO STORAGE CUBES", "WoW_Pct"), d="up"),
   wv("Set/4 White Retro Storage Cubes", "revenue", "TW", "£1,132.52", cell("SET/4 WHITE RETRO STORAGE CUBES", "Revenue_TW"), why="1,123.52 transposed"),
   c("Set/4 White Retro Storage Cubes", "units", "TW", "30", cell("SET/4 WHITE RETRO STORAGE CUBES", "Units_TW"))]),
 ("- Wooden Rounders Garden Set rose 423.5% WoW to £1,085.30 TW, while Wooden Croquet Garden Set rose 4.7% YoY to £395.50 TW.", [
   c("Wooden Rounders Garden Set", "revenue", "WoW", "423.5%", cell("WOODEN ROUNDERS GARDEN SET", "WoW_Pct"), d="up"),
   c("Wooden Rounders Garden Set", "revenue", "TW", "£1,085.30", cell("WOODEN ROUNDERS GARDEN SET", "Revenue_TW")),
   wd("Wooden Croquet Garden Set", "revenue", "YoY", "4.7%", cell(CROQ, "YoY_Pct"), d="up"),
   wm("Wooden Croquet Garden Set", "revenue", "TW", "£395.50", cell(CROQ, "Revenue_TW"), src=cell(CROQ, "Revenue_LY"), mode="other_period")]),
 ("- The two Heart of Wicker lines together took £1,374.06 TW.", [
   gc("GROUP", "revenue", "TW", "£1,374.06", fam("HEART OF WICKER", "sum", "Revenue_TW"))]),
])

RTP, RTG, RTR = "PINK REGENCY TEACUP AND SAUCER", "GREEN REGENCY TEACUP AND SAUCER", "ROSES REGENCY TEACUP AND SAUCER"
note("n14", G, "terse KPI, parenthetical variants, total", [
 ("- Regency Teacup and Saucer (Pink): £407.91 TW, +16.1% WoW, 141 units TW.", [
   c("Regency Teacup and Saucer (Pink)", "revenue", "TW", "£407.91", cell(RTP, "Revenue_TW")),
   c("Regency Teacup and Saucer (Pink)", "revenue", "WoW", "+16.1%", cell(RTP, "WoW_Pct")),
   c("Regency Teacup and Saucer (Pink)", "units", "TW", "141", cell(RTP, "Units_TW"))]),
 ("- Regency Teacup and Saucer (Green): £397.41 TW, +5.6% WoW; (Roses): £350.46 TW, -29.8% WoW.", [
   c("Regency Teacup and Saucer (Green)", "revenue", "TW", "£397.41", cell(RTG, "Revenue_TW")),
   wd("Regency Teacup and Saucer (Green)", "revenue", "WoW", "+5.6%", cell(RTG, "WoW_Pct")),
   we("Regency Teacup and Saucer (Roses)", "revenue", "TW", "£350.46", cell(RTR, "Revenue_TW"), src=cell("REGENCY TEA PLATE ROSES", "Revenue_TW"), why="similar name: Regency Tea Plate Roses"),
   c("Regency Teacup and Saucer (Roses)", "revenue", "WoW", "-29.8%", cell(RTR, "WoW_Pct"))]),
 ("- Kitchen Scales (Ivory): £550.43 TW, 67 units TW; (Red): £333.20 TW, 30 units TW.", [
   c("Kitchen Scales (Ivory)", "revenue", "TW", "£550.43", cell("IVORY KITCHEN SCALES", "Revenue_TW")),
   c("Kitchen Scales (Ivory)", "units", "TW", "67", cell("IVORY KITCHEN SCALES", "Units_TW")),
   c("Kitchen Scales (Red)", "revenue", "TW", "£333.20", cell("RED KITCHEN SCALES", "Revenue_TW")),
   wm("Kitchen Scales (Red)", "units", "TW", "30", cell("RED KITCHEN SCALES", "Units_TW"), src=cell("RED KITCHEN SCALES", "Units_LW"), mode="other_period")]),
 ("- Bread Bin Diner Style (Ivory): £295.44 TW, -41.1% WoW; (Pink): £230.25 TW, 18 units TW.", [
   c("Bread Bin Diner Style (Ivory)", "revenue", "TW", "£295.44", cell("BREAD BIN DINER STYLE IVORY", "Revenue_TW")),
   c("Bread Bin Diner Style (Ivory)", "revenue", "WoW", "-41.1%", cell("BREAD BIN DINER STYLE IVORY", "WoW_Pct")),
   c("Bread Bin Diner Style (Pink)", "revenue", "TW", "£230.25", cell("BREAD BIN DINER STYLE PINK", "Revenue_TW")),
   we("Bread Bin Diner Style (Pink)", "units", "TW", "18", cell("BREAD BIN DINER STYLE PINK", "Units_TW"), src=cell("BREAD BIN DINER STYLE IVORY", "Units_TW"), why="similar name: Ivory variant")]),
 ("- Gumball Coat Rack: £572.14 TW, +190.5% WoW, 19 orders TW, about £30.11 per order.", [
   c("Gumball Coat Rack", "revenue", "TW", "£572.14", cell("GUMBALL COAT RACK", "Revenue_TW")),
   c("Gumball Coat Rack", "revenue", "WoW", "+190.5%", cell("GUMBALL COAT RACK", "WoW_Pct")),
   c("Gumball Coat Rack", "orders", "TW", "19", cell("GUMBALL COAT RACK", "Orders_TW")),
   c("Gumball Coat Rack", "revenue", "TW", "£30.11", der("GUMBALL COAT RACK", "aov"), q="approx", why="average per order")]),
 ("- Spaceboy Lunch Box: £819 TW, +135.4% WoW; Dolly Girl Lunch Box: £735 TW, +107.0% WoW.", [
   c("Spaceboy Lunch Box", "revenue", "TW", "£819", cell("SPACEBOY LUNCH BOX", "Revenue_TW")),
   c("Spaceboy Lunch Box", "revenue", "WoW", "+135.4%", cell("SPACEBOY LUNCH BOX", "WoW_Pct")),
   wv("Dolly Girl Lunch Box", "revenue", "TW", "£735", cell("DOLLY GIRL LUNCH BOX", "Revenue_TW"), mode="rounding", why="734.30 rounds to 734"),
   c("Dolly Girl Lunch Box", "revenue", "WoW", "+107.0%", cell("DOLLY GIRL LUNCH BOX", "WoW_Pct"))]),
 ("- Lunch Box range (Spaceboy, Dolly Girl, Circus Parade): £2,070.96 TW combined.", [
   gw("GROUP", "revenue", "TW", "£2,070.96", fam("LUNCH BOX", "sum", "Revenue_TW"), why="true 2,007.96")]),
 ("- All 113 lines: £56,180.18 TW, -7.3% WoW.", [
   gc("TOTAL", "revenue", "TW", "£56,180.18", tot("sum", "Revenue_TW")),
   gc("TOTAL", "revenue", "WoW", "-7.3%", tot("wow"))]),
])

note("n15", G, "narrative, comparisons, rank, family total", [
 ("- White Hanging Heart T-Light Holder ranked 4th at £1,708.56 TW, down 46.4% WoW but up 9.7% YoY.", [
   c("White Hanging Heart T-Light Holder", "rank", "TW", "4th", cell(WHH, "Rank")),
   we("White Hanging Heart T-Light Holder", "revenue", "TW", "£1,708.56", cell(WHH, "Revenue_TW"), src=cell("BLACK RECORD COVER FRAME", "Revenue_TW"), why="row above"),
   c("White Hanging Heart T-Light Holder", "revenue", "WoW", "46.4%", cell(WHH, "WoW_Pct"), d="down"),
   c("White Hanging Heart T-Light Holder", "revenue", "YoY", "9.7%", cell(WHH, "YoY_Pct"), d="up")]),
 ("- Spotty Bunting grew 25.4% WoW to £1,341.43 TW on 46 orders TW.", [
   c("Spotty Bunting", "revenue", "WoW", "25.4%", cell("SPOTTY BUNTING", "WoW_Pct"), d="up"),
   c("Spotty Bunting", "revenue", "TW", "£1,341.43", cell("SPOTTY BUNTING", "Revenue_TW")),
   wm("Spotty Bunting", "orders", "TW", "46", cell("SPOTTY BUNTING", "Orders_TW"), src=cell("SPOTTY BUNTING", "Orders_LW"), mode="other_period")]),
 ("- Round Snack Boxes Set of 4 Woodland rose 263.1% WoW to £996.62 TW, up 444.7% YoY.", [
   wm("Round Snack Boxes Set of 4 Woodland", "revenue", "WoW", "263.1%", cell("ROUND SNACK BOXES SET OF4 WOODLAND", "WoW_Pct"), src=der("ROUND SNACK BOXES SET OF4 WOODLAND", "uwow"), d="up", mode="other_metric", why="units change reported as revenue change"),
   c("Round Snack Boxes Set of 4 Woodland", "revenue", "TW", "£996.62", cell("ROUND SNACK BOXES SET OF4 WOODLAND", "Revenue_TW")),
   c("Round Snack Boxes Set of 4 Woodland", "revenue", "YoY", "444.7%", cell("ROUND SNACK BOXES SET OF4 WOODLAND", "YoY_Pct"), d="up")]),
 ("- Small Popcorn Holder fell 87.3% WoW to £426.23 TW from £3,331.18 LW, on 533 units TW vs 4,562 LW.", [
   c("Small Popcorn Holder", "revenue", "WoW", "87.3%", cell("SMALL POPCORN HOLDER", "WoW_Pct"), d="down"),
   wv("Small Popcorn Holder", "revenue", "TW", "£426.23", cell("SMALL POPCORN HOLDER", "Revenue_TW"), why="true 422.63"),
   c("Small Popcorn Holder", "revenue", "LW", "£3,331.18", cell("SMALL POPCORN HOLDER", "Revenue_LW")),
   c("Small Popcorn Holder", "units", "TW", "533", cell("SMALL POPCORN HOLDER", "Units_TW")),
   c("Small Popcorn Holder", "units", "LW", "4,562", cell("SMALL POPCORN HOLDER", "Units_LW"))]),
 ("- Lunch Bag Red Retrospot fell 71.9% WoW to £339.18 TW, while Lunch Bag Apple Design rose 58.3% to £341.56 TW.", [
   c("Lunch Bag Red Retrospot", "revenue", "WoW", "71.9%", cell("LUNCH BAG RED RETROSPOT", "WoW_Pct"), d="down"),
   c("Lunch Bag Red Retrospot", "revenue", "TW", "£339.18", cell("LUNCH BAG RED RETROSPOT", "Revenue_TW")),
   wd("Lunch Bag Apple Design", "revenue", "WoW", "58.3%", cell("LUNCH BAG APPLE DESIGN", "WoW_Pct"), d="up"),
   c("Lunch Bag Apple Design", "revenue", "TW", "£341.56", cell("LUNCH BAG APPLE DESIGN", "Revenue_TW"))]),
 ("- The nine Lunch Bag lines combined took £3,161.11 TW.", [
   gc("GROUP", "revenue", "TW", "£3,161.11", fam("LUNCH BAG", "sum", "Revenue_TW"))]),
 ("- Regency Cakestand 3 Tier units rose 42% YoY to 263 TW.", [
   uv("Regency Cakestand 3 Tier", "units", "YoY", "42%", "units LY not in table", d="up"),
   c("Regency Cakestand 3 Tier", "units", "TW", "263", cell(RCS, "Units_TW"))]),
 ("- Lunch Bag Cars Blue took £348.72 TW, -14.1% WoW and -6.6% YoY.", [
   c("Lunch Bag Cars Blue", "revenue", "TW", "£348.72", cell("LUNCH BAG CARS BLUE", "Revenue_TW")),
   wv("Lunch Bag Cars Blue", "revenue", "WoW", "-14.1%", cell("LUNCH BAG CARS BLUE", "WoW_Pct"), why="true -14.7"),
   c("Lunch Bag Cars Blue", "revenue", "YoY", "-6.6%", cell("LUNCH BAG CARS BLUE", "YoY_Pct"))]),
])

HWB_, HWO = "HAND WARMER BIRD DESIGN", "HAND WARMER OWL DESIGN"
note("n16", H, "narrative, hand warmer family, advent calendar variants", [
 ("- Regency Cakestand 3 Tier led with £6,009.01 TW, up 13.0% WoW on 429 units TW, and up 20% YoY.", [
   c("Regency Cakestand 3 Tier", "revenue", "TW", "£6,009.01", cell(RCS, "Revenue_TW")),
   c("Regency Cakestand 3 Tier", "revenue", "WoW", "13.0%", cell(RCS, "WoW_Pct"), d="up"),
   wm("Regency Cakestand 3 Tier", "units", "TW", "429", cell(RCS, "Units_TW"), src=cell(RCS, "Units_LW"), mode="other_period"),
   uv("Regency Cakestand 3 Tier", "revenue", "YoY", "20%", "Revenue_LY blank for this 2010 table", d="up")]),
 ("- Rotating Silver Angels T-Light Holder surged 681.7% WoW to £5,866.33 TW, a 6.8% share.", [
   c("Rotating Silver Angels T-Light Holder", "revenue", "WoW", "681.7%", cell("ROTATING SILVER ANGELS", "WoW_Pct"), d="up"),
   c("Rotating Silver Angels T-Light Holder", "revenue", "TW", "£5,866.33", cell("ROTATING SILVER ANGELS", "Revenue_TW")),
   we("Rotating Silver Angels T-Light Holder", "share", "TW", "6.8%", cell("ROTATING SILVER ANGELS", "Share_Pct"), src=cell(RCS, "Share_Pct"), why="row above")]),
 ("- Paper Chain Kit 50's Christmas rose 192.1% WoW to £4,046.96 TW, while Paper Chain Kit Vintage Christmas fell 32.0% to £2,338.26 TW.", [
   c("Paper Chain Kit 50's Christmas", "revenue", "WoW", "192.1%", cell("PAPER CHAIN KIT 50'S CHRISTMAS", "WoW_Pct"), d="up"),
   c("Paper Chain Kit 50's Christmas", "revenue", "TW", "£4,046.96", cell("PAPER CHAIN KIT 50'S CHRISTMAS", "Revenue_TW")),
   c("Paper Chain Kit Vintage Christmas", "revenue", "WoW", "32.0%", cell("PAPER CHAIN KIT VINTAGE CHRISTMAS", "WoW_Pct"), d="down"),
   wm("Paper Chain Kit Vintage Christmas", "revenue", "TW", "£2,338.26", cell("PAPER CHAIN KIT VINTAGE CHRISTMAS", "Revenue_TW"), src=cell("PAPER CHAIN KIT VINTAGE CHRISTMAS", "Revenue_LW"), mode="other_period")]),
 ("- Black Record Cover Frame jumped 1,445.8% WoW to £2,185.14 TW, a gain of £2,043.78 on LW's £141.36.", [
   c("Black Record Cover Frame", "revenue", "WoW", "1,445.8%", cell("BLACK RECORD COVER FRAME", "WoW_Pct"), d="up"),
   c("Black Record Cover Frame", "revenue", "TW", "£2,185.14", cell("BLACK RECORD COVER FRAME", "Revenue_TW")),
   c("Black Record Cover Frame", "revenue", "WoW", "£2,043.78", der("BLACK RECORD COVER FRAME", "diff"), d="up", why="absolute change"),
   c("Black Record Cover Frame", "revenue", "LW", "£141.36", cell("BLACK RECORD COVER FRAME", "Revenue_LW"))]),
 ("- Hand Warmer Union Jack sold 394 units TW with no sales LW, while Hand Warmer Bird Design rose 85.4% WoW to £651.95 TW.", [
   c("Hand Warmer Union Jack", "units", "TW", "394", cell("HAND WARMER UNION JACK", "Units_TW")),
   c("Hand Warmer Bird Design", "revenue", "WoW", "85.4%", cell(HWB_, "WoW_Pct"), d="up"),
   we("Hand Warmer Bird Design", "revenue", "TW", "£651.95", cell(HWB_, "Revenue_TW"), src=cell(HWO, "Revenue_TW"), why="similar name: Owl design")]),
 ("- The four hand warmer lines (Bird, Union Jack, Owl, Scotty Dog) combined took £2,970.52 TW.", [
   gw("GROUP", "revenue", "TW", "£2,970.52", fam("HAND WARMER", "sum", "Revenue_TW"), why="true 2,907.52")]),
 ("- Wooden Advent Calendar (Cream) rose 15.0% WoW to £917.20 TW; (Red) rose 23.5% to £764.55 TW.", [
   wv("Wooden Advent Calendar (Cream)", "revenue", "WoW", "15.0%", cell("WOODEN ADVENT CALENDAR CREAM", "WoW_Pct"), d="up", mode="rounding", why="14.94 rounds to 14.9"),
   c("Wooden Advent Calendar (Cream)", "revenue", "TW", "£917.20", cell("WOODEN ADVENT CALENDAR CREAM", "Revenue_TW")),
   wv("Wooden Advent Calendar (Red)", "revenue", "WoW", "23.5%", cell("WOODEN ADVENT CALENDAR RED", "WoW_Pct"), d="up", why="32.5 transposed"),
   c("Wooden Advent Calendar (Red)", "revenue", "TW", "£764.55", cell("WOODEN ADVENT CALENDAR RED", "Revenue_TW"))]),
 ("- The top three lines took 18.0% of revenue TW.", [
   gc("GROUP", "share", "TW", "18.0%", topk(3, "share"))]),
])

note("n17", H, "terse KPI, averages per order, rest of table", [
 ("- Set of 3 Babushka Stacking Tins: £1,406.34 TW vs £54.54 LW, +2,482.8% WoW.", [
   c("Set of 3 Babushka Stacking Tins", "revenue", "TW", "£1,406.34", cell("SET OF 3 BABUSHKA STACKING TINS", "Revenue_TW")),
   wv("Set of 3 Babushka Stacking Tins", "revenue", "LW", "£54.54", cell("SET OF 3 BABUSHKA STACKING TINS", "Revenue_LW"), why="54.45 transposed"),
   c("Set of 3 Babushka Stacking Tins", "revenue", "WoW", "+2,482.8%", cell("SET OF 3 BABUSHKA STACKING TINS", "WoW_Pct"))]),
 ("- Set 7 Babushka Nesting Boxes: £1,322.54 TW, 282 units, 22 orders TW.", [
   c("Set 7 Babushka Nesting Boxes", "revenue", "TW", "£1,322.54", cell("SET 7 BABUSHKA NESTING BOXES", "Revenue_TW")),
   we("Set 7 Babushka Nesting Boxes", "units", "TW", "282", cell("SET 7 BABUSHKA NESTING BOXES", "Units_TW"), src=cell("SET OF 3 BABUSHKA STACKING TINS", "Units_TW"), why="similar name: Babushka tins"),
   wm("Set 7 Babushka Nesting Boxes", "orders", "TW", "22", cell("SET 7 BABUSHKA NESTING BOXES", "Orders_TW"), src=cell("SET 7 BABUSHKA NESTING BOXES", "Orders_LW"), mode="other_period")]),
 ("- Pink Blue Felt Craft Trinket Box: £807.93 TW, +271.4% WoW; Pink Cream Felt Craft Trinket Box: £663.80 TW, +25.3% WoW.", [
   c("Pink Blue Felt Craft Trinket Box", "revenue", "TW", "£807.93", cell("PINK BLUE FELT CRAFT TRINKET BOX", "Revenue_TW")),
   c("Pink Blue Felt Craft Trinket Box", "revenue", "WoW", "+271.4%", cell("PINK BLUE FELT CRAFT TRINKET BOX", "WoW_Pct")),
   c("Pink Cream Felt Craft Trinket Box", "revenue", "TW", "£663.80", cell("PINK CREAM FELT CRAFT TRINKET BOX", "Revenue_TW")),
   wd("Pink Cream Felt Craft Trinket Box", "revenue", "WoW", "+25.3%", cell("PINK CREAM FELT CRAFT TRINKET BOX", "WoW_Pct"))]),
 ("- Pack of 72 Retrospot Cake Cases: 2,491 units TW, £1,153.17 TW, about £32.95 per order.", [
   c("Pack of 72 Retrospot Cake Cases", "units", "TW", "2,491", cell("PACK OF 72 RETROSPOT CAKE CASES", "Units_TW")),
   c("Pack of 72 Retrospot Cake Cases", "revenue", "TW", "£1,153.17", cell("PACK OF 72 RETROSPOT CAKE CASES", "Revenue_TW")),
   c("Pack of 72 Retrospot Cake Cases", "revenue", "TW", "£32.95", der("PACK OF 72 RETROSPOT CAKE CASES", "aov"), q="approx", why="average per order")]),
 ("- Brocade Ring Purse: 5,632 units TW from 6 orders, revenue up 3,063.7% WoW.", [
   c("Brocade Ring Purse", "units", "TW", "5,632", cell("BROCADE RING PURSE", "Units_TW")),
   wm("Brocade Ring Purse", "orders", "TW", "6", cell("BROCADE RING PURSE", "Orders_TW"), src=cell("BROCADE RING PURSE", "Orders_LW"), mode="other_period"),
   c("Brocade Ring Purse", "revenue", "WoW", "3,063.7%", cell("BROCADE RING PURSE", "WoW_Pct"), d="up")]),
 ("- Gold Wine Glass: £976.32 TW from a single order, 1.1% share, units up 25% YoY.", [
   c("Gold Wine Glass", "revenue", "TW", "£976.32", cell("GOLD WINE GLASS", "Revenue_TW")),
   c("Gold Wine Glass", "share", "TW", "1.1%", cell("GOLD WINE GLASS", "Share_Pct")),
   uv("Gold Wine Glass", "units", "YoY", "25%", "units LY not in table (and no LY for 2010)", d="up")]),
 ("- Chilli Lights: £2.4k TW, +92.2% WoW, rank 5.", [
   wv("Chilli Lights", "revenue", "TW", "£2.4k", cell("CHILLI LIGHTS", "Revenue_TW"), mode="rounding", why="2.474k rounds to 2.5k"),
   c("Chilli Lights", "revenue", "WoW", "+92.2%", cell("CHILLI LIGHTS", "WoW_Pct")),
   c("Chilli Lights", "rank", "TW", "5", cell("CHILLI LIGHTS", "Rank"))]),
 ("- Rest of the table outside the top ten: £58,037.59 TW.", [
   gc("GROUP", "revenue", "TW", "£58,037.59", rest(10, "sum", "Revenue_TW"))]),
])

note("n18", H, "mixed narrative, totals, similar-name swaps", [
 ("- All 84 lines generated £88,458.98 TW, up 46.5% WoW.", [
   gc("TOTAL", "revenue", "TW", "£88,458.98", tot("sum", "Revenue_TW")),
   gw("TOTAL", "revenue", "WoW", "46.5%", tot("wow"), d="up", why="true +45.6%, digits transposed")]),
 ("- Party Bunting fell 70.1% WoW to £1,078.12 TW, on 16 orders TW vs 36 LW.", [
   c("Party Bunting", "revenue", "WoW", "70.1%", cell(PB, "WoW_Pct"), d="down"),
   c("Party Bunting", "revenue", "TW", "£1,078.12", cell(PB, "Revenue_TW")),
   c("Party Bunting", "orders", "TW", "16", cell(PB, "Orders_TW")),
   c("Party Bunting", "orders", "LW", "36", cell(PB, "Orders_LW"))]),
 ("- Heart of Wicker Large fell 38.3% WoW to £1,142.12 TW; Heart of Wicker Small fell 44.2% to £975.46 TW.", [
   we("Heart of Wicker Large", "revenue", "WoW", "38.3%", cell(HWL, "WoW_Pct"), src=cell(HWS, "WoW_Pct"), d="down", why="Large/Small changes swapped"),
   c("Heart of Wicker Large", "revenue", "TW", "£1,142.12", cell(HWL, "Revenue_TW")),
   we("Heart of Wicker Small", "revenue", "WoW", "44.2%", cell(HWS, "WoW_Pct"), src=cell(HWL, "WoW_Pct"), d="down", why="Large/Small changes swapped"),
   c("Heart of Wicker Small", "revenue", "TW", "£975.46", cell(HWS, "Revenue_TW"))]),
 ("- Hot Water Bottle Tea and Sympathy rose 65.6% WoW to £1,270.08 TW on 27 orders TW.", [
   c("Hot Water Bottle Tea and Sympathy", "revenue", "WoW", "65.6%", cell("HOT WATER BOTTLE TEA AND SYMPATHY", "WoW_Pct"), d="up"),
   c("Hot Water Bottle Tea and Sympathy", "revenue", "TW", "£1,270.08", cell("HOT WATER BOTTLE TEA AND SYMPATHY", "Revenue_TW")),
   wm("Hot Water Bottle Tea and Sympathy", "orders", "TW", "27", cell("HOT WATER BOTTLE TEA AND SYMPATHY", "Orders_TW"), src=cell("HOT WATER BOTTLE TEA AND SYMPATHY", "Orders_LW"), mode="other_period")]),
 ("- Chocolate Hot Water Bottle rose 47.0% WoW to £710.64 TW, while Scottie Dog Hot Water Bottle rose 32.1% to £901.09 TW.", [
   wd("Chocolate Hot Water Bottle", "revenue", "WoW", "47.0%", cell("CHOCOLATE HOT WATER BOTTLE", "WoW_Pct"), d="up"),
   c("Chocolate Hot Water Bottle", "revenue", "TW", "£710.64", cell("CHOCOLATE HOT WATER BOTTLE", "Revenue_TW")),
   c("Scottie Dog Hot Water Bottle", "revenue", "WoW", "32.1%", cell("SCOTTIE DOG HOT WATER BOTTLE", "WoW_Pct"), d="up"),
   c("Scottie Dog Hot Water Bottle", "revenue", "TW", "£901.09", cell("SCOTTIE DOG HOT WATER BOTTLE", "Revenue_TW"))]),
 ("- Five hot water bottle lines (Tea and Sympathy, Scottie Dog, Chocolate, Retrospot Heart, I Am So Poorly) totalled £4,229.32 TW.", [
   gc("GROUP", "revenue", "TW", "£4,229.32", grp(["HOT WATER BOTTLE TEA AND SYMPATHY", "SCOTTIE DOG HOT WATER BOTTLE", "CHOCOLATE HOT WATER BOTTLE", "RETROSPOT HEART HOT WATER BOTTLE", "HOT WATER BOTTLE I AM SO POORLY"], "sum", "Revenue_TW"))]),
 ("- Jumbo Bag Red Retrospot fell 40.8% WoW to £2,063.56 TW, 2.3% of revenue.", [
   c("Jumbo Bag Red Retrospot", "revenue", "WoW", "40.8%", cell(JRR, "WoW_Pct"), d="down"),
   we("Jumbo Bag Red Retrospot", "revenue", "TW", "£2,063.56", cell(JRR, "Revenue_TW"), src=cell("COLOUR GLASS. STAR T-LIGHT HOLDER", "Revenue_TW"), why="row above"),
   c("Jumbo Bag Red Retrospot", "share", "TW", "2.3%", cell(JRR, "Share_Pct"))]),
 ("- World War 2 Gliders sold 3,552 units TW, up from 770 LW, with revenue of £710 TW.", [
   c("World War 2 Gliders", "units", "TW", "3,552", cell("WORLD WAR 2 GLIDERS", "Units_TW")),
   c("World War 2 Gliders", "units", "LW", "770", cell("WORLD WAR 2 GLIDERS", "Units_LW")),
   wv("World War 2 Gliders", "revenue", "TW", "£710", cell("WORLD WAR 2 GLIDERS", "Revenue_TW"), mode="rounding", why="709.44 rounds to 709")]),
])

# ================================================================ checking
EXPECTED_GATE = {"Correct": "Correct", "Wrong_entity": "Wrong_entity", "Wrong_metric": "Wrong_metric",
                 "Wrong_value": "Wrong_value", "Wrong_direction": "Wrong_value",
                 "Group_correct": "Correct", "Group_wrong": "Wrong_*", "Unverifiable": "Unverifiable"}
NUMCOLS = ["Revenue_TW", "Revenue_LW", "Revenue_LY", "Units_TW", "Units_LW", "Orders_TW", "Orders_LW",
           "WoW_Pct", "YoY_Pct", "Share_Pct", "Rank"]


def resolve(df, name):
    ents = list(df["Entity"])
    if name in ents:
        return ents.index(name)
    hits = [i for i, e in enumerate(ents) if name.upper() in e.upper()]
    assert len(hits) == 1, f"entity {name!r} resolves to {len(hits)} rows"
    return hits[0]


def fnum(x):
    return None if pd.isna(x) else float(x)


def derived(df, i, kind):
    r = df.iloc[i]
    if kind == "aov":
        return r.Revenue_TW / r.Orders_TW
    if kind == "ratio":
        return r.Revenue_TW / r.Revenue_LW if r.Revenue_LW else None
    if kind == "diff":
        return r.Revenue_TW - r.Revenue_LW
    if kind == "uwow":
        return 100 * (r.Units_TW / r.Units_LW - 1) if r.Units_LW else None
    raise ValueError(kind)


def agg(df, idx, kind, col):
    sub = df.iloc[list(idx)]
    if kind == "sum":
        return float(sub[col].sum())
    if kind == "share":
        return 100 * float(sub.Revenue_TW.sum()) / float(df.Revenue_TW.sum())
    if kind == "wow":
        return 100 * (float(sub.Revenue_TW.sum()) / float(sub.Revenue_LW.sum()) - 1)
    raise ValueError(kind)


def evaluate(df, ref):
    """Return (value, description, member rows)."""
    k = ref[0]
    if k == "cell":
        i = resolve(df, ref[1])
        return fnum(df.iloc[i][ref[2]]), f"{df.Entity[i]} | {ref[2]}", [i]
    if k == "der":
        i = resolve(df, ref[1])
        return derived(df, i, ref[2]), f"{df.Entity[i]} | derived {ref[2]}", [i]
    if k == "grp":
        idx = [resolve(df, n) for n in ref[1]]
    elif k == "fam":
        idx = [i for i, e in enumerate(df.Entity) if ref[1] in e.replace("  ", " ")]
    elif k == "tot":
        idx = list(range(len(df)))
    elif k == "topk":
        idx = list(df.sort_values("Rank", kind="stable").index[: ref[1]])
    elif k == "rest":
        idx = list(df.sort_values("Rank", kind="stable").index[ref[1]:])
    kind, col = ref[-2], ref[-1]
    names = [df.Entity[i] for i in idx]
    return agg(df, idx, kind, col), f"{k} {kind} {col or ''} over {len(idx)} rows: " + "; ".join(names), idx


VT = re.compile(r"^(?P<sign>[+\-])?(?P<cur>£)?(?P<num>\d[\d,]*(?:\.\d+)?)(?P<suf>k|M|%|x|st|nd|rd|th)?$")


def parse_vt(vt):
    m = VT.match(vt)
    assert m, f"cannot parse value_text {vt!r}"
    num = m.group("num").replace(",", "")
    dec = len(num.split(".")[1]) if "." in num else 0
    suf = m.group("suf") or ""
    scale = {"k": 1000, "M": 1000000}.get(suf, 1)
    sign = {"+": 1, "-": -1}.get(m.group("sign"))
    return dict(num=Decimal(num), dec=dec, scale=scale, sign=sign, suf=suf)


def dq(x):
    return Decimal(repr(round(float(x), 9)))


def half_up(x, dec):
    return x.quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP)


def is_change(claim, p):
    return claim["period"] in ("WoW", "YoY") and p["suf"] != "x"


def note_sign(claim, p):
    if p["sign"] is not None:
        return p["sign"]
    return {"up": 1, "down": -1}.get(claim["direction"])


def matches(true, claim, magnitude_only=False):
    """Does the cited value equal `true` at the cited precision (half-up), honouring
    qualifier and, for changes, the stated sign/direction?"""
    if true is None:
        return False
    p = parse_vt(claim["value_text"])
    tv = dq(true) / p["scale"]
    cited = p["num"]
    if is_change(claim, p):
        s = note_sign(claim, p)
        if s is not None and not magnitude_only:
            cited = s * cited
        else:
            tv = abs(tv)
    q = claim["qualifier"]
    if q == "over":
        return tv > cited
    if q == "under":
        return tv < cited
    return half_up(tv, p["dec"]) == cited


def is_tie(true, claim):
    p = parse_vt(claim["value_text"])
    x = abs(dq(true) / p["scale"]).scaleb(p["dec"])
    return x - int(x) == Decimal("0.5")


def coincidences(df, i, claim, own_value):
    """Other table values the cited number also matches (same row, same column elsewhere)."""
    hits = []
    for col in NUMCOLS:
        v = fnum(df.iloc[i][col])
        if v is not None and matches(v, claim, magnitude_only=True):
            hits.append(f"{df.Entity[i]}|{col}")
    for kind in ("aov", "ratio", "diff", "uwow"):
        v = derived(df, i, kind)
        if v is not None and matches(v, claim, magnitude_only=True):
            hits.append(f"{df.Entity[i]}|{kind}")
    return hits


rows, out_notes, problems, warnings = [], [], [], []
for n in NOTES:
    df = TABLES[n["table_id"]]
    text_lines, claims_out = [], []
    for li, (text, claims) in enumerate(n["lines"], start=1):
        assert text.startswith("- "), text
        assert "—" not in text, "em dash"
        text_lines.append(text)
        pos = 0
        for ci, cl in enumerate(claims):
            vt = cl["value_text"]
            j = text.find(vt, pos)
            assert j >= 0, f"{n['note_id']} L{li}: value_text {vt!r} not found in order in: {text}"
            pos = j + len(vt)
            truth = cl["truth"]
            true_val, ref_desc, idx = (None, "", [])
            if cl["ref"] is not None:
                true_val, ref_desc, idx = evaluate(df, cl["ref"])
            src_val, src_desc = (None, "")
            if cl["src"] is not None:
                src_val, src_desc, _ = evaluate(df, cl["src"])
            ok, note_ = True, ""
            nwarn = len(warnings)
            if truth in ("Correct", "Group_correct"):
                ok = matches(true_val, cl)
                if true_val is not None and cl["qualifier"] in ("none", "approx") and is_tie(true_val, cl):
                    warnings.append(f"{n['note_id']} L{li} {vt}: exact half at cited precision")
            elif truth in ("Wrong_entity", "Wrong_metric"):
                ok = (not matches(true_val, cl)) and matches(src_val, cl)
                own = coincidences(df, idx[0], cl, true_val)
                if truth == "Wrong_entity" and own:
                    warnings.append(f"{n['note_id']} L{li} {vt}: other-row value also matches own row {own}")
                if truth == "Wrong_metric" and cl["ref"][0] == "cell":
                    col = cl["ref"][2]
                    others = [df.Entity[k2] for k2 in range(len(df)) if k2 != idx[0]
                              and fnum(df.iloc[k2][col]) is not None and matches(fnum(df.iloc[k2][col]), cl, magnitude_only=True)]
                    if others:
                        warnings.append(f"{n['note_id']} L{li} {vt}: other-period value also matches {col} of {others}")
            elif truth == "Wrong_direction":
                ok = (not matches(true_val, cl)) and matches(true_val, cl, magnitude_only=True)
            elif truth in ("Wrong_value", "Group_wrong"):
                ok = not matches(true_val, cl) and true_val is not None
                if truth == "Wrong_value":
                    hits = coincidences(df, idx[0], cl, true_val) if idx else []
                    col = cl["ref"][2] if cl["ref"][0] == "cell" else None
                    if col:
                        for k2 in range(len(df)):
                            v = fnum(df.iloc[k2][col])
                            if k2 != idx[0] and v is not None and matches(v, cl, magnitude_only=True):
                                hits.append(f"{df.Entity[k2]}|{col}")
                    if hits:
                        warnings.append(f"{n['note_id']} L{li} {vt}: drift value coincides with {hits}")
            elif truth == "Unverifiable":
                ok = cl["ref"] is None
            if not ok:
                problems.append(f"{n['note_id']} L{li} {truth} {vt}: true={true_val} src={src_val} ({ref_desc})")
            # group share: record the sum-of-rounded-shares alternative
            if cl["ref"] is not None and cl["ref"][0] in ("grp", "fam", "topk", "rest", "tot") and cl["ref"][-2] == "share":
                alt = float(df.iloc[idx]["Share_Pct"].sum())
                agree = matches(alt, cl)
                note_ = f"sum of rounded Share_Pct = {alt:.1f} ({'agrees' if agree else 'DIFFERS'})"
            amb = [w.split(": ", 1)[1] for w in warnings[nwarn:]]
            p = parse_vt(vt)
            kind = "total" if cl["entity"] == "TOTAL" else "group" if cl["entity"] == "GROUP" else "single"
            claim_json = dict(line=li, entity=cl["entity"], metric=cl["metric"], period=cl["period"],
                              value_text=vt, direction=cl["direction"], qualifier=cl["qualifier"])
            claims_out.append(claim_json)
            rows.append(dict(
                note_id=n["note_id"], table_id=n["table_id"], claim_idx=len(claims_out) - 1,
                line=li, entity=cl["entity"], metric=cl["metric"], period=cl["period"], value_text=vt,
                direction=cl["direction"], qualifier=cl["qualifier"], kind=kind, truth=truth,
                expected_gate=EXPECTED_GATE[truth], error_mode=cl["mode"] if truth not in ("Correct", "Group_correct") else "",
                true_value="" if true_val is None else round(true_val, 4), true_ref=ref_desc,
                cited_source=src_desc, source_value="" if src_val is None else round(src_val, 4),
                rounding_check="pass" if ok else "FAIL", remark="; ".join(x for x in [cl["why"], note_] + amb if x)))
    out_notes.append(dict(note_id=n["note_id"], table_id=n["table_id"], style=n["style"],
                          text="\n".join(text_lines), claims=claims_out))
    assert 6 <= len(text_lines) <= 8, n["note_id"]

for w in warnings:
    print("WARN", w)
if problems:
    for p_ in problems:
        print("PROBLEM", p_)
    sys.exit(f"{len(problems)} truth assertions failed")

tdf = pd.DataFrame(rows)
print(f"notes={len(out_notes)} claims={len(tdf)}")
print(tdf.truth.value_counts().to_string())
print((tdf.truth.value_counts(normalize=True) * 100).round(1).to_string())
print(tdf[tdf.error_mode != ""].groupby(["truth", "error_mode"]).size().to_string())
print(tdf.groupby("note_id").size().to_string())
print("similar-name swaps:", tdf.remark.str.contains("similar name|swapped").sum())

if not DRY:
    with open(os.path.join(OUT, "notes.json"), "w", encoding="utf-8") as f:
        json.dump(out_notes, f, indent=1, ensure_ascii=False)
    tdf.to_csv(os.path.join(OUT, "truth.csv"), index=False, quoting=csv.QUOTE_MINIMAL)
    print("wrote notes.json and truth.csv")
