from pathlib import Path
import re, json
from fractions import Fraction as F
from decimal import Decimal
import pandas as pd
T = str(Path(__file__).resolve().parents[4] / "data" / "tables") + "/"
B = str(Path(__file__).resolve().parents[1]) + "/"
COLS = ["Revenue_TW","Revenue_LW","Revenue_LY","Units_TW","Units_LW","Orders_TW","Orders_LW","WoW_Pct","YoY_Pct","Share_Pct","Rank"]

def fr(x):
    if x is None or (isinstance(x, float) and pd.isna(x)): return None
    return F(Decimal(repr(float(x))))

def load(tid):
    return pd.read_csv(T + tid + ".csv")

def parse(vt):
    s = vt.strip()
    signed = s[0] in "+-−"
    neg = s[0] in "-−"
    m = re.search(r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*([kKmMbB]?)", s)
    num = m.group(1).replace(",", "")
    dec = len(num.split(".")[1]) if "." in num else 0
    suf = m.group(2).lower()
    scale = {"": 1, "k": 1000, "m": 10**6, "b": 10**9}[suf]
    unit = "pct" if "%" in s else "gbp" if "£" in s else "ratio" if s.endswith("x") else "rank" if re.search(r"\d(st|nd|rd|th)", s) else "plain"
    return dict(val=F(Decimal(num)), dec=dec, scale=scale, signed=signed, neg=neg, unit=unit, suf=suf)

def match(true, p, strict_sign=True, ktol=False):
    """half-up rounding at cited precision: -h <= |true|/scale - |cited| < h"""
    if true is None: return False
    h = F(1, 2) / (10 ** p["dec"])
    d = abs(true) / p["scale"] - p["val"]
    ok = (-h <= d < h)
    if ktol and p["suf"] and not ok:
        ok = abs(abs(true) / p["scale"] - p["val"]) <= F(5, 1000) * abs(true) / p["scale"]  # k/M 0.5% relative
    if ok and strict_sign and p["signed"]:
        ok = (true < 0) == p["neg"] or true == 0
    return ok

def pct(a, b):
    a, b = fr(a), fr(b)
    if a is None or b is None or b == 0: return None
    return (a - b) / b * 100

def derived(df, i):
    r = df.loc[i]
    tot = {c: sum(fr(x) for x in df[c].dropna()) for c in ["Revenue_TW","Revenue_LW","Revenue_LY","Units_TW","Units_LW","Orders_TW","Orders_LW"] if df[c].notna().any()}
    out = {}
    for c in COLS:
        if pd.notna(r[c]): out["cell:" + c] = fr(r[c])
    out["rev_wow_raw"] = pct(r.Revenue_TW, r.Revenue_LW)
    out["rev_yoy_raw"] = pct(r.Revenue_TW, r.Revenue_LY) if pd.notna(r.Revenue_LY) else None
    out["units_wow"] = pct(r.Units_TW, r.Units_LW)
    out["orders_wow"] = pct(r.Orders_TW, r.Orders_LW)
    out["rev_diff"] = fr(r.Revenue_TW) - fr(r.Revenue_LW)
    out["units_diff"] = fr(r.Units_TW) - fr(r.Units_LW)
    out["rev_ratio"] = fr(r.Revenue_TW) / fr(r.Revenue_LW) if r.Revenue_LW else None
    out["share_raw"] = fr(r.Revenue_TW) / tot["Revenue_TW"] * 100
    out["share_LW"] = fr(r.Revenue_LW) / tot["Revenue_LW"] * 100 if tot["Revenue_LW"] else None
    if "Revenue_LY" in tot: out["share_LY"] = fr(r.Revenue_LY) / tot["Revenue_LY"] * 100
    out["units_share"] = fr(r.Units_TW) / tot["Units_TW"] * 100
    out["orders_share"] = fr(r.Orders_TW) / tot["Orders_TW"] * 100
    out["rev_per_order"] = fr(r.Revenue_TW) / fr(r.Orders_TW) if r.Orders_TW else None
    out["rev_per_unit"] = fr(r.Revenue_TW) / fr(r.Units_TW) if r.Units_TW else None
    return {k: v for k, v in out.items() if v is not None}

def search(df, p, strict_sign=True):
    """every cell of the table matching the cited value"""
    hits = []
    for i, r in df.iterrows():
        for c in COLS:
            if pd.notna(r[c]) and match(fr(r[c]), p, strict_sign):
                hits.append((r.Entity, c, float(r[c])))
    return hits
