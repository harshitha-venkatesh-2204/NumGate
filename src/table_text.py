"""Render report tables as markdown for prompts, parse them back, and format numbers for notes."""
import io

import numpy as np
import pandas as pd

DECIMALS = {"Revenue_TW": 2, "Revenue_LW": 2, "Revenue_LY": 2, "WoW_Pct": 1, "YoY_Pct": 1, "Share_Pct": 1}


def cell_text(col, v):
    if pd.isna(v):
        return ""
    if col in DECIMALS:
        return f"{float(v):.{DECIMALS[col]}f}"
    if isinstance(v, (int, np.integer, float, np.floating)):
        return str(int(v))
    return str(v)


def to_markdown(df):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell_text(c, row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def from_markdown(text):
    """First markdown table found in text, as a DataFrame with numeric columns converted."""
    rows = [l.strip() for l in text.splitlines() if l.strip().startswith("|")]
    rows = [r for r in rows if not set(r) <= set("|-: ")]
    if not rows:
        return None
    cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
    df = pd.DataFrame(cells[1:], columns=cells[0])
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col].replace("", np.nan))
    return df


def money(v, compact=False):
    """£ with thousands separators, or compact k / M form rounded to one decimal."""
    if compact and abs(v) >= 1e6:
        return f"£{v / 1e6:.1f}M"
    if compact and abs(v) >= 1e3:
        return f"£{v / 1e3:.1f}k"
    return f"£{v:,.2f}"


def pct(v, signed=False):
    return f"{v:+.1f}%" if signed else f"{abs(v):.1f}%"


def csv_head(df, n=3):
    buf = io.StringIO()
    df.head(n).to_csv(buf, index=False)
    return buf.getvalue()
