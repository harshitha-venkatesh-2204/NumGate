"""Build weekly report tables from UCI Online Retail II.

Usage: python src/build_tables.py
Writes data/tables/<table_id>.csv and data/tables/index.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

from config import DATA_INTERIM, DATA_RAW, TABLES_DIR, load_config

UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
COLUMNS = ["Entity", "Revenue_TW", "Revenue_LW", "Revenue_LY", "Units_TW", "Units_LW",
           "Orders_TW", "Orders_LW", "WoW_Pct", "YoY_Pct", "Share_Pct", "Rank"]


def download_uci():
    xlsx = DATA_RAW / "online_retail_II.xlsx"
    if xlsx.exists():
        return xlsx
    import io
    import zipfile

    import requests
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print("Downloading UCI Online Retail II ...")
    resp = requests.get(UCI_URL, timeout=300)
    resp.raise_for_status()
    (DATA_RAW / "online_retail_ii.zip").write_bytes(resp.content)
    zipfile.ZipFile(io.BytesIO(resp.content)).extractall(DATA_RAW)
    return xlsx


def load_clean():
    """Read both yearly sheets, drop the overlap, clean, and cache as CSV."""
    cached = DATA_INTERIM / "online_retail_clean.csv"
    if cached.exists():
        return pd.read_csv(cached, parse_dates=["InvoiceDate"], dtype={"InvoiceNo": str, "StockCode": str})

    xlsx = download_uci()
    print("Reading Excel (takes a minute or two) ...")
    sheets = pd.read_excel(xlsx, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    first, second = [sheets[name] for name in sorted(sheets)]  # "Year 2009-2010", "Year 2010-2011"
    # The two sheets overlap in early December 2010; keep each invoice once, from the later sheet.
    first = first[~first["Invoice"].isin(set(second["Invoice"]))]
    df = pd.concat([first, second], ignore_index=True)
    df = df.rename(columns={"Invoice": "InvoiceNo", "Price": "UnitPrice", "Customer ID": "CustomerID"})

    n_before = len(df)
    df = df[~df["InvoiceNo"].str.startswith("C")]  # cancelled invoices
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df[df["Country"].notna() & (df["Country"] != "Unspecified")]
    # Keep real products only: stock codes start with 5 digits. Drops POST, DOT, M, BANK CHARGES, etc.
    df = df[df["StockCode"].str.match(r"^\d{5}")]
    df["Description"] = df["Description"].astype(str).str.strip().str.upper()
    df = df[df["Description"].ne("NAN") & df["Description"].ne("")]
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    print(f"Cleaned rows: {len(df):,} of {n_before:,}")

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_csv(cached, index=False)
    return df


def add_weeks(df):
    iso = df["InvoiceDate"].dt.isocalendar()
    df["iso_year"] = iso["year"].astype(int)
    df["iso_week"] = iso["week"].astype(int)
    df["week"] = df["iso_year"].astype(str) + "-W" + df["iso_week"].astype(str).str.zfill(2)
    return df


def weekly_by_entity(df, entity_col):
    g = df.groupby(["week", entity_col])
    out = pd.DataFrame({
        "Revenue": g["Revenue"].sum(),
        "Units": g["Quantity"].sum(),
        "Orders": g["InvoiceNo"].nunique(),
    }).reset_index().rename(columns={entity_col: "Entity"})
    return out


def previous_week(week):
    year, wk = int(week[:4]), int(week[-2:])
    monday = pd.Timestamp.fromisocalendar(year, wk, 1) - pd.Timedelta(days=7)
    y, w, _ = monday.isocalendar()
    return f"{y}-W{w:02d}"


def last_year_week(week):
    return f"{int(week[:4]) - 1}{week[4:]}"


def eligible_weeks(df):
    """Weeks with at least 4 trading days whose previous week also has at least 4."""
    days = df.groupby("week")["InvoiceDate"].apply(lambda s: s.dt.date.nunique())
    good = set(days[days >= 4].index)
    return sorted(w for w in good if previous_week(w) in good)


def pct_change(new, old):
    # Blank when the base is zero or missing, so the CSV never holds inf.
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (new - old) / old * 100
    return pct.where(old > 0).round(1)


def make_table(weekly, week, n_rows, all_weeks):
    tw = weekly[weekly["week"] == week].set_index("Entity")
    lw = weekly[weekly["week"] == previous_week(week)].set_index("Entity")
    ly_week = last_year_week(week)
    ly = weekly[weekly["week"] == ly_week].set_index("Entity")

    top = tw[tw["Revenue"] > 0].sort_values("Revenue", ascending=False).head(n_rows)
    t = pd.DataFrame({"Entity": top.index})
    t["Revenue_TW"] = top["Revenue"].round(2).values
    t["Revenue_LW"] = t["Entity"].map(lw["Revenue"]).fillna(0).round(2)
    if ly_week in all_weeks:
        t["Revenue_LY"] = t["Entity"].map(ly["Revenue"]).fillna(0).round(2)
    else:
        t["Revenue_LY"] = np.nan  # prior year not covered by the data
    t["Units_TW"] = top["Units"].astype(int).values
    t["Units_LW"] = t["Entity"].map(lw["Units"]).fillna(0).astype(int)
    t["Orders_TW"] = top["Orders"].astype(int).values
    t["Orders_LW"] = t["Entity"].map(lw["Orders"]).fillna(0).astype(int)
    add_derived(t)
    return t[COLUMNS]


def add_derived(t):
    """Derived columns are computed from the rounded stored values, so they recompute exactly."""
    t["WoW_Pct"] = pct_change(t["Revenue_TW"], t["Revenue_LW"])
    t["YoY_Pct"] = pct_change(t["Revenue_TW"], t["Revenue_LY"])
    t["Share_Pct"] = (t["Revenue_TW"] / t["Revenue_TW"].sum() * 100).round(1)
    t["Rank"] = t["Revenue_TW"].rank(ascending=False, method="min").astype(int)
    return t


def verify_table(t):
    """Return the list of derived columns that do not recompute from the raw columns."""
    check = add_derived(t[COLUMNS[:8]].copy())
    bad = []
    for col in ["WoW_Pct", "YoY_Pct", "Share_Pct", "Rank"]:
        a, b = t[col].astype(float), check[col].astype(float)
        same = ((a - b).abs() < 1e-6) | (a.isna() & b.isna())
        if not same.all():
            bad.append(col)
    return bad


def plan_tables(cfg):
    """(size, entity_type, count) plan.

    Only about 13 countries trade in a typical week (never more than 20), so countries
    can fill small tables only; medium and large tables use products.
    """
    n = cfg["tables_per_size"]
    half = n // 2
    return [("small", "country", half), ("small", "product", n - half),
            ("medium", "product", n), ("large", "product", n)]


def build_all():
    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])  # one seeded generator drives every random choice
    df = add_weeks(load_clean())
    weeks = eligible_weeks(df)
    all_weeks = set(df["week"].unique())
    weekly = {"country": weekly_by_entity(df, "Country"), "product": weekly_by_entity(df, "Description")}
    row_range = {s: cfg["tables"][f"{s}_rows"] for s in ["small", "medium", "large"]}

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for old in TABLES_DIR.glob("*.csv"):
        old.unlink()
    index = []
    for size, etype, count in plan_tables(cfg):
        lo, hi = row_range[size]
        candidate_weeks = list(rng.permutation(weeks))
        made = 0
        for week in candidate_weeks:
            if made == count:
                break
            n_rows = int(rng.integers(lo, hi + 1))
            available = (weekly[etype].query("week == @week")["Revenue"] > 0).sum()
            if available < lo:
                continue  # not enough entities trading this week for this size
            t = make_table(weekly[etype], week, min(n_rows, available), all_weeks)
            table_id = f"{size}_{etype}_{week}"
            t.to_csv(TABLES_DIR / f"{table_id}.csv", index=False)
            index.append({"table_id": table_id, "source": "uci_online_retail_ii", "entity_type": etype,
                          "week": week, "size": size, "n_rows": len(t)})
            made += 1
        if made < count:
            print(f"Warning: only {made} of {count} {size} {etype} tables could be built")

    idx = pd.DataFrame(index).sort_values(["size", "table_id"])
    idx.to_csv(TABLES_DIR / "index.csv", index=False)
    return idx


def maybe_m5():
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("Kaggle credentials found, but the M5 table builder is not implemented yet; skipping M5.")
    else:
        print("No Kaggle credentials; skipping optional M5 source.")


def main():
    idx = build_all()
    maybe_m5()
    bad = {}
    for table_id in idx["table_id"]:
        cols = verify_table(pd.read_csv(TABLES_DIR / f"{table_id}.csv"))
        if cols:
            bad[table_id] = cols
    print(idx.groupby("size")["n_rows"].agg(["count", "min", "max"]).to_string())
    print(f"Tables: {len(idx)}. Derived-column recompute failures: {len(bad)}")
    if bad:
        print(bad)
        sys.exit(1)


if __name__ == "__main__":
    main()
