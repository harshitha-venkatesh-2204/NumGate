"""Turn the blind-review workflow output into one verdict per item.

Usage: python paper/adjudication/collect_verdicts.py <workflow_result.json> [--run-id full]

The final verdict is the shared verdict when the two independent reviewers agree, otherwise the third reviewer's.
Writes paper/adjudication/verdicts_<run_id>.csv (joined to the item key) and reviews_<run_id>.csv (every review).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--run-id", default="full")
    args = ap.parse_args()
    raw = json.loads(Path(args.result).read_text())
    raw = raw.get("result", raw)
    key = pd.DataFrame(json.loads((HERE / f"key_{args.run_id}.json").read_text()))
    rows, reviews = {}, []
    for b in raw["results"]:
        if b.get("error"):
            raise SystemExit(f"batch {b['batch']} incomplete: {b['error']}")
        a = {x["item_id"]: x for x in b["a"]}
        bb = {x["item_id"]: x for x in b["b"]}
        t = {x["item_id"]: x for x in b.get("third", [])}
        for iid in a:
            va, vb = a[iid]["verdict"], bb.get(iid, {}).get("verdict")
            final = va if va == vb else t.get(iid, {}).get("verdict")
            rows[iid] = {"item_id": iid, "verdict_a": va, "verdict_b": vb, "verdict_third": t.get(iid, {}).get("verdict"),
                         "final": final, "agree": va == vb}
            for who, src in (("a", a), ("b", bb), ("third", t)):
                if iid in src:
                    reviews.append({"item_id": iid, "reviewer": who, **{k: src[iid].get(k) for k in ("verdict", "reading", "expected", "reason")}})
    v = pd.DataFrame(rows.values())
    missing = sorted(set(key.item_id) - set(v.item_id))
    unresolved = v[v.final.isna()].item_id.tolist()
    if missing or unresolved:
        raise SystemExit(f"{len(missing)} items without reviews, {len(unresolved)} without a final verdict: {(missing + unresolved)[:10]}")
    out = key.merge(v, on="item_id")
    out.to_csv(HERE / f"verdicts_{args.run_id}.csv", index=False)
    pd.DataFrame(reviews).to_csv(HERE / f"reviews_{args.run_id}.csv", index=False)
    print(f"{len(out)} items; first two reviewers agreed on {out.agree.mean():.1%}; final: {out.final.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
