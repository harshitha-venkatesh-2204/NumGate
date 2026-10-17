"""Build the blind review items for the full study (see paper/preregistration.md, outcomes 3 and 4).

Usage: python paper/adjudication/build_items.py [--run-id full] [--batch 50]

One item is one number mention in a note. Items are:
  * every mention the gate flags (Level 1 Unsupported, or the value of a Level 2 wrong claim), and
  * a seeded random sample of unflagged mentions: 300 the gate supports at Level 1 (100 per model) and
    200 that are the values of claims the gate marks Correct (stratified by model), for the miss rate.
Reviewers get only the note, the marked number and the table path, shuffled together, so they are blind to
the model, the strategy and the gate's verdict. The key that maps items back is written separately.
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import dashboard  # noqa: E402

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="full")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--out", default=str(HERE / "items"))
    args = ap.parse_args()
    R = dashboard.run_payload(ROOT / "runs" / args.run_id)
    notes, nums, flags, texts = R["notes"], R["numbers"], R["flags"], R["texts"]
    claims_by_line = {}
    for c in R["claims"]:
        claims_by_line.setdefault((c[0], c[1]), []).append(c)
    rng = random.Random(SEED)
    flagged = [i for i in range(len(nums)) if flags[i]]
    models = sorted({n["model"] for n in notes})
    supported = {m: [i for i in range(len(nums)) if not flags[i] and nums[i][3] in ("Supported_cell", "Supported_derived")
                     and notes[nums[i][0]]["model"] == m] for m in models}
    sample_l1 = []
    for m in models:
        pool = supported[m]
        sample_l1 += rng.sample(pool, min(100, len(pool)))
    taken = set(sample_l1)
    keys = dashboard.number_keys(nums)
    correct_claim_nums = {m: [] for m in models}
    for c in R["claims"]:
        if c[6] != "Correct":
            continue
        k = dashboard.claim_number(keys, c[0], c[1], c[5], set())
        if k is None:
            continue
        i = keys.index(k)
        if flags[i] or i in taken:
            continue
        correct_claim_nums[notes[c[0]]["model"]].append(i)
    sample_l2, per_model = [], {m: 200 // len(models) + (1 if j < 200 % len(models) else 0) for j, m in enumerate(models)}
    for m in models:
        pool = sorted(set(correct_claim_nums[m]))
        pick = rng.sample(pool, min(per_model[m], len(pool)))
        sample_l2 += pick
        taken.update(pick)
    chosen = [(i, "flag") for i in flagged] + [(i, "sample_l1") for i in sample_l1] + [(i, "sample_l2") for i in sample_l2]
    rng.shuffle(chosen)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blind, key = [], []
    for k, (i, kind) in enumerate(chosen):
        x = nums[i]
        n = notes[x[0]]
        item_id = f"it{k + 1:04d}"
        lines = texts[x[0]]
        blind.append({"item_id": item_id, "table_csv": f"data/tables/{n['table']}.csv",
                      "note_lines": [f"{j + 1}: {t}" for j, t in enumerate(lines)],
                      "line": x[1] + 1, "number": x[2], "char_offset_in_line": x[6]})
        key.append({"item_id": item_id, "kind": kind, "note_id": n["id"], "table_id": n["table"], "model": n["model"],
                    "strategy": n["strategy"], "size": n["size"], "repeat": n["repeat"], "line_idx": x[1], "raw_text": x[2],
                    "position": x[6], "gate_label": x[3], "gate_evidence": x[4], "gate_flag": int(flags[i]),
                    "claim_labels": sorted({c[6] for c in claims_by_line.get((x[0], x[1]), []) if c[5] == x[2] or dashboard.digits(c[5]) == dashboard.digits(x[2])})})
    batches = [blind[j:j + args.batch] for j in range(0, len(blind), args.batch)]
    for b, items in enumerate(batches):
        (out / f"batch_{b + 1:03d}.json").write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
    (HERE / f"key_{args.run_id}.json").write_text(json.dumps(key, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(blind)} items ({len(flagged)} flagged, {len(sample_l1)} supported sample, {len(sample_l2)} correct-claim sample) "
          f"in {len(batches)} batches of up to {args.batch}; key: {HERE / f'key_{args.run_id}.json'}")


if __name__ == "__main__":
    main()
