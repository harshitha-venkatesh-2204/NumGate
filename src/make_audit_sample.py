"""Sample numbers and claims for human audit of the gate.

Usage: python src/make_audit_sample.py [--run-id full] [--n 200]
Writes results/audit_sample.csv with an empty human_label column. Fill it with the same
label vocabulary as gate_label, then run src/score_audit.py.
"""
import argparse

import pandas as pd

from config import RESULTS_DIR, RUNS_DIR, load_config


def stratified(df, n, seed):
    """About n rows spread evenly over strategy x label cells, topping up from the rest if cells are small."""
    groups = list(df.groupby(["strategy", "label"]))
    per_cell = max(1, n // max(len(groups), 1))
    picked = pd.concat([g.sample(min(len(g), per_cell), random_state=seed) for _, g in groups])
    rest = df.drop(picked.index)
    if len(picked) < n and len(rest):
        picked = pd.concat([picked, rest.sample(min(len(rest), n - len(picked)), random_state=seed)])
    return picked.head(n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="full")
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()
    seed = load_config()["seed"]
    run_dir = RUNS_DIR / args.run_id

    numbers = pd.read_csv(run_dir / "numbers.csv")
    numbers = numbers[numbers["label"] != "Skipped"]
    claims = pd.read_csv(run_dir / "claims.csv")

    num_s = stratified(numbers, args.n, seed)
    num_rows = pd.DataFrame({
        "item_type": "number", "note_id": num_s["note_id"], "table_id": num_s["table_id"],
        "strategy": num_s["strategy"], "model": num_s["model"], "line_idx": num_s["line_idx"],
        "text": num_s["sentence"], "item": num_s["raw_text"], "gate_label": num_s["label"],
        "evidence": num_s["evidence"]})
    cl_s = stratified(claims, args.n, seed)
    claim_rows = pd.DataFrame({
        "item_type": "claim", "note_id": cl_s["note_id"], "table_id": cl_s["table_id"],
        "strategy": cl_s["strategy"], "model": cl_s["model"], "line_idx": cl_s["line_idx"],
        "text": "", "item": cl_s["entity"] + " | " + cl_s["metric"] + " | " + cl_s["period"].astype(str)
        + " | " + cl_s["value_text"].astype(str), "gate_label": cl_s["label"], "evidence": cl_s["evidence"]})
    out = pd.concat([num_rows, claim_rows], ignore_index=True)
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle so auditors do not see label runs
    out["human_label"] = ""
    RESULTS_DIR.mkdir(exist_ok=True)
    out.to_csv(RESULTS_DIR / "audit_sample.csv", index=False)
    print(f"Wrote results/audit_sample.csv: {len(num_rows)} numbers, {len(claim_rows)} claims")
    print("Number labels: Supported_cell, Supported_derived, Unsupported. "
          "Claim labels: Correct, Wrong_entity, Wrong_metric, Wrong_value, Unverifiable.")


if __name__ == "__main__":
    main()
