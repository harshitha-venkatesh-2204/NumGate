"""Score the gate against human labels in results/audit_sample.csv.

Usage: python src/score_audit.py
Reports precision and recall per label, and for the binary error-versus-fine decision.
Writes results/audit_scores.csv.
"""
import pandas as pd

from config import RESULTS_DIR

ERROR_LABELS = {"Unsupported", "Wrong_entity", "Wrong_metric", "Wrong_value"}


def per_label(df):
    rows = []
    for label in sorted(set(df["gate_label"]) | set(df["human_label"])):
        gate, human = df["gate_label"] == label, df["human_label"] == label
        tp = (gate & human).sum()
        rows.append({"label": label, "n_gate": gate.sum(), "n_human": human.sum(),
                     "precision": tp / gate.sum() if gate.sum() else float("nan"),
                     "recall": tp / human.sum() if human.sum() else float("nan")})
    return rows


def main():
    df = pd.read_csv(RESULTS_DIR / "audit_sample.csv", dtype={"human_label": str})
    df = df[df["human_label"].fillna("").str.strip() != ""].copy()
    if df.empty:
        print("No human labels yet. Fill the human_label column of results/audit_sample.csv first.")
        return
    df["human_label"] = df["human_label"].str.strip()
    rows = []
    for item_type, g in df.groupby("item_type"):
        for r in per_label(g):
            rows.append({"item_type": item_type, **r})
        binary = g.assign(gate_label=g["gate_label"].isin(ERROR_LABELS).map({True: "ERROR", False: "OK"}),
                          human_label=g["human_label"].isin(ERROR_LABELS).map({True: "ERROR", False: "OK"}))
        for r in per_label(binary):
            rows.append({"item_type": f"{item_type}_binary", **r})
        print(f"{item_type}: {len(g)} labelled, agreement {(g['gate_label'] == g['human_label']).mean():.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "audit_scores.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
