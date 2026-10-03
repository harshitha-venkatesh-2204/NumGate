# Auditor sample: 70 digit-form measure mentions (label C or W), seed 20260928,
# stratified: 20 from recorded-wrong (W), 50 from recorded-correct (C).
from pathlib import Path
import json, pandas as pd
base = str(Path(__file__).resolve().parents[1]) + "/"
t = pd.read_csv(base + "truth.csv")
t["row_id"] = t.index
pop = t[(t.is_word == 0) & (t.label.isin(["C", "W"]))]
W = pop[pop.label == "W"].sample(n=20, random_state=20260928)
C = pop[pop.label == "C"].sample(n=50, random_state=20260928)
s = pd.concat([W, C]).sort_values(["note_id", "line", "idx"])
s.to_csv(base + "audit/sample.csv", index=False)
notes = {json.loads(l)["note_id"]: json.loads(l) for l in open(base + "notes.jsonl")}
for _, r in s.iterrows():
    line = notes[r.note_id]["text"].split("\n")[r.line - 1]
    print(f"[{r.row_id}] {r.note_id} {r.table_id} L{r.line} idx{r.idx} '{r.value_text}' prec={r.precision} qual={r.qualifier}")
    print("    ", line)
