from pathlib import Path
import json, pandas as pd
base = str(Path(__file__).resolve().parents[4]) + "/"
s = pd.read_csv(base + "paper/heldout2/L1/audit/sample.csv")
notes = {json.loads(l)["note_id"]: json.loads(l) for l in open(base + "paper/heldout2/L1/notes.jsonl")}
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
seen = set()
for _, r in s.iterrows():
    key = (r.note_id, r.line)
    if key in seen: continue
    seen.add(key)
    tb = pd.read_csv(base + f"data/tables/{r.table_id}.csv")
    line = notes[r.note_id]["text"].split("\n")[r.line - 1]
    hits = tb[tb.Entity.apply(lambda e: e in line or e.lower() in line.lower())]
    print(f"### {r.note_id} L{r.line} ({r.table_id}, {len(tb)} rows): {line}")
    print(hits.to_string(index=False))
