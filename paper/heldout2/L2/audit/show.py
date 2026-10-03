from pathlib import Path
import json, pandas as pd, sys
B = str(Path(__file__).resolve().parents[1]) + "/"
T = str(Path(__file__).resolve().parents[4] / "data" / "tables") + "/"
notes = {n["note_id"]: n for n in json.load(open(B + "notes.json"))}
s = pd.read_csv(B + "audit/sample_blind.csv", index_col=0)
lo, hi = int(sys.argv[1]), int(sys.argv[2])
for i, r in s.iloc[lo:hi].iterrows():
    n = notes[r.note_id]
    lines = n["text"].split("\n")
    print(f"### row{i} {r.note_id} {r.table_id} c{r.claim_idx} L{r.line} [{r.stratum}]")
    print("  LINE:", lines[r.line - 1])
    print(f"  CLAIM: ent={r.entity!r} met={r.metric} per={r.period} val={r.value_text!r} dir={r.direction} q={r.qualifier}")
