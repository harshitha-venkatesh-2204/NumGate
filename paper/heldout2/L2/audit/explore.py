from pathlib import Path
import sys, json, re
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import *
ALIAS = {"uk": "United Kingdom", "irish": "EIRE", "dutch": "Netherlands", "danish": "Denmark", "australian": "Australia",
         "uae": "United Arab Emirates", "swedish": "Sweden"}
def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())
def resolve(df, name):
    n = ALIAS.get(name.lower(), name)
    hits = df.index[df.Entity.map(norm) == norm(n)].tolist()
    if hits: return hits
    words = [w for w in re.findall(r"[a-z0-9]+", n.lower())]
    return df.index[df.Entity.map(lambda e: all(w in re.findall(r"[a-z0-9]+", e.lower().replace("/", " ")) for w in words))].tolist()
notes = {n["note_id"]: n for n in json.load(open(B + "notes.json"))}
s = pd.read_csv(B + "audit/sample_blind.csv", index_col=0)
lo, hi = int(sys.argv[1]), int(sys.argv[2])
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
for i, r in s.iloc[lo:hi].iterrows():
    df = load(r.table_id)
    line = notes[r.note_id]["text"].split("\n")[r.line - 1]
    p = parse(r.value_text)
    print(f"\n### row{i} {r.note_id} L{r.line} [{r.stratum}] {r.entity!r} {r.metric}/{r.period} {r.value_text!r} dir={r.direction} q={r.qualifier}")
    print("  LINE:", line)
    idx = resolve(df, r.entity) if r.entity != "GROUP" else []
    for j in idx:
        print("  ROW:", df.loc[j].to_dict())
        d = derived(df, j)
        hits = {k: round(float(v), 4) for k, v in d.items() if match(v, p, True)}
        hits_us = {k: round(float(v), 4) for k, v in d.items() if match(v, p, False) and k not in hits}
        print("  row matches:", hits, " sign-ignored:", hits_us)
    if not idx and r.entity != "GROUP": print("  ENTITY NOT RESOLVED")
    print("  table cell matches:", search(df, p, True)[:8])
