"""Lane 10: per-row comparison of two results files' per-call gen_tokens (same pack re-sat).

  python3 scripts/lane10_tokens.py quant-gemma-4-e4b-it-8bit champion-e4b-8bit    # results/<RUN>.json
Prints rows compared and rows whose per-call gen_tokens lists are identical (key: load, exam, case id)."""
import json
import sys


def calls_by_row(run):
    d = json.load(open(f"results/{run}.json"))
    return {(r.get("load"), r.get("exam"), r.get("case_id") or r.get("case")): [c["gen_tokens"] for c in r["calls"]]
            for r in d["rows"]}


a, b = (calls_by_row(r) for r in sys.argv[1:3])
same = sum(a[k] == b.get(k) for k in a)
print(f"{sys.argv[1]} vs {sys.argv[2]}: rows {len(a)} / {len(b)}; identical per-call gen_tokens {same} of {len(a)}")
