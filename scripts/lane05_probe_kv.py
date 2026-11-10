"""Lane 05 probe (not graded): which KV-cache settings mlx-lm 0.31.3 actually APPLIES on each lane model.

For each model: one ~N-token prompt built from wilco train inputs (no held-out text), greedy 32 tokens, under
default / kv_bits=8 / kv_bits=4 / max_kv_size=1024. Prints the cache classes make_prompt_cache returns with and
without max_kv_size, whether each setting raised, and whether its tokens equal the default's.
Usage: python3 scripts/lane05_probe_kv.py <model ref> [target_tokens]
"""
import json, sys, traceback
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import cases as C
from lab.run import resolve_model

ref = sys.argv[1]; target = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
path, snap = resolve_model(ref)
model, tok = load(path)
train = [c["input"] for c in json.loads(C.cases_path("recap").read_text())["cases"] if c["split"] == "train"]
text, i = "", 0
while len(tok.encode(text)) < target:
    text += train[i % len(train)] + "\n\n"; i += 1
msgs = [{"role": "system", "content": "Summarise the items below in three bullet points."}, {"role": "user", "content": text}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
out = {"model": ref, "snapshot": snap, "prompt_tokens": len(prompt)}
classes = lambda cs: sorted({type(c).__name__ for c in cs})
out["cache_default"] = classes(make_prompt_cache(model))
out["cache_max_kv_size_1024"] = classes(make_prompt_cache(model, max_kv_size=1024))
out["has_make_cache"] = hasattr(model, "make_cache")
arms = {"default": {}, "kv_bits_8": {"kv_bits": 8}, "kv_bits_4": {"kv_bits": 4}, "max_kv_size_1024": {"max_kv_size": 1024}}
toks = {}
for name, kw in arms.items():
    try:
        mx.reset_peak_memory(); ids, last = [], None
        for r in stream_generate(model, tok, prompt, max_tokens=32, sampler=make_sampler(temp=0.0), **kw):
            ids.append(r.token); last = r
        toks[name] = ids
        out[name] = {"ok": True, "prompt_tps": round(last.prompt_tps, 1), "gen_tps": round(last.generation_tps, 1),
                     "peak_gb": round(last.peak_memory, 3), "same_tokens_as_default": ids == toks.get("default")}
    except Exception as e:
        out[name] = {"ok": False, "exception": f"{type(e).__name__}: {e}",
                     "where": traceback.extract_tb(e.__traceback__)[-1].filename.split("site-packages/")[-1]}
print(json.dumps(out, indent=1))
