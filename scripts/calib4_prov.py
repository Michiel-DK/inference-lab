"""Lane calibrated-4bit: what the packs are, before any generation. CPU only, no generation.

  LAB_WILCO_DIR=<pin clone> python3 scripts/calib4_prov.py PACKS_DIR    # PACKS_DIR as given to scripts/calib4_build.py

1. per-tensor format (dtype/shape) from safetensors headers: C must equal A (the QAT pack), B must equal the plain 4-bit;
2. relative L2 distance of dequantised weights vs the plain 8-bit pack, layers 0/10/20/41, q/o/gate/down;
3. bits / group size that mlx-lm 0.31.3 builds on a lazy load (no weights evaluated), one MLP and one attention module;
4. rendered chat-template token ids of the first prompt of every email-triage train case (wilco's run_plain_case
   messages) per arm, compared with the plain 4-bit pack's.
"""
import glob, json, os, re, struct, sys
import mlx.core as mx

mx.set_default_device(mx.cpu)
HUB = os.path.expanduser("~/.cache/huggingface/hub/")
packs_dir = sys.argv[1].rstrip("/") + "/"
ARMS = {"plain-4bit": glob.glob(HUB + "models--mlx-community--gemma-4-e4b-it-4bit/snapshots/475b*/")[0],
        "plain-8bit": glob.glob(HUB + "models--mlx-community--gemma-4-e4b-it-8bit/snapshots/4255*/")[0],
        "A-qat-mixed": glob.glob(HUB + "models--mlx-community--gemma-4-E4B-it-qat-4bit/snapshots/0f35*/")[0],
        "B-qat-4bit-requant": packs_dir + "gemma-4-e4b-it-qat-4bit-requant/",
        "C-splice-mlp8": packs_dir + "gemma-4-e4b-it-splice-mlp8/"}


def headers(d):
    out = {}
    for f in glob.glob(d + "*.safetensors"):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        out.update({k: (v["dtype"], tuple(v["shape"])) for k, v in h.items() if k != "__metadata__"})
    return out


H = {a: headers(d) for a, d in ARMS.items()}
for a, ref in (("C-splice-mlp8", "A-qat-mixed"), ("B-qat-4bit-requant", "plain-4bit"), ("A-qat-mixed", "plain-4bit")):
    diff = sorted({re.sub(r"\.\d+\.", ".N.", k) for k in set(H[a]) | set(H[ref]) if H[a].get(k) != H[ref].get(k)})
    print(f"[1] headers {a} vs {ref}: {len(H[a])} vs {len(H[ref])} tensors; differing (dtype/shape) groups: {diff}")


def W(arm, key, cache):
    d = ARMS[arm]
    f = d + json.load(open(d + "model.safetensors.index.json"))["weight_map"][key + ".weight"]
    t = cache.setdefault(f, mx.load(f))
    q = json.load(open(d + "config.json"))["quantization"]
    o = q.get(key, q)
    return mx.dequantize(t[key + ".weight"], t[key + ".scales"], t.get(key + ".biases"),
                         group_size=o["group_size"], bits=o["bits"]).astype(mx.float32)


rel = lambda a, b: (mx.linalg.norm(a - b) / mx.linalg.norm(b)).item()
print("[2] rel L2 vs plain-8bit: key | plain-4bit | A | B | C")
for L in (0, 10, 20, 41):
    for p in ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"):
        k, cache = f"language_model.model.layers.{L}.{p}", {}
        e = W("plain-8bit", k, cache)
        print(f"    layers.{L}.{p} | " + " | ".join(
            f"{rel(W(a, k, cache), e):.4f}" for a in ("plain-4bit", "A-qat-mixed", "B-qat-4bit-requant", "C-splice-mlp8")))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lab.run import import_wilco_runner
from mlx_lm.utils import load
runner, _ = import_wilco_runner()
agent = runner.load_agent("email-triage")
cases = [c for c in runner.load_exam("email-triage")["cases"] if c["split"] == "train"]
msgs = {c["id"]: [{"role": "system", "content": runner._with_lessons(agent, agent["system_prompt"], c["input"])},
                  {"role": "user", "content": c["input"]}] for c in cases}
ids = {}
for a, d in ARMS.items():
    model, tok = load(d, lazy=True)
    lm = model.language_model.model.layers[0]
    print(f"[3] {a}: layer0 mlp.gate_proj bits={lm.mlp.gate_proj.bits} gs={lm.mlp.gate_proj.group_size}; "
          f"self_attn.q_proj bits={lm.self_attn.q_proj.bits} gs={lm.self_attn.q_proj.group_size}")
    ids[a] = {c: tok.apply_chat_template(m, add_generation_prompt=True) for c, m in msgs.items()}
    del model
for a in ARMS:
    print(f"[4] {a}: rendered prompts of {len(ids[a])} email-triage train cases identical to plain-4bit: "
          f"{ids[a] == ids['plain-4bit']}")
