"""Lane calibrated-4bit: build arms B and C from cached packs (no download, no bf16 load). CPU only.

  python3 scripts/calib4_build.py OUT_DIR      # writes OUT_DIR/{gemma-4-e4b-it-qat-4bit-requant,gemma-4-e4b-it-splice-mlp8}
                                              # and prints a manifest (sha256 of every output file) as JSON on stdout

B  qat-4bit-requant: mlx-community/gemma-4-E4B-it-qat-4bit with its 126 8-bit MLP weights (gate/up/down x 42 layers)
   dequantised and re-quantised by mx.quantize at 4 bits, group 64 (affine); the 126 per-layer overrides removed from the
   config. Every other tensor and file is the QAT pack's, byte for byte.
C  splice-mlp8: mlx-community/gemma-4-e4b-it-4bit with the same 126 MLP weights (weight/scales/biases) taken from
   mlx-community/gemma-4-e4b-it-8bit, and the QAT pack's 126 overrides (8 bits, group 64) added to the config. Every other
   tensor and file is the plain 4-bit pack's.
"""
import glob, hashlib, json, os, re, shutil, sys
import mlx.core as mx

mx.set_default_device(mx.cpu)
HUB = os.path.expanduser("~/.cache/huggingface/hub/")
SNAP = {"qat": "gemma-4-E4B-it-qat-4bit/snapshots/0f35c6f6d386f7f74e628bd7c6526ce531212300",
        "q4": "gemma-4-e4b-it-4bit/snapshots/475b9088d29754a3379866cf5aeb6b41acd313c2",
        "q8": "gemma-4-e4b-it-8bit/snapshots/4255b21bd9a9d3fc807ef7abd80373f5e3a52a73"}
MLP = re.compile(r"^language_model\.model\.layers\.\d+\.mlp\.(gate|up|down)_proj$")


def src(k):
    return HUB + "models--mlx-community--" + SNAP[k] + "/"


def tensors(k):
    out = {}
    for f in sorted(glob.glob(src(k) + "*.safetensors")):
        out.update(mx.load(f))
    return out


def copy_side_files(k, dst):
    for f in os.listdir(src(k)):
        if not f.endswith(".safetensors") and f not in ("config.json", "model.safetensors.index.json"):
            shutil.copyfile(src(k) + f, dst + f)


def write(dst, ts, cfg):
    mx.save_safetensors(dst + "model.safetensors", ts, metadata={"format": "mlx"})
    total = sum(v.nbytes for v in ts.values())
    json.dump({"metadata": {"total_size": total}, "weight_map": {k: "model.safetensors" for k in sorted(ts)}},
              open(dst + "model.safetensors.index.json", "w"), indent=2)
    json.dump(cfg, open(dst + "config.json", "w"), indent=2)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


out_root = sys.argv[1].rstrip("/") + "/"
qcfg = json.load(open(src("qat") + "config.json"))
overrides = {k: v for k, v in qcfg["quantization"].items() if isinstance(v, dict)}
assert len(overrides) == 126 and all(MLP.match(k) and v == {"group_size": 64, "bits": 8} for k, v in overrides.items())
manifest = {}

# B
dst = out_root + "gemma-4-e4b-it-qat-4bit-requant/"
os.makedirs(dst, exist_ok=True)
ts = tensors("qat")
for p in overrides:
    w = mx.dequantize(ts[p + ".weight"], ts[p + ".scales"], ts[p + ".biases"], group_size=64, bits=8)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    assert s.dtype == ts[p + ".scales"].dtype
    ts[p + ".weight"], ts[p + ".scales"], ts[p + ".biases"] = wq, s, b
cfg = json.loads(json.dumps(qcfg))
for key in ("quantization", "quantization_config"):
    cfg[key] = {k: v for k, v in cfg[key].items() if not isinstance(v, dict)}
copy_side_files("qat", dst)
write(dst, ts, cfg)
del ts

# C
dst = out_root + "gemma-4-e4b-it-splice-mlp8/"
os.makedirs(dst, exist_ok=True)
ts, t8 = tensors("q4"), tensors("q8")
for p in overrides:
    for s in (".weight", ".scales", ".biases"):
        ts[p + s] = t8[p + s]
cfg = json.load(open(src("q4") + "config.json"))
for key in ("quantization", "quantization_config"):
    cfg[key].update(overrides)
copy_side_files("q4", dst)
write(dst, ts, cfg)
del ts, t8

for d in sorted(glob.glob(out_root + "*/")):
    manifest[os.path.basename(d.rstrip("/"))] = {f: sha(d + f) for f in sorted(os.listdir(d))}
print(json.dumps({"sources": SNAP, "packs": manifest}, indent=2))
