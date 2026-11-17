"""Effective decode bandwidth per pack: weight bytes read per generated token x measured gen tok/s (stdlib only).

Bytes read per token = the text model's weights (language_model.*) minus the per-layer embedding table (embed_tokens_per_layer:
one row looked up per token, not streamed) plus embed_tokens once (tied to the output head, so the whole matrix is read per token).
Vision and audio towers are not read on text decode. Not counted: KV-cache reads, the per-layer rows, activations, kernel launches.
B has the same tensor dtypes and shapes as plain 4-bit (docs/probes/calib4/prov.txt [1]), so it reads the same bytes.

  python3 scripts/calib4_bandwidth.py
"""
import glob, json, os, struct

HUB = os.path.expanduser("~/.cache/huggingface/hub/")
SRC = json.load(open("docs/probes/calib4/packs.json"))["sources"]
PACKS = [  # (arm, snapshot key in packs.json, results file)
    ("plain 4-bit", "q4", "results/dense-gemma-4-e4b-it-4bit.json"),
    ("B. QAT, pure 4-bit g64", "q4", "results/calib4-gemma-4-e4b-it-qat-4bit-requant.json"),
    ("A. QAT pack (4 attn / 8 MLP)", "qat", "results/calib4-gemma-4-e4b-it-qat-mixed.json"),
    ("plain 8-bit", "q8", "results/quant-gemma-4-e4b-it-8bit.json"),
]


def read_bytes(snap):
    d = HUB + "models--mlx-community--" + snap
    total = 0
    for f in sorted(glob.glob(d + "/*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            head = json.loads(fh.read(n))
        for k, v in head.items():
            if k == "__metadata__" or not k.startswith("language_model.") or "embed_tokens_per_layer" in k:
                continue
            a, b = v["data_offsets"]
            total += b - a
    return total


print("| arm | weight GB read / token | gen tok/s | effective GB/s | ms / token |")
print("|---|---|---|---|---|")
for arm, key, res in PACKS:
    gb = read_bytes(SRC[key]) / 1e9
    tps = json.load(open(res))["aggregates"]["gen_tps"]
    print(f"| {arm} | {gb:.2f} | {tps:.1f} | {gb * tps:.0f} | {1000 / tps:.1f} |")
