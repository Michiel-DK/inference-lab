"""Load the cached Gemma 4 e2b 4-bit model, generate greedily with and without the patch, compare text and tok/s.

Usage: python3 scripts/smoke_gemma.py [max_tokens]
Needs the model in the HF cache (mlx-community/gemma-4-e2b-it-4bit). Not a benchmark: one run, one machine.
"""
import sys, time
import mlx.core as mx
from mlx_lm import load, generate
from mk.geglu import patch_gemma4

MODEL = "mlx-community/gemma-4-e2b-it-4bit"
PROMPT = "Explain in two sentences what a GPU kernel is."


def run(model, tok, max_tokens):
    msgs = [{"role": "user", "content": PROMPT}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
    t = time.perf_counter()
    text = generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False)
    return text, time.perf_counter() - t


def main():
    max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    model, tok = load(MODEL)
    base, tb = run(model, tok, max_tokens)
    base, tb = run(model, tok, max_tokens)  # second run = warm
    restore = patch_gemma4()
    try:
        patched, tp = run(model, tok, max_tokens)
        patched, tp = run(model, tok, max_tokens)
    finally:
        restore()
    print("--- stock ---\n" + base)
    print("--- patched ---\n" + patched)
    print(f"\nidentical text: {base == patched}")
    print(f"stock {tb:.2f}s · patched {tp:.2f}s  (one run each after warm-up, {max_tokens} max tokens; NOT evidence)")


if __name__ == "__main__":
    main()
