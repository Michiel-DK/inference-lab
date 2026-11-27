"""Lane 15 worker: a Runpod serverless handler around an MLX model, run in the SDK's local mode.

    LANE15_MODEL=<local model dir> LANE15_SPAWN_T=<driver time.time()> HF_HUB_OFFLINE=1 \
        ~/.venvs/lab-inf-15/bin/python scripts/lane15_worker.py --rp_serve_api --rp_api_port 18100

The model is loaded at module level, as a Runpod handler does. Every stamp is (time.time(), time.perf_counter()); the driver
reads the cross-process stages off time.time() and the in-process ones off perf_counter(). The handler returns all stamps,
so the driver writes one row per trial. Greedy only, max 64 new tokens; nothing is graded here.
"""
import time

STAMPS = {"worker_first_line": (time.time(), time.perf_counter())}


def _stamp(name):
    STAMPS[name] = (time.time(), time.perf_counter())


import os  # noqa: E402

import runpod  # noqa: E402

_stamp("import_runpod_done")
import mlx.core as mx  # noqa: E402
import mlx_lm  # noqa: E402
from mlx_lm import load, stream_generate  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

_stamp("imports_done")
MODEL, TOKENIZER = load(os.environ["LANE15_MODEL"])   # default lazy=False: mx.eval(model.parameters()) inside
_stamp("weights_loaded")
REQUESTS = 0


def handler(job):
    global REQUESTS
    REQUESTS += 1
    s = {"handler_in": (time.time(), time.perf_counter())}
    inp = job["input"]
    prompt = TOKENIZER.apply_chat_template(inp["messages"], add_generation_prompt=True)
    s["templated"] = (time.time(), time.perf_counter())
    first, last, parts = None, None, []
    for resp in stream_generate(MODEL, TOKENIZER, prompt, max_tokens=int(inp.get("max_tokens", 64)),
                                sampler=make_sampler(temp=0.0)):
        if first is None:
            first = (time.time(), time.perf_counter())
        parts.append(resp.text)
        last = resp
    s["first_token"] = first
    s["last_token"] = (time.time(), time.perf_counter())
    return {
        "request_index": REQUESTS,
        "pid": os.getpid(),
        "spawn_t": float(os.environ.get("LANE15_SPAWN_T", "nan")),
        "module_stamps": STAMPS,
        "stamps": s,
        "prompt_tokens": last.prompt_tokens,
        "prompt_tps": float(last.prompt_tps),
        "gen_tokens": last.generation_tokens,
        "gen_tps": float(last.generation_tps),
        "finish_reason": last.finish_reason,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,     # process peak since start (load included); never reset here
        "active_memory_gb": mx.get_active_memory() / 1e9,
        "text": "".join(parts),
        "versions": {"mlx": mx.__version__, "mlx_lm": mlx_lm.__version__, "runpod": runpod.__version__},
    }


runpod.serverless.start({"handler": handler})
