"""Wh per case from a macOS `powermetrics` capture. Stdlib only, never starts a process, never needs root.

The method is agent-sandbox's, not new here: `scripts/energy_report.py` at agent-sandbox@2806cbd0 (PR #158), as used for
docs/probes/energy-resnapshot-2026-11-01/RESULTS.md (0.031-0.169 Wh per case, 1 Nov). Same rules, restated:

  capture   the OPERATOR runs, under sudo, around `lab run`:
                sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -f plist -o FILE
            a stream of XML plists separated by NUL bytes.
  power     processor.combined_power (mW) if present, else the sum of the present processor.{cpu,gpu,ane}_power;
            a sample with none of them RAISES (an unknown format must never integrate to 0 J).
  interval  [timestamp - elapsed_ns, timestamp], timestamps read as UTC (plist dates are naive UTC, whole seconds).
  energy    overlap-weighted: power_W x seconds of each sample's interval inside the window.
  coverage  sampled seconds inside the window / window seconds; below 0.9 RAISES for the run window.
  idle      MEDIAN power of samples whose END lies in the idle window (inclusive); net = gross - idle_W x window seconds.
            net is reported beside gross, never instead of it; a negative net is reported as computed.
  per case  run-window net J / 3600 / n_cases  (the sandbox's posted figure). A per-case window of a few seconds at 1-2 s
            sampling with whole-second timestamps is too coarse for its own coverage gate, so per-row Wh is recorded
            only where that row's own coverage clears 0.9, and is never the headline.

Scope (same caveat as the sandbox): package power (CPU+GPU+ANE) only; no display, DRAM, SSD or charger loss. A floor on
what a wall meter would show.
"""
import plistlib
import statistics
from datetime import datetime, timedelta, timezone

METHOD = ("agent-sandbox@2806cbd0 scripts/energy_report.py (PR #158): powermetrics cpu_power,gpu_power,ane_power, "
          "processor.combined_power, overlap-weighted, median idle subtracted, coverage >= 0.9; per case = run net J / n")
TAG = "metered: powermetrics package power (CPU+GPU+ANE), laptop, serial"
PARTS = ("cpu_power", "gpu_power", "ane_power")
MIN_COVERAGE = 0.9


def to_utc(dt):
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)   # replace, never astimezone: naive plist dates are UTC
    return dt.astimezone(timezone.utc)


def parse_iso(s):
    s = s.strip()
    if s[-1:] in ("Z", "z"):
        s = s[:-1] + "+00:00"
    return to_utc(datetime.fromisoformat(s))


def parse_capture(data):
    docs = [plistlib.loads(c.strip()) for c in data.split(b"\x00") if c.strip()]
    if not docs:
        raise ValueError("capture holds no plist samples")
    return docs


def sample_power_mw(doc):
    proc = doc.get("processor") if isinstance(doc.get("processor"), dict) else {}
    if "combined_power" in proc:
        return float(proc["combined_power"])
    present = [p for p in PARTS if p in proc]
    if not present:
        raise ValueError(f"sample carries none of processor.combined_power/{'/'.join(PARTS)}; "
                         f"top-level keys {sorted(doc)}; processor keys {sorted(proc)}")
    return float(sum(proc[p] for p in present))


def read_samples(docs):
    out = []
    for doc in docs:
        for k in ("elapsed_ns", "timestamp"):
            if k not in doc:
                raise ValueError(f"sample missing {k}; keys {sorted(doc)}")
        if not isinstance(doc["timestamp"], datetime):
            raise ValueError(f"sample timestamp is not a datetime: {doc['timestamp']!r}")
        end = to_utc(doc["timestamp"])
        out.append((end - timedelta(microseconds=int(doc["elapsed_ns"]) / 1000), end, sample_power_mw(doc)))
    return out


def integrate(samples, start, end):
    """(joules, coverage, n_samples) of the window [start, end]."""
    start, end = to_utc(start), to_utc(end)
    if end <= start:
        raise ValueError(f"window end <= start ({start.isoformat()} .. {end.isoformat()})")
    covered, joules, n = timedelta(0), 0.0, 0
    for b, e, mw in samples:
        ov = min(e, end) - max(b, start)
        if ov > timedelta(0):
            n += 1
            covered += ov
            joules += mw / 1000.0 * ov.total_seconds()
    return joules, covered.total_seconds() / (end - start).total_seconds(), n


def idle_mw(samples, start, end):
    start, end = to_utc(start), to_utc(end)
    if end <= start:
        raise ValueError("idle window end <= start")
    vals = [mw for _, e, mw in samples if start <= e <= end]
    if not vals:
        raise ValueError("no sample ends inside the idle window")
    return statistics.median(vals)


def energy_block(capture_bytes, rows, idle_start, idle_end):
    """The `energy` block of a results file, from a capture and the rows' own UTC windows."""
    samples = read_samples(parse_capture(capture_bytes))
    run_start = min(parse_iso(r["start_utc"]) for r in rows)
    run_end = max(parse_iso(r["end_utc"]) for r in rows)
    gross_j, coverage, n = integrate(samples, run_start, run_end)
    if n == 0:
        raise ValueError("no sample in the capture overlaps the run window")
    if coverage < MIN_COVERAGE:
        raise ValueError(f"run window coverage {coverage:.3f} < {MIN_COVERAGE}")
    idle = idle_mw(samples, parse_iso(idle_start), parse_iso(idle_end))
    window_s = (run_end - run_start).total_seconds()
    net_j = gross_j - idle / 1000.0 * window_s
    per_row = {}
    for r in rows:
        s, e = parse_iso(r["start_utc"]), parse_iso(r["end_utc"])
        j, cov, _ = integrate(samples, s, e)
        per_row[r["case"]] = (None if cov < MIN_COVERAGE else
                              {"gross_j": j, "net_j": j - idle / 1000.0 * (e - s).total_seconds(), "coverage": cov})
    return {"method": METHOD, "tag": TAG, "run_start_utc": run_start.isoformat(), "run_end_utc": run_end.isoformat(),
            "window_s": window_s, "n_samples": n, "coverage": coverage, "gross_j": gross_j,
            "idle_window_utc": [idle_start, idle_end], "idle_mw": idle, "net_j": net_j, "net_negative": net_j < 0,
            "n_cases": len(rows), "per_row": per_row}
