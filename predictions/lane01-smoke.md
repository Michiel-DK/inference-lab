# lane01-smoke

Lane 01 (6 Nov 2026): skeleton and metering. Wiring only; no number from this run is posted.

## Prediction
`lab run mlx-community/Qwen2.5-Coder-1.5B-bf16 --name lane01-smoke --smoke` completes both fixed email-triage TRAIN cases
(`client-quote-request`, `overdue-invoice`) end to end: wilco grades each one (PASS or FAIL, either is fine), every case has
at least one metered call with tokens/s, peak memory and wall time, and Wh is recorded as not metered (no capture,
no sudo). `lab check results/lane01-smoke.json` exits 0. On the verdicts and the speed: unsure, and neither is a claim.

Kill clause: if a case lands in wilco's `error` bucket because the adapter's meta does not satisfy wilco (not because the
model's JSON was bad), or if wilco's tree is dirtied by the import, the wiring is wrong and the lane does not post.

## Mechanism
The adapter returns the `(text, meta)` pair wilco's `call_model` expects, and `run_exam` is called once per case with
`adapter_for` swapped. So prompt, parse, re-ask and the labels scorer are wilco's at the pin, and only generation moves
in-process. A 1.5B bf16 base coder model is the smallest model cached here that mlx-lm loads with a chat template. It may
not emit the label JSON, and that would be a FAIL, not a wiring fault.
