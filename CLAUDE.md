# inference-lab

One area: inference on this laptop under mlx-lm, graded by wilco's exams, metered per case. Read `SHOWCASE.md`,
`LESSONS.md` and the newest `docs/inbox/*-restart.md` before changing any method. The Metal kernel work (`mk/`, `bench/`, `tests/test_geglu.py`) predates the lab
and keeps its own rules (README, bottom half).

## Bearings

```sh
P=~/.pyenv/versions/lewagon/bin/python3     # the env with mlx 0.32.2 / mlx-lm 0.31.3 (requirements.txt pins)
$P -m pytest -q                             # the lab gate (tests/lab; no GPU, no model)
$P -m lab check results/<run>.json          # re-derives every posted aggregate; exit 1 on drift
$P -m lab diff <a> <b>
$P -m lab run <model> --name <run> --smoke  # needs a committed predictions/<run>.md and the wilco pin; never overwrites
$P -m lab energy results/<run>.json --capture F --idle-start ISO --idle-end ISO --out results/<run>+energy.json
```

## Rules

- `results/*.json` are committed evidence; a lane reads them and never regenerates them.
- A number is posted only with the command that re-derives it.
- No hosted model calls. mlx-lm on this machine only, cached models only (never download inside a run).
- wilco is read-only and pinned (`docs/CASES.md`); moving the pin is its own commit with the new hashes.
- Wh comes only from an operator-started `sudo powermetrics` capture (`lab/energy.py`); without one it is `null`, never 0.
- Work in a worktree from `origin/main`; open a PR; never merge from a lane.
