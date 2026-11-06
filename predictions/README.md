# predictions/

`lab run` writes into `results/` only once this run's prediction is committed here, one file per run name. The gate checks
that `predictions/<run>.md` exists, has both sections below, is tracked by git, and has no uncommitted edits. It then stamps
the commit that last touched the file into the results file as `prediction_commit`.

## Format

```markdown
# <run name>

One line: lane, date, what this run is for.

## Prediction
The number this run will produce: metric (pass_rate, gen_tps, peak_memory_gb_max, wh_per_case), case set (exam + split +
ids or "all"), the baseline it is compared against (a committed results file, by path), and the direction and size you
expect. Say "unsure" when you are. Add a kill clause: the result that would make you drop the method.

## Mechanism
Why. The argument behind the prediction, written before the result exists. Name the source (LITERATURE.md row or radar
verdict) if the run tests someone else's claim.
```

After the run, `LESSONS.md` gets one row per prediction: run, prediction, result as `lab check` prints it, and verdict
(held / refuted / void). A run whose prediction is refuted is kept.
