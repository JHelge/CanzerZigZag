# zigzag (safe refactor)

This package is a **safe Phase-1 refactor** of your original `zigzag_full_pipeline.py`:
- No control-flow was split in unsafe positions.
- The run-stage is preserved exactly.
- Posthoc can be re-run via `posthoc.py` after a completed run.

## Run full pipeline
python run.py [all original args...]

## Run only compute stage (skip posthoc at end)
python run.py ... --skip_posthoc

## Re-run posthoc later
python posthoc.py --outdir /path/to/outdir

Posthoc re-run expects these files in `--outdir` (saved automatically during run stage):
- `real_h_eval.h5ad`
- `real_t_eval.h5ad`
- `summary_all_runs.csv`
# CanzerZigZag
