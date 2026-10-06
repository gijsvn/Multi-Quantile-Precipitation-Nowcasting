# Ablation: where do the multi-quantile gains come from? (R2 / editor)

## What the paper already has
- MSE, MAE and full quantile models, each trained 5x, but only the **best-of-5 by validation loss** is reported on the test set (one number per model, no spread). The repo holds only those three best checkpoints.
- Fig. 3 grid search over w0.90 = w0.95, including w = 0, but on **validation** median MSE only.

## What is missing (new runs)
| Variant | Channels | train.py args | Why |
|---|---|---|---|
| `median_pinball` | 12 | `--loss quantile --quantiles 0.5` | = ½·L1; confirms it matches MAE |
| `median_x3` | 36 | `--quantiles 0.5 0.5 0.5 --higher-q-weight 0.5` | extra heads + extra gradient, but **no asymmetry** |

5 seeds each (0–4) = 10 runs. The **MAE**, **w = 0** (`w0`, 36 channels, no upper-quantile gradient; now evaluated on **test**) and **full model** rows reuse the 5 original runs of each (see "Adding the original runs" below). `median_x3` keeps the full model's weights (1, 0.5, 0.5), so the only difference from `full` is the quantile levels.

Reading the table: `w0 ≈ mae` means channels alone don't help; `median_x3 ≈ full` would mean the gain is just extra gradient/heads; `full` better than `median_x3` means the upper-quantile (asymmetric) signal is what helps.

Caveat: if the original 5 runs all used the default `--seed 42`, their spread only reflects GPU nondeterminism and will look smaller than the new variants' seed spread; mention this in the paper.

**No repo code changes are needed**: the loss already accepts repeated quantiles, `--higher-q-weight 0` works, `--seed` exists, and `eval.py` picks head 0 (the q=0.5 head) for all 36-channel models.

## Files
- `run_ablation.slurm`: SLURM array (10 tasks): train, copy best checkpoint + config, evaluate on test.
- `import_original_runs.slurm`: brings the 5 original MAE, w = 0 and full-model runs into the same results folder.
- `aggregate_ablation.py`: builds the table (mean ± std over seeds, mean ± 95% CI over runs, plus best-of-5 by val loss as in the original paper). Writes `ablation_summary.csv/.md`, `ablation_best_of_seeds.csv`, `ablation_all_runs.csv`.

## Running on Snellius
1. Fill in the `PLACEHOLDER` lines (repo path, dataset path, output dir on project space, venv, modules, partition, time, account).
2. One-time env setup on a login node:
   ```bash
   module load 2024 Python/3.12.3-GCCcore-13.3.0   # adjust to `module avail`
   python -m venv ~/venvs/mqpn && source ~/venvs/mqpn/bin/activate
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
   ```
3. From the directory holding the script: `mkdir -p logs && sbatch run_ablation.slurm`.
   Task numbers: 0–4 `median_pinball`, 5–9 `median_x3`. Run a subset with e.g. `sbatch --array=0-2 run_ablation.slurm`; finished tasks are skipped, so resubmitting the full array resumes.
4. When all tasks are done: `python aggregate_ablation.py --root <OUT_ROOT>`.

## Adding the original runs (MAE, w = 0 and full model)
Fill in the three lists in `import_original_runs.slurm` (5 entries each), then `sbatch import_original_runs.slurm`.
Each entry is either an original run directory (re-evaluated on test with the current `eval.py`; recommended) or an existing `<run>_test_results.json` (copied as is; for quantile runs use the `_q0` = q50 file). Results land in `<OUT_ROOT>/eval/{mae,w0,full}/seed0-4/`, next to the new runs.

## Building the table locally (alternative)
Copy only the result files from Snellius and add your existing five-run results with `--extra`:
```bash
# Snellius
cd <OUT_ROOT> && tar czf ~/ablation_eval.tgz $(find eval -name '*.json' -o -name '*.txt')
# laptop, from hybrid-nowcasting-thesis-v0
scp <user>@snellius.surf.nl:ablation_eval.tgz evaluation_models/ablation_runs/
cd evaluation_models/ablation_runs && tar xzf ablation_eval.tgz && cd ../..
python aggregate_ablation.py --root evaluation_models/ablation_runs \
  --extra "mae=evaluation_models/five_run_aggregates/five_run_loss_experiments/<MAE folder>/*_test_results.json" \
  --extra "full=evaluation_models/five_run_aggregates/five_run_loss_experiments/<quantile folder>/*_q0_test_results.json" \
  --extra "w0=<folder with the w = 0 test results>/*_q0_test_results.json"
```

## Building the table
`python aggregate_ablation.py --root <OUT_ROOT>` prints and saves the table: mean ± 95% CI half-width over the 5 runs (Student t, df = 4, = 1.24 × sample std), the same format as the revised main results table. `ablation_summary.csv` also has the plain std, and `ablation_all_runs.csv` every individual run.

Tested locally end to end (training, import, evaluation and aggregation) on a tiny synthetic dataset with 1 epoch, CPU only.
