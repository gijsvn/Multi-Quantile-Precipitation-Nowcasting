# Ablation: where do the multi-quantile gains come from? (R2 / editor)

## What the paper already has
- MSE, MAE and full quantile models, each trained 5x, but only the **best-of-5 by validation loss** is reported on the test set (one number per model, no spread). The repo holds only those three best checkpoints.
- Fig. 3 grid search over w0.90 = w0.95, including w = 0, but on **validation** median MSE only.

## What is missing (new runs)
| Variant | Channels | train.py args | Why |
|---|---|---|---|
| `mae` | 12 | `--loss l1` | baseline, now with mean ± std |
| `median_pinball` | 12 | `--loss quantile --quantiles 0.5` | = ½·L1; confirms it matches MAE |
| `w0` | 36 | `--quantiles 0.5 0.9 0.95 --higher-q-weight 0` | extra channels, no upper-quantile gradient, now on **test** |
| `median_x3` | 36 | `--quantiles 0.5 0.5 0.5 --higher-q-weight 0.5` | extra heads + extra gradient, but **no asymmetry** |
| `full` | 36 | `--quantiles 0.5 0.9 0.95 --higher-q-weight 0.5` | paper model, now with mean ± std |

5 seeds each (0–4) = 25 runs. `median_x3` keeps the full model's weights (1, 0.5, 0.5), so the only difference from `full` is the quantile levels. An MSE line is commented out in the script if you also want a multi-seed MSE row (`--array=0-29`).

Reading the table: `w0 ≈ mae` means channels alone don't help; `median_x3 ≈ full` would mean the gain is just extra gradient/heads; `full` better than `median_x3` means the upper-quantile (asymmetric) signal is what helps.

**No repo code changes are needed**: the loss already accepts repeated quantiles, `--higher-q-weight 0` works, `--seed` exists, and `eval.py` picks head 0 (the q=0.5 head) for all 36-channel models.

## Files
- `run_ablation.slurm`: SLURM array (25 tasks): train, copy best checkpoint + config, evaluate on test.
- `aggregate_ablation.py`: builds the table (mean ± std over seeds, plus best-of-5 by val loss to match the paper's protocol). Writes `ablation_summary.csv/.md`, `ablation_best_of_seeds.csv`, `ablation_all_runs.csv`.

## Running on Snellius
1. Fill in the `PLACEHOLDER` lines (repo path, dataset path, output dir on project space, venv, modules, partition, time, account).
2. One-time env setup on a login node:
   ```bash
   module load 2024 Python/3.12.3-GCCcore-13.3.0   # adjust to `module avail`
   python -m venv ~/venvs/mqpn && source ~/venvs/mqpn/bin/activate
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
   ```
3. From the directory holding the script: `mkdir -p logs && sbatch run_ablation.slurm`.
   Tip: first try one task (`sbatch --array=24 run_ablation.slurm`) to check paths and wall time.
4. When all tasks are done: `python aggregate_ablation.py --root <OUT_ROOT>`.

Tested locally end to end (all 5 variants train, evaluate and aggregate) on a tiny synthetic dataset with 1 epoch, CPU only.
