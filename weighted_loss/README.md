# Weighted-loss baseline: B-MSE and B-MAE (R2 / editor)

SmaAt-UNet trained with the balanced losses of Shi et al. (2017): each pixel's squared (B-MSE) or absolute (B-MAE) error is weighted by the class of its **ground-truth** rain rate.

| rain rate (mm/h) | < 2 | 2–5 | 5–10 | 10–30 | ≥ 30 |
|---|---|---|---|---|---|
| weight | 1 | 2 | 5 | 10 | 30 |

The thresholds are divided by 47.83 × 12 = 573.96 to get the normalized scale of the data (`util/balanced_loss.py`), so the error itself is on the same scale as the MSE/MAE runs. Reduction is `sum` divided by batch size, as for the other losses.

## Fairness
Everything except the loss is identical to the paper's MSE/MAE/quantile runs: same data file and rainy-sample filtering, same validation split, SmaAt-UNet with 12 output channels, Adam 1e-3, ReduceLROnPlateau (patience 4), early stopping at 15, best-validation checkpoint, batch 16. Evaluation computes the same metrics as the thesis `util/evaluate_model.py` (checked against this repo's `eval.py`: identical up to float rounding).

Caveats to state in the paper:
- Seeds 0–4 are used. The original five runs probably all used the fixed seed 42, so their spread reflects GPU nondeterminism only and may look smaller.
- Like the MSE/MAE models, each model selects its checkpoint on its own (weighted) validation loss.

## Files
- `run_weighted_loss.slurm`: SLURM array, 10 tasks. **0–4 B-MSE, 5–9 B-MAE** (seeds 0–4). Trains, copies the best checkpoint and evaluates it on the test set.
- `eval_thesis_format.py`: writes `test_results.json` in the format of the laptop's `five_run_loss_experiments/<loss>/<run>_test_results.json`.
- `package_results.sh`: packs the results in the laptop's `evaluation_models` layout.

## Running on Snellius
```bash
cd ~/Multi-Quantile-Precipitation-Nowcasting
git fetch origin && git checkout claude/weighted-loss-baseline-wk6sns
cd weighted_loss && mkdir -p logs
sbatch run_weighted_loss.slurm              # B-MSE and B-MAE
# or, if the budget is short:  sbatch --array=0-4 run_weighted_loss.slurm   (B-MSE only)
```
Results go to `weighted_loss/results/` in the repo on your home directory. Training itself runs on node-local `$TMPDIR`; only the best checkpoint, config, metrics.csv and test results (~50 MB per run, ~0.5 GB for all 10) are copied to home; no per-epoch images. Check `myquota` before submitting: you need ~0.5 GB free plus room for the logs. Finished tasks are skipped, so resubmitting resumes. Uses the venv `ablation/.venv`.

## Into the paper's tables (laptop)
```bash
# Snellius, when all tasks are done
bash ~/Multi-Quantile-Precipitation-Nowcasting/weighted_loss/package_results.sh
# laptop, from hybrid-nowcasting-thesis-v0
scp <user>@snellius.surf.nl:weighted_loss_results.tgz evaluation_models/
tar xzf evaluation_models/weighted_loss_results.tgz -C evaluation_models
```
Then add two lines to `experiments` in `src/compare_models_five_runs.py`:
```python
    ("SmaAt-UNet (B-MSE)", "five_run_loss_experiments/B-MSE", "SmaAt-UNet", None),
    ("SmaAt-UNet (B-MAE)", "five_run_loss_experiments/B-MAE", "SmaAt-UNet", None),
```
and run it: the results are read from the cache, so nothing is re-evaluated. The per-run JSONs work for Fig. 4 the same way as the MSE/MAE ones.

Tested locally end to end (training, evaluation, packaging) on a small synthetic dataset, 2 epochs, CPU only.
