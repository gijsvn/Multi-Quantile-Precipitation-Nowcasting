#!/bin/bash
# =============================================================================
# Bring the original MAE, w=0 and full-model runs into the ablation results.
#
# For each run listed below: copies its best checkpoint + config.json into
#   $OUT_ROOT/eval/<variant>/seed<N>/
# and evaluates it on the test set, exactly like run_ablation.slurm does for
# the new runs. Runs that already have test_results.json are skipped.
#
# Fill in the three lists (5 entries each). An entry can be either
#   - a run directory (the folder train.py created, holding config.json and a
#     checkpoints/ subfolder): it is re-evaluated on the test set on a GPU, or
#   - an existing <run>_test_results.json from your earlier evaluation (for the
#     quantile runs use the _q0 = q50 file): it is copied as is, no GPU needed.
# Then:  mkdir -p logs && sbatch import_original_runs.slurm
# (with only .json entries you can also just run: bash import_original_runs.slurm)
# =============================================================================
#SBATCH --job-name=mq-import
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ----------------------------- PLACEHOLDERS ----------------------------------
REPO_DIR="$HOME/Multi-Quantile-Precipitation-Nowcasting"
DATA_FILE="/home/gvannieuwkoop/hybrid-nowcasting-thesis-v0/data/kevin_18_18.h5"
OUT_ROOT="/home/gvannieuwkoop/Multi-Quantile-Precipitation-Nowcasting/ablation/results"
VENV="$HOME/Multi-Quantile-Precipitation-Nowcasting/ablation/.venv"

MAE_RUNS=(   # PLACEHOLDER: the 5 original MAE runs
  "/path/to/mae_run_1"
  "/path/to/mae_run_2"
  "/path/to/mae_run_3"
  "/path/to/mae_run_4"
  "/path/to/mae_run_5"
)
W0_RUNS=(    # PLACEHOLDER: the 5 original w0.90 = w0.95 = 0 runs
  "/path/to/w0_run_1"
  "/path/to/w0_run_2"
  "/path/to/w0_run_3"
  "/path/to/w0_run_4"
  "/path/to/w0_run_5"
)
FULL_RUNS=(  # PLACEHOLDER: the 5 original full-model (w = 0.5) runs
  "/path/to/full_run_1"
  "/path/to/full_run_2"
  "/path/to/full_run_3"
  "/path/to/full_run_4"
  "/path/to/full_run_5"
)

module purge 2>/dev/null || true
module load 2024 Python/3.12.3-GCCcore-13.3.0 2>/dev/null || true
source "$VENV/bin/activate"
# -----------------------------------------------------------------------------

SUBMIT_DIR="$PWD"   # relative entries in the lists are taken relative to here
cd "$REPO_DIR"

# Copy the dataset to node-local scratch once, only if a checkpoint needs evaluating
LOCAL_DATA=""
local_data() {
  if [[ -z "$LOCAL_DATA" ]]; then
    LOCAL_DATA="${TMPDIR:-/tmp}/mq_import_data.h5"
    cp "$DATA_FILE" "$LOCAL_DATA"
  fi
}

# Best checkpoint of a run: lowest val_loss in the filename, else the newest file
best_ckpt() {
  local dir="$1" ckpts
  ckpts=$(ls "$dir"/checkpoints/*.ckpt "$dir"/*.ckpt 2>/dev/null || true)
  [[ -n "$ckpts" ]] || { echo "No .ckpt found in $dir" >&2; return 1; }
  if grep -q "val_loss=" <<< "$ckpts"; then
    grep "val_loss=" <<< "$ckpts" | sed -E 's/.*val_loss=([0-9.eE+-]+)\.ckpt$/\1 &/' \
      | sort -g | head -n1 | cut -d' ' -f2-
  else
    ls -t $ckpts | head -n1
  fi
}

import_variant() {
  local name="$1"; shift
  local seed=0
  for run in "$@"; do
    [[ "$run" = /* ]] || run="$SUBMIT_DIR/$run"
    local dest="$OUT_ROOT/eval/$name/seed$seed"
    if [[ -f "$dest/test_results.json" ]]; then
      echo "Already done, skipping: $name seed$seed"
    elif [[ "$run" == *.json ]]; then
      echo "Importing $name seed$seed <- $run (existing results)"
      mkdir -p "$dest"
      cp "$run" "$dest/test_results.json"
      echo "$run" > "$dest/source_checkpoint.txt"
    else
      [[ -f "$run/config.json" ]] || { echo "Missing $run/config.json" >&2; exit 1; }
      local ckpt; ckpt=$(best_ckpt "$run")
      echo "Importing $name seed$seed <- $ckpt"
      mkdir -p "$dest"
      cp "$ckpt" "$dest/model.ckpt"
      cp "$run/config.json" "$dest/config.json"
      echo "$ckpt" > "$dest/source_checkpoint.txt"
      local_data
      python eval.py \
        --model-path "$dest/model.ckpt" \
        --data-file "$LOCAL_DATA" \
        --evaluation-set test \
        --precipitation-thresholds 0.5 10.0 20.0
    fi
    seed=$((seed + 1))
  done
}

import_variant mae  "${MAE_RUNS[@]}"
import_variant w0   "${W0_RUNS[@]}"
import_variant full "${FULL_RUNS[@]}"

echo "Done. Next: python $REPO_DIR/ablation/aggregate_ablation.py --root $OUT_ROOT"
