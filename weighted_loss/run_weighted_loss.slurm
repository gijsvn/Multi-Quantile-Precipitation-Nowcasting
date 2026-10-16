#!/bin/bash
# =============================================================================
# Weighted-loss baseline (R2 / editor): SmaAt-UNet trained with the balanced
# MSE and MAE of Shi et al. (2017), weights 1, 2, 5, 10, 30 for
# <2, 2-5, 5-10, 10-30, >=30 mm/h (thresholds converted to the normalized scale
# in util/balanced_loss.py).
#
# Same data file, architecture and training protocol as the paper's
# MSE/MAE/quantile runs (Adam 1e-3, ReduceLROnPlateau patience 4, early
# stopping 15, best-validation checkpoint, sum reduction, batch 16).
#
# Task index = loss_index * 5 + seed_index:
#   0-4 B-MSE (seeds 0-4) | 5-9 B-MAE (seeds 0-4)
# Submit all:        mkdir -p logs && sbatch run_weighted_loss.slurm
# Only B-MSE:        sbatch --array=0-4 run_weighted_loss.slurm
# Finished tasks are skipped, so resubmitting the array resumes.
#
# Each task writes $OUT_ROOT/eval/<loss>/seed<seed>/ with model.ckpt,
# config.json and test_results.json (same JSON format as the laptop's
# five-run results). Afterwards: bash package_results.sh
# =============================================================================
#SBATCH --job-name=mq-weighted
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-9%8
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

REPO_DIR="$HOME/Multi-Quantile-Precipitation-Nowcasting"
DATA_FILE="/home/gvannieuwkoop/hybrid-nowcasting-thesis-v0/data/kevin_18_18.h5"
OUT_ROOT="/scratch-shared/$USER/weighted_loss"   # scratch: home quota filled up last time
VENV="$HOME/Multi-Quantile-Precipitation-Nowcasting/ablation/.venv"

module purge
module load 2024 Python/3.12.3-GCCcore-13.3.0
source "$VENV/bin/activate"

SEEDS=(0 1 2 3 4)
LOSSES=(
  "B-MSE|--loss b-mse"
  "B-MAE|--loss b-mae"
)

N_SEEDS=${#SEEDS[@]}
L_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}
NAME="${LOSSES[$L_IDX]%%|*}"
ARGS="${LOSSES[$L_IDX]#*|}"

echo "Task $SLURM_ARRAY_TASK_ID: loss=$NAME seed=$SEED args=[$ARGS] host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

TRAIN_DIR="$OUT_ROOT/train/$NAME/seed$SEED"
EVAL_DIR="$OUT_ROOT/eval/$NAME/seed$SEED"

if [[ -f "$EVAL_DIR/test_results.json" ]]; then
  echo "Already done, skipping: $NAME seed $SEED"
  exit 0
fi

cp "$DATA_FILE" "$TMPDIR/data.h5"

rm -rf "$TRAIN_DIR"   # drop a partial run from an interrupted attempt
mkdir -p "$TRAIN_DIR" "$EVAL_DIR"

cd "$REPO_DIR"
echo "Repo commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)" | tee "$EVAL_DIR/commit.txt"

# shellcheck disable=SC2086
python train.py \
  --data-file "$TMPDIR/data.h5" \
  --result-log-dir "$TRAIN_DIR" \
  --seed "$SEED" \
  --num-workers 8 \
  $ARGS

# Best checkpoint (top-1 by val_loss) + its config.json
RUN_DIR=$(dirname "$(ls -td "$TRAIN_DIR"/*/config.json | head -n1)")
CKPT=$(ls "$RUN_DIR"/checkpoints/*.ckpt | head -n1)
cp "$CKPT" "$EVAL_DIR/model.ckpt"
cp "$RUN_DIR/config.json" "$EVAL_DIR/config.json"
cp "$RUN_DIR/metrics.csv" "$EVAL_DIR/metrics.csv" 2>/dev/null || true
echo "$CKPT" > "$EVAL_DIR/source_checkpoint.txt"

python weighted_loss/eval_thesis_format.py \
  --model-path "$EVAL_DIR/model.ckpt" \
  --data-file "$TMPDIR/data.h5" \
  --output "$EVAL_DIR/test_results.json" \
  --precipitation-thresholds 0.5 10.0 20.0

echo "Done: $EVAL_DIR"
