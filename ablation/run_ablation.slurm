#!/bin/bash
# =============================================================================
# Ablation: where do the multi-quantile gains come from? (R2 / editor)
#
# 5 variants x 5 seeds = 25 array tasks. Each task trains one model with the
# unmodified repo code, copies the best checkpoint + config.json into
#   $OUT_ROOT/eval/<variant>/seed<seed>/
# and evaluates it on the test set (test_results.json).
# Afterwards run aggregate_ablation.py to get the mean +/- std table.
#
# Submit:   sbatch run_ablation.slurm
# Rerun one task (e.g. index 7):   sbatch --array=7 run_ablation.slurm
# =============================================================================
#SBATCH --job-name=mq-ablation
#SBATCH --partition=gpu_h100            # PLACEHOLDER: or gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16              # 1/4 of an H100 node
#SBATCH --time=12:00:00                 # PLACEHOLDER: set from one paper run's wall time + margin
#SBATCH --array=0-24%8                  # 25 tasks, at most 8 at once
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# #SBATCH --account=<PLACEHOLDER>       # only if your project requires it

set -euo pipefail

# ----------------------------- PLACEHOLDERS ----------------------------------
REPO_DIR="$HOME/Multi-Quantile-Precipitation-Nowcasting"     # PLACEHOLDER
DATA_FILE="/projects/0/<project>/data/precipitation_dataset.h5" # PLACEHOLDER
OUT_ROOT="/projects/0/<project>/ablation_results"             # PLACEHOLDER (persistent storage, not $TMPDIR)
VENV="$HOME/venvs/mqpn"                                       # PLACEHOLDER

module purge
module load 2024                                              # PLACEHOLDER: check `module avail`
module load Python/3.12.3-GCCcore-13.3.0                      # PLACEHOLDER
source "$VENV/bin/activate"
# -----------------------------------------------------------------------------

SEEDS=(0 1 2 3 4)

# name | train.py arguments (everything else stays at the paper defaults:
# Adam 1e-3, batch 16, LR patience 4, early stopping 15, sum reduction)
VARIANTS=(
  "mae|--loss l1"
  "median_pinball|--loss quantile --quantiles 0.5"
  "w0|--loss quantile --quantiles 0.5 0.9 0.95 --higher-q-weight 0.0"
  "median_x3|--loss quantile --quantiles 0.5 0.5 0.5 --higher-q-weight 0.5"
  "full|--loss quantile --quantiles 0.5 0.9 0.95 --higher-q-weight 0.5"
  # Optional: multi-seed MSE baseline for the main table (raise --array to 0-29)
  # "mse|--loss mse"
)

N_SEEDS=${#SEEDS[@]}
V_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}
NAME="${VARIANTS[$V_IDX]%%|*}"
ARGS="${VARIANTS[$V_IDX]#*|}"

echo "Task $SLURM_ARRAY_TASK_ID: variant=$NAME seed=$SEED args=[$ARGS] host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Copy the dataset to node-local scratch for fast I/O
cp "$DATA_FILE" "$TMPDIR/data.h5"

# One log dir per task: train.py names runs by date + run number, which
# would race if several array tasks shared a directory.
TRAIN_DIR="$OUT_ROOT/train/$NAME/seed$SEED"
EVAL_DIR="$OUT_ROOT/eval/$NAME/seed$SEED"
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

# Best checkpoint (ModelCheckpoint keeps top-1 by val_loss) + its config.json
RUN_DIR=$(dirname "$(ls -td "$TRAIN_DIR"/*/config.json | head -n1)")
CKPT=$(ls "$RUN_DIR"/checkpoints/*.ckpt | head -n1)
cp "$CKPT" "$EVAL_DIR/model.ckpt"
cp "$RUN_DIR/config.json" "$EVAL_DIR/config.json"
echo "$CKPT" > "$EVAL_DIR/source_checkpoint.txt"

python eval.py \
  --model-path "$EVAL_DIR/model.ckpt" \
  --data-file "$TMPDIR/data.h5" \
  --evaluation-set test \
  --precipitation-thresholds 0.5 10.0 20.0

# Note: the best val_loss is in the checkpoint filename (source_checkpoint.txt),
# which aggregate_ablation.py uses for the paper's best-of-5 protocol.

echo "Done: $EVAL_DIR"
