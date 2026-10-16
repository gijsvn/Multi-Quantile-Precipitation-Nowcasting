#!/bin/bash
# Pack the weighted-loss test results in the laptop's evaluation_models layout:
#   five_run_loss_experiments/<loss>/full_training_logs/<run>/config.json
#   five_run_aggregates/five_run_loss_experiments/<loss>/<run>_test_results.json
# compare_models_five_runs.py finds the runs through the first and reads the
# cached results from the second, so no checkpoints are needed on the laptop.
#
# Usage: bash package_results.sh [OUT_ROOT]   -> ~/weighted_loss_results.tgz
set -euo pipefail

OUT_ROOT="${1:-/scratch-shared/$USER/weighted_loss}"
STAGE=$(mktemp -d)

for eval_dir in "$OUT_ROOT"/eval/*/seed*; do
  [[ -f "$eval_dir/test_results.json" ]] || { echo "Skipping unfinished $eval_dir"; continue; }
  loss=$(basename "$(dirname "$eval_dir")")
  run="SmaAt-UNet_${loss}_$(basename "$eval_dir")"

  logs="$STAGE/five_run_loss_experiments/$loss/full_training_logs/$run"
  agg="$STAGE/five_run_aggregates/five_run_loss_experiments/$loss"
  mkdir -p "$logs" "$agg"

  # The laptop scripts read model_backbone from config.json
  python - "$eval_dir/config.json" "$logs/config.json" <<'EOF'
import json, sys
config = json.load(open(sys.argv[1]))
config.setdefault("model_backbone", "SmaAt-UNet")
json.dump(config, open(sys.argv[2], "w"), indent=4)
EOF
  cp "$eval_dir/source_checkpoint.txt" "$logs/" 2>/dev/null || true
  cp "$eval_dir/metrics.csv" "$logs/" 2>/dev/null || true
  cp "$eval_dir/test_results.json" "$agg/${run}_test_results.json"
  echo "Packed $loss $(basename "$eval_dir")"
done

tar czf "$HOME/weighted_loss_results.tgz" -C "$STAGE" .
rm -rf "$STAGE"
echo "Wrote $HOME/weighted_loss_results.tgz"
