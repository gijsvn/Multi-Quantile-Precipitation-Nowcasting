#!/bin/bash
# Pack the weighted-loss test results in the laptop's evaluation_models layout:
#   five_run_loss_experiments/<loss>/full_training_logs/<run>/config.json
#   five_run_aggregates/five_run_loss_experiments/<loss>/<run>_test_results.json
# compare_models_five_runs.py finds the runs through the first and reads the
# cached results from the second, so no checkpoints are needed on the laptop.
#
# Usage: bash package_results.sh [OUT_ROOT]   -> ~/weighted_loss_results.tgz
set -euo pipefail

OUT_ROOT="${1:-$HOME/Multi-Quantile-Precipitation-Nowcasting/weighted_loss/results}"
STAGE=$(mktemp -d)

for eval_dir in "$OUT_ROOT"/eval/*/seed*; do
  [[ -f "$eval_dir/test_results.json" ]] || { echo "Skipping unfinished $eval_dir"; continue; }
  loss=$(basename "$(dirname "$eval_dir")")
  run="SmaAt-UNet_${loss}_$(basename "$eval_dir")"

  logs="$STAGE/five_run_loss_experiments/$loss/full_training_logs/$run"
  agg="$STAGE/five_run_aggregates/five_run_loss_experiments/$loss"
  mkdir -p "$logs" "$agg"

  # The laptop scripts read model_backbone from config.json
  python3 - "$eval_dir/config.json" "$logs/config.json" <<'EOF'
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

# Fig. 4 uses the best-of-5 run by validation loss, as for MSE/MAE/quantile:
#   five_run_loss_experiments/<loss>/best_val_models/SmaAt-UNet/test_results.json
for loss_dir in "$OUT_ROOT"/eval/*; do
  loss=$(basename "$loss_dir")
  best=$(python3 - "$loss_dir" <<'PY'
import pathlib, re, sys
runs = []
for d in sorted(pathlib.Path(sys.argv[1]).glob("seed*")):
    src = d / "source_checkpoint.txt"
    if (d / "test_results.json").exists() and src.exists():
        m = re.search(r"val_loss=([0-9.eE+-]+?)\.ckpt", src.read_text())
        if m:
            runs.append((float(m.group(1)), str(d)))
print(min(runs)[1] if runs else "")
PY
)
  [[ -n "$best" ]] || continue
  dest="$STAGE/five_run_loss_experiments/$loss/best_val_models/SmaAt-UNet"
  mkdir -p "$dest"
  cp "$best/test_results.json" "$best/source_checkpoint.txt" "$dest/"
  cp "$STAGE/five_run_loss_experiments/$loss/full_training_logs/SmaAt-UNet_${loss}_$(basename "$best")/config.json" "$dest/"
  echo "Best-of-5 for $loss (lowest val loss): $(basename "$best")"
done

tar czf "$HOME/weighted_loss_results.tgz" -C "$STAGE" .
rm -rf "$STAGE"
echo "Wrote $HOME/weighted_loss_results.tgz"
