"""
Aggregate the ablation runs into one table.

Reads <root>/eval/<variant>/seed<N>/test_results.json (written by eval.py via
run_ablation.slurm) and prints/saves, per variant:
  - mean +/- std over seeds of test MSE, MAE and CSI at each threshold
    (metrics averaged over the 12 lead times, as in the paper tables);
  - the paper's protocol: the seed with the lowest validation loss.

Usage:
    python aggregate_ablation.py --root /projects/0/<project>/ablation_results
"""

import argparse
import json
import pathlib
import re

import numpy as np
import pandas as pd

ORDER = ["mse", "mae", "median_pinball", "w0", "median_x3", "full"]
LABELS = {
    "mse": "MSE",
    "mae": "MAE (L1)",
    "median_pinball": "Median-only pinball (12 ch)",
    "w0": "Quantile, w0.90 = w0.95 = 0 (36 ch)",
    "median_x3": "3 x q=0.5 heads (36 ch)",
    "full": "Full multi-quantile (36 ch)",
}


def load_runs(root: pathlib.Path, thresholds: list[str]) -> pd.DataFrame:
    rows = []
    for res_path in sorted(root.glob("eval/*/seed*/test_results.json")):
        run_dir = res_path.parent
        with open(res_path, encoding="utf-8") as f:
            res = json.load(f)

        val_loss = np.nan
        src = run_dir / "source_checkpoint.txt"
        if src.exists():
            m = re.search(r"val_loss=([0-9.eE+-]+?)\.ckpt", src.read_text())
            if m:
                val_loss = float(m.group(1))

        row = {
            "variant": run_dir.parent.name,
            "seed": int(run_dir.name.removeprefix("seed")),
            "val_loss": val_loss,
            "MSE": float(np.mean(res["MSE"])),
            "MAE": float(np.mean(res["MAE"])),
        }
        for thr in thresholds:
            row[f"CSI@{thr}"] = float(np.mean(res[thr]["CSI"]))
        rows.append(row)

    if not rows:
        raise SystemExit(f"No test_results.json found under {root}/eval/")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--thresholds", nargs="+", default=["0.5", "10.0", "20.0"])
    args = parser.parse_args()

    df = load_runs(args.root, args.thresholds)
    metrics = ["MSE", "MAE"] + [f"CSI@{t}" for t in args.thresholds]
    variants = [v for v in ORDER if v in set(df.variant)] + sorted(
        set(df.variant) - set(ORDER)
    )

    df.sort_values(["variant", "seed"]).to_csv(args.root / "ablation_all_runs.csv", index=False)

    summary, best = [], []
    for v in variants:
        d = df[df.variant == v]
        row = {"Variant": LABELS.get(v, v), "n": len(d)}
        for m in metrics:
            row[m] = f"{d[m].mean():.4f} ± {d[m].std(ddof=1):.4f}" if len(d) > 1 else f"{d[m].mean():.4f}"
        summary.append(row)

        if d.val_loss.notna().any():
            b = d.loc[d.val_loss.idxmin()]
            best.append({"Variant": LABELS.get(v, v), "seed": int(b.seed),
                         **{m: round(b[m], 4) for m in metrics}})

    summary = pd.DataFrame(summary)
    print("\nMean ± std over seeds (test set, averaged over lead times)\n")
    print(summary.to_string(index=False))
    summary.to_csv(args.root / "ablation_summary.csv", index=False)
    (args.root / "ablation_summary.md").write_text(summary.to_markdown(index=False) if _has_tabulate() else summary.to_string(index=False))

    if best:
        best = pd.DataFrame(best)
        print("\nBest-of-seeds by validation loss (paper protocol)\n")
        print(best.to_string(index=False))
        best.to_csv(args.root / "ablation_best_of_seeds.csv", index=False)

    print(f"\nSaved tables to {args.root}")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
