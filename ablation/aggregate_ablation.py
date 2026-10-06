"""
Aggregate the ablation runs into one table.

Reads <root>/eval/<variant>/seed<N>/test_results.json (written by eval.py via
run_ablation.slurm) and prints/saves, per variant:
  - mean +/- 95% CI half-width over seeds (Student t, df = n-1; for n = 5 this
    is 1.24 x sample std) of test MSE, MAE and CSI at each threshold
    (metrics averaged over the 12 lead times, as in the paper tables);
  - the paper's protocol: the seed with the lowest validation loss.

Usage:
    python aggregate_ablation.py --root /projects/0/<project>/ablation_results

Results stored elsewhere (e.g. your earlier five-run evaluations) can be added
per variant with --extra VARIANT=GLOB, one *_test_results.json file per run:
    python aggregate_ablation.py --root evaluation_models/ablation_runs \
        --extra "mae=evaluation_models/five_run_aggregates/five_run_loss_experiments/<MAE folder>/*_test_results.json" \
        --extra "full=evaluation_models/five_run_aggregates/five_run_loss_experiments/<quantile folder>/*_q0_test_results.json"
For quantile runs use the _q0 (q50) files.
"""

import argparse
import glob
import json
import pathlib
import re

import numpy as np
import pandas as pd

# Two-sided 95% Student t quantiles, t_{0.975, df}, for df = 1..10
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def ci95_halfwidth(x: pd.Series) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    return T975.get(n - 1, 1.96) * x.std(ddof=1) / np.sqrt(n)


ORDER = ["mse", "mae", "median_pinball", "w0", "median_x3", "full"]
LABELS = {
    "mse": "MSE",
    "mae": "MAE (L1)",
    "median_pinball": "Median-only pinball (12 ch)",
    "w0": "Quantile, w0.90 = w0.95 = 0 (36 ch)",
    "median_x3": "3 x q=0.5 heads (36 ch)",
    "full": "Full multi-quantile (36 ch)",
}


def result_row(res_path: pathlib.Path, variant: str, seed: int,
               source: str, thresholds: list[str]) -> dict:
    with open(res_path, encoding="utf-8") as f:
        res = json.load(f)

    # Best val_loss is encoded in the checkpoint filename, when known
    m = re.search(r"val_loss=([0-9.eE+-]+?)\.ckpt", source)
    row = {
        "variant": variant,
        "seed": seed,
        "source": source.strip(),
        "val_loss": float(m.group(1)) if m else np.nan,
        "MSE": float(np.mean(_find(res, "MSE", res_path))),
        "MAE": float(np.mean(_find(res, "MAE", res_path))),
    }
    for thr in thresholds:
        row[f"CSI@{thr}"] = float(np.mean(_csi(_find(res, thr, res_path), res_path)))
    return row


def _lookup(d, name: str):
    """Case-insensitive key lookup, searching nested dicts; None if absent."""
    if not isinstance(d, dict):
        return None
    for k, v in d.items():
        if str(k).lower() == name.lower():
            return v
    for v in d.values():
        found = _lookup(v, name)
        if found is not None:
            return found
    return None


def _find(d: dict, name: str, path: pathlib.Path):
    value = _lookup(d, name)
    if value is None:
        raise SystemExit(f"{path}: no '{name}' entry. Top-level keys: {list(d)}")
    return value


def _csi(block, path: pathlib.Path) -> np.ndarray:
    """CSI per lead time from a threshold block, whichever way it was stored."""
    csi = _lookup(block, "CSI")
    if csi is not None:
        return np.asarray(csi, dtype=float)
    tp, fp, fn = (_lookup(block, k) for k in ("tp", "fp", "fn"))
    if tp is not None and fp is not None and fn is not None:
        tp, fp, fn = (np.asarray(x, dtype=float) for x in (tp, fp, fn))
        return np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)
    pod = _lookup(block, "POD")
    if pod is None:
        pod = _lookup(block, "Recall")
    far = _lookup(block, "FAR")
    if pod is not None and far is not None:
        pod, far = np.asarray(pod, dtype=float), np.asarray(far, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            csi = 1.0 / (1.0 / pod + 1.0 / (1.0 - far) - 1.0)
        return np.nan_to_num(csi)
    keys = list(block) if isinstance(block, dict) else type(block).__name__
    raise SystemExit(f"{path}: can't find CSI (or TP/FP/FN, or POD+FAR) in a threshold entry. Its keys: {keys}")


def load_runs(root: pathlib.Path, thresholds: list[str], extra: list[str]) -> pd.DataFrame:
    rows = []
    for res_path in sorted(root.glob("eval/*/seed*/test_results.json")):
        run_dir = res_path.parent
        src = run_dir / "source_checkpoint.txt"
        source = src.read_text() if src.exists() else str(res_path)
        rows.append(result_row(res_path, run_dir.parent.name,
                               int(run_dir.name.removeprefix("seed")), source, thresholds))

    for spec in extra:
        variant, _, pattern = spec.partition("=")
        files = sorted(glob.glob(pattern))
        if not variant or not files:
            raise SystemExit(f"--extra {spec!r}: no files match {pattern!r}")
        if variant in {r["variant"] for r in rows}:
            raise SystemExit(f"--extra {spec!r}: variant {variant!r} already has runs under {root}/eval/")
        print(f"{variant}: {len(files)} runs from {pattern}")
        for seed, path in enumerate(files):
            p = pathlib.Path(path)
            with open(p, encoding="utf-8") as f:
                source = json.load(f).get("checkpoint", str(p)) if p.suffix == ".json" else str(p)
            rows.append(result_row(p, variant, seed, str(source), thresholds))

    if not rows:
        raise SystemExit(f"No test_results.json found under {root}/eval/")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--thresholds", nargs="+", default=["0.5", "10.0", "20.0"])
    parser.add_argument("--extra", action="append", default=[], metavar="VARIANT=GLOB",
                        help="Add runs from existing *_test_results.json files (repeatable).")
    args = parser.parse_args()

    df = load_runs(args.root, args.thresholds, args.extra)
    metrics = ["MSE", "MAE"] + [f"CSI@{t}" for t in args.thresholds]
    variants = [v for v in ORDER if v in set(df.variant)] + sorted(
        set(df.variant) - set(ORDER)
    )

    df.sort_values(["variant", "seed"]).to_csv(args.root / "ablation_all_runs.csv", index=False)

    summary, numeric, best = [], [], []
    for v in variants:
        d = df[df.variant == v]
        row = {"Variant": LABELS.get(v, v), "n": len(d)}
        num = {"variant": v, "n": len(d)}
        for m in metrics:
            mean, std, ci = d[m].mean(), d[m].std(ddof=1), ci95_halfwidth(d[m])
            row[m] = f"{mean:.4f} ± {ci:.4f}" if len(d) > 1 else f"{mean:.4f}"
            num.update({f"{m}_mean": mean, f"{m}_std": std, f"{m}_ci95": ci})
        summary.append(row)
        numeric.append(num)

        if d.val_loss.notna().any():
            b = d.loc[d.val_loss.idxmin()]
            best.append({"Variant": LABELS.get(v, v), "seed": int(b.seed),
                         **{m: round(b[m], 4) for m in metrics}})

    summary = pd.DataFrame(summary)
    print("\nMean ± 95% CI half-width over seeds (test set, averaged over lead times)\n")
    print(summary.to_string(index=False))
    pd.DataFrame(numeric).to_csv(args.root / "ablation_summary.csv", index=False)
    (args.root / "ablation_summary.md").write_text(summary.to_markdown(index=False) if _has_tabulate() else summary.to_string(index=False))

    if best:
        best = pd.DataFrame(best)
        print("\nBest-of-seeds by validation loss (original paper protocol)\n")
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
