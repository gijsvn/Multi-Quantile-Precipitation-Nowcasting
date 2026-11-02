"""
Performance diagram + matched-FAR table from threshold_sweep.py outputs.

Performance diagram (Roebber, 2009): POD (= recall) against success ratio
SR = 1 - FAR (= precision), with CSI isolines and frequency-bias lines.
Each curve is one output (MSE, MAE, q50, q90, q95) with its decision
threshold tau swept; the marker is the native operating point
(tau = event threshold), i.e. the numbers in Tables 1-2.

Metrics follow eval.py / the tables: computed per lead time and then averaged
over lead times; curves and markers are the mean over runs.

Outputs (in --output-dir):
  performance_diagram.pdf/.png   single-column figure, one panel per threshold
  matched_far.csv / .md          POD of every output at the native FAR of q90
                                 and q95, frequency bias, max CSI over tau
  sweep_curves.csv               all curve points (per run), for re-plotting

Usage:
  python plot_performance_diagram.py --sweep-dir sweep_results --output-dir sweep_results \
      --thresholds 10 20
"""

import argparse
import pathlib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# label in npz + channel -> display name
DISPLAY = {
    ("MSE", "point"): "MSE",
    ("MAE", "point"): "MAE",
    ("Quantile", "q50"): r"$q_{0.50}$",
    ("Quantile", "q90"): r"$q_{0.90}$",
    ("Quantile", "q95"): r"$q_{0.95}$",
}
ORDER = list(DISPLAY.keys())
STYLE = {
    ("MSE", "point"): dict(color="#7f7f7f", marker="s"),
    ("MAE", "point"): dict(color="#bcbd22", marker="D"),
    ("Quantile", "q50"): dict(color="#1f77b4", marker="o"),
    ("Quantile", "q90"): dict(color="#ff7f0e", marker="^"),
    ("Quantile", "q95"): dict(color="#d62728", marker="v"),
}


def safe_div(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    out = np.zeros(np.broadcast(a, b).shape)
    np.divide(a, b, out=out, where=b != 0)
    return out


def metrics_from_counts(c):
    """c[..., T, K, 4] (TN, FP, FN, TP) -> dict of horizon-mean metrics [..., K]."""
    tn, fp, fn, tp = (c[..., i].astype(np.float64) for i in range(4))
    pod = safe_div(tp, tp + fn)
    far = safe_div(fp, tp + fp)              # eval.py convention: 0 if no forecasts
    csi = safe_div(tp, tp + fp + fn)
    bias = safe_div(tp + fp, tp + fn)
    defined = (tp + fp).sum(axis=-2) > 0      # SR undefined without any forecast positives
    return {
        "POD": pod.mean(axis=-2),
        "FAR": far.mean(axis=-2),
        "SR": 1 - far.mean(axis=-2),
        "CSI": csi.mean(axis=-2),
        "bias": bias.mean(axis=-2),
        "defined": defined,
    }


def load_sweeps(sweep_dir):
    rows = []
    for f in sorted(sweep_dir.glob("*_run*.npz")):
        d = np.load(f, allow_pickle=False)
        label = str(d["label"])
        for qi, ch in enumerate(d["channels"]):
            rows.append(dict(label=label, channel=str(ch), run=f.stem, tau=d["tau"],
                             events=d["event_thresholds"], counts=d["counts"][qi]))
    if not rows:
        raise SystemExit(f"No *_run*.npz files in {sweep_dir}")
    return rows


def pod_at_far(far, pod, target):
    """Linear interpolation of POD at a given FAR along the tau curve."""
    if len(far) == 0:
        return np.nan
    o = np.argsort(far)
    far, pod = far[o], pod[o]
    if target < far[0] or target > far[-1]:
        return np.nan
    return float(np.interp(target, far, pod))


def background(ax):
    sr = np.linspace(0.001, 1, 300)
    pod = np.linspace(0.001, 1, 300)
    SR, POD = np.meshgrid(sr, pod)
    CSI = 1.0 / (1.0 / SR + 1.0 / POD - 1.0)
    cs = ax.contour(SR, POD, CSI, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    colors="0.75", linewidths=0.5)
    ax.clabel(cs, fmt="%.1f", fontsize=5, inline=True)
    for b in [0.25, 0.5, 1, 2, 4]:
        x = np.linspace(0, 1, 50)
        ax.plot(x, b * x, ls="--", color="0.6", lw=0.4)
        if b * 1.0 <= 1:
            ax.text(0.99, b * 0.99, f"{b:g}", fontsize=5, color="0.4", ha="right", va="bottom")
        else:
            ax.text(1 / b, 0.995, f"{b:g}", fontsize=5, color="0.4", ha="center", va="bottom")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", type=pathlib.Path, required=True)
    p.add_argument("--output-dir", type=pathlib.Path, default=None)
    p.add_argument("--thresholds", type=float, nargs="+", default=[10.0, 20.0])
    p.add_argument("--width", type=float, default=3.5, help="figure width in inches (single column ~3.5)")
    args = p.parse_args()
    out = args.output_dir or args.sweep_dir
    out.mkdir(parents=True, exist_ok=True)

    sweeps = load_sweeps(args.sweep_dir)
    curve_rows, table_rows = [], []

    # per (key, event threshold): stack of per-run metrics
    per_key = {}
    for s in sweeps:
        key = (s["label"], s["channel"])
        for ei, ev in enumerate(s["events"]):
            m = metrics_from_counts(s["counts"][ei])
            tau = s["tau"]
            k_native = int(np.argmin(np.abs(tau - ev)))
            per_key.setdefault((key, float(ev)), []).append((m, tau, k_native, s["run"]))
            for k in range(len(tau)):
                curve_rows.append(dict(model=key[0], channel=key[1], run=s["run"], event_thr=float(ev),
                                       tau=float(tau[k]), POD=m["POD"][k], FAR=m["FAR"][k], SR=m["SR"][k],
                                       CSI=m["CSI"][k], bias=m["bias"][k], defined=bool(m["defined"][k])))
    pd.DataFrame(curve_rows).to_csv(out / "sweep_curves.csv", index=False)

    keys_present = [k for k in ORDER if any(kk == k for (kk, _) in per_key)]
    keys_present += sorted({kk for (kk, _) in per_key} - set(keys_present))

    # ---------------- matched-FAR table ----------------
    for ev in args.thresholds:
        native_far = {}
        for key in keys_present:
            runs = per_key.get((key, ev), [])
            if runs:
                native_far[key] = np.mean([m["FAR"][kn] for m, _, kn, _ in runs])
        targets = [k for k in [("Quantile", "q90"), ("Quantile", "q95")] if k in native_far]
        for key in keys_present:
            runs = per_key.get((key, ev), [])
            if not runs:
                continue
            row = {"event_thr": ev, "output": DISPLAY.get(key, f"{key[0]}:{key[1]}").replace("$", ""),
                   "n_runs": len(runs)}
            for name, vals in [
                ("POD_native", [m["POD"][kn] for m, _, kn, _ in runs]),
                ("FAR_native", [m["FAR"][kn] for m, _, kn, _ in runs]),
                ("CSI_native", [m["CSI"][kn] for m, _, kn, _ in runs]),
                ("bias_native", [m["bias"][kn] for m, _, kn, _ in runs]),
                ("CSI_max_over_tau", [m["CSI"].max() for m, _, _, _ in runs]),
                ("tau_at_CSI_max", [tau[m["CSI"].argmax()] for m, tau, _, _ in runs]),
            ]:
                row[f"{name}_mean"] = float(np.mean(vals))
                row[f"{name}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            for t in targets:
                f_star = native_far[t]
                vals = [pod_at_far(m["FAR"][m["defined"]], m["POD"][m["defined"]], f_star) for m, _, _, _ in runs]
                tag = t[1]
                row[f"FAR_target_{tag}"] = float(f_star)
                row[f"POD_at_FAR_{tag}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
                row[f"POD_at_FAR_{tag}_std"] = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else np.nan
            table_rows.append(row)

    table = pd.DataFrame(table_rows)
    table.to_csv(out / "matched_far.csv", index=False)

    md = []
    for ev in args.thresholds:
        t = table[table.event_thr == ev]
        if t.empty:
            continue
        md.append(f"### >= {ev:g} mm/h (mean ± std over runs; metrics averaged over lead times)\n")
        cols = ["Output", "POD", "FAR", "CSI", "Bias", "max CSI (tau)"]
        tags = [c.replace("FAR_target_", "") for c in t.columns if c.startswith("FAR_target_")]
        for tag in tags:
            cols.append(f"POD @ FAR of {tag} ({t[f'FAR_target_{tag}'].iloc[0]:.3f})")
        md.append("| " + " | ".join(cols) + " |")
        md.append("|" + "---|" * len(cols))
        for _, r in t.iterrows():
            cells = [r.output]
            for n in ["POD_native", "FAR_native", "CSI_native", "bias_native"]:
                cells.append(f"{r[n + '_mean']:.3f} ± {r[n + '_std']:.3f}")
            cells.append(f"{r.CSI_max_over_tau_mean:.3f} ({r.tau_at_CSI_max_mean:.3g} mm/h)")
            for tag in tags:
                v, s = r[f"POD_at_FAR_{tag}_mean"], r[f"POD_at_FAR_{tag}_std"]
                cells.append("n/a" if not np.isfinite(v) else f"{v:.3f} ± {s:.3f}")
            md.append("| " + " | ".join(cells) + " |")
        md.append("")
    (out / "matched_far.md").write_text("\n".join(md))
    print("\n".join(md))

    # ---------------- figure ----------------
    n = len(args.thresholds)
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6})
    fig, axes = plt.subplots(1, n, figsize=(args.width, args.width / n + 0.75), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ev in zip(axes, args.thresholds):
        background(ax)
        for key in keys_present:
            runs = per_key.get((key, ev), [])
            if not runs:
                continue
            st = STYLE.get(key, dict(color="k", marker="x"))
            sr = np.mean([m["SR"] for m, _, _, _ in runs], axis=0)
            pod = np.mean([m["POD"] for m, _, _, _ in runs], axis=0)
            ok = np.all([m["defined"] for m, _, _, _ in runs], axis=0)
            ax.plot(sr[ok], pod[ok], color=st["color"], lw=0.9)
            kn = runs[0][2]
            ax.plot(sr[kn], pod[kn], ls="none", ms=4, mec="k", mew=0.4, **st)
        ax.set_title(rf"$\geq${ev:g} mm/h", fontsize=7, pad=2)
        ax.set_xlabel("Success ratio (1 − FAR)")
    axes[0].set_ylabel("POD")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], lw=0.9, ms=4, mec="k", mew=0.4, **STYLE.get(k, dict(color="k", marker="x")))
         for k in keys_present]
    l = [DISPLAY.get(k, f"{k[0]}:{k[1]}") for k in keys_present]
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=6, frameon=False, handlelength=1.5,
               columnspacing=1.0, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"performance_diagram.{ext}", dpi=300)
    print(f"Saved {out / 'performance_diagram.pdf'}")


if __name__ == "__main__":
    main()
