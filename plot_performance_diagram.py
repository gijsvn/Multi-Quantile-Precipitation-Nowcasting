"""
Precision-recall (performance-diagram) figure + matched-FAR table from threshold_sweep.py outputs.

Performance diagram (Roebber, 2009): POD (= recall) against success ratio
SR = 1 - FAR (= precision), with CSI isolines and frequency-bias lines.
Each curve is one output (MSE, MAE, q50, q90, q95) with its decision
threshold tau swept; the marker is the native operating point
(tau = event threshold), i.e. the numbers in Tables 1-2.

Metrics follow eval.py / the tables: computed per lead time and then averaged
over lead times; curves and markers are the mean over runs.

Outputs (in --output-dir):
  performance_diagram.pdf/.png   single-column figure, one panel per threshold
  matched_far.csv / .md          per output: native POD/FAR/CSI/bias, best CSI
                                 over tau, and POD at the native FAR of q90/q95
  sweep_curves.csv               all curve points (per run), for re-plotting

POD at matched FAR is read from the monotone part of each curve (tau up to the
point of maximum SR, and tau <= --tau-max). At very high tau almost nothing is
forecast and SR becomes noisy, so FAR is no longer monotone in tau there.

Usage:
  python plot_performance_diagram.py --sweep-dir sweep_results --thresholds 10 20
"""

import argparse
import pathlib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# (npz label, channel) -> display name; colours/markers follow Fig. 4 of the paper
DISPLAY = {
    ("MSE", "point"): "MSE",
    ("MAE", "point"): "MAE",
    ("Quantile", "q50"): "Quantile (q=0.50)",
    ("Quantile", "q90"): "Quantile (q=0.90)",
    ("Quantile", "q95"): "Quantile (q=0.95)",
}
ORDER = list(DISPLAY.keys())
STYLE = {
    ("MSE", "point"): dict(color="C0", marker="o"),
    ("MAE", "point"): dict(color="C1", marker="s"),
    ("Quantile", "q50"): dict(color="C2", marker="^"),
    ("Quantile", "q90"): dict(color="C3", marker="^"),
    ("Quantile", "q95"): dict(color="C4", marker="D"),
}
SHORT = {("Quantile", "q50"): "q50", ("Quantile", "q90"): "q90", ("Quantile", "q95"): "q95"}


def safe_div(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    out = np.zeros(np.broadcast(a, b).shape)
    np.divide(a, b, out=out, where=b != 0)
    return out


def metrics_from_counts(c):
    """c[..., T, K, 4] (TN, FP, FN, TP) -> dict of lead-time-mean metrics [..., K]."""
    tn, fp, fn, tp = (c[..., i].astype(np.float64) for i in range(4))
    far = safe_div(fp, tp + fp)  # eval.py convention: 0 if nothing forecast
    return {
        "POD": safe_div(tp, tp + fn).mean(axis=-2),
        "FAR": far.mean(axis=-2),
        "SR": 1 - far.mean(axis=-2),
        "CSI": safe_div(tp, tp + fp + fn).mean(axis=-2),
        "bias": safe_div(tp + fp, tp + fn).mean(axis=-2),
        "defined": (tp + fp).sum(axis=-2) > 0,
    }


def load_sweeps(sweep_dir):
    rows = []
    for f in sorted(sweep_dir.glob("*_run*.npz")):
        d = np.load(f, allow_pickle=False)
        for qi, ch in enumerate(d["channels"]):
            rows.append(dict(label=str(d["label"]), channel=str(ch), run=f.stem, tau=d["tau"],
                             events=d["event_thresholds"], counts=d["counts"][qi]))
    if not rows:
        raise SystemExit(f"No *_run*.npz files in {sweep_dir}")
    return rows


def monotone_branch(tau, sr, pod, tau_max):
    """Part of the curve with tau <= tau_max up to the maximum SR (SR increasing, POD decreasing)."""
    ok = tau <= tau_max
    tau, sr, pod = tau[ok], sr[ok], pod[ok]
    i = int(np.argmax(sr))
    return sr[: i + 1], pod[: i + 1]


def pod_at_sr(tau, sr, pod, sr_target, tau_max):
    s, p = monotone_branch(tau, sr, pod, tau_max)
    if len(s) == 0 or sr_target > s.max() or sr_target < s.min():
        return np.nan
    o = np.argsort(s)
    return float(np.interp(sr_target, s[o], p[o]))


def display_curve(key, tau, sr, pod, ev, tau_max):
    """Mean curve for plotting: tau <= tau_max and only up to the maximum SR (beyond it almost
    nothing is forecast and SR is noise), but always including the native point tau = ev."""
    kn = int(np.argmin(np.abs(tau - ev)))
    ok = np.where(tau <= tau_max)[0]
    end = max(int(ok[np.argmax(sr[ok])]), kn) + 1
    return key, tau[:end], sr[:end], pod[:end], kn, None


# two-sided 95% Student-t critical values (scipy is not a dependency of this repo)
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def t_crit_95(df):
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df=df))
    except ImportError:
        return T975.get(df, 1.96)


# Style of the paper's other figures (src/plot_metrics_across_lead_time_five_runs.py)
RC_STYLE = {
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "legend.title_fontsize": 13,
}


def draw_panel(ax, curves, title, xmax=None):
    """curves: list of (key, tau, SR, POD, k_native, ci) with mean-over-runs SR/POD (already truncated);
    ci is None or (SR half-width, POD half-width) of the 95% CI at the native point."""
    if xmax is None:
        xmax = max(float(np.nanmax(c[2])) for c in curves) * 1.2  # room for the legend (upper right)
    for key, tau, s, p, kn, ci in curves:
        st = STYLE.get(key, dict(color="k", marker="x"))
        ax.plot(s, p, color=st["color"], linestyle="-", label=DISPLAY.get(key, f"{key[0]}:{key[1]}"))
        ax.plot(s[kn], p[kn], ls="none", marker=st["marker"], markersize=9, color=st["color"], zorder=4)
        if ci is not None:
            ax.errorbar(s[kn], p[kn], xerr=ci[0], yerr=ci[1], fmt="none", ecolor=st["color"],
                        elinewidth=1.2, capsize=3, zorder=3)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("1 - FAR (Precision)")
    ax.set_ylabel("POD (Recall)")
    ax.grid(True)


def make_figure(panels, out, figsize=(12, 4.75), dpi=180):
    """panels: list of (title, curves) for draw_panel."""
    with plt.rc_context(RC_STYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=figsize)
        axes = np.atleast_1d(axes)
        for ax, (title, curves) in zip(axes, panels):
            draw_panel(ax, curves, title)
            # legend handles as in the lead-time figure: line + marker
            keys = [c[0] for c in curves]
            h = [Line2D([], [], linestyle="-", markersize=8, **STYLE.get(k, dict(color="k", marker="x")))
                 for k in keys]
            ax.legend(h, [DISPLAY.get(k, f"{k[0]}:{k[1]}") for k in keys], loc="upper right")
        plt.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out / f"performance_diagram.{ext}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", type=pathlib.Path, required=True)
    p.add_argument("--output-dir", type=pathlib.Path, default=None)
    p.add_argument("--thresholds", type=float, nargs="+", default=[10.0, 20.0])
    p.add_argument("--tau-max", type=float, default=40.0,
                   help="ignore decision thresholds above this (too few forecasts, noisy SR)")
    p.add_argument("--ci", action="store_true",
                   help="draw 95%% CI error bars (over runs) at the native operating points")
    p.add_argument("--figsize", type=float, nargs=2, default=[12, 4.75],
                   help="figure size in inches (lead-time figure uses 6 x 4.75 per panel)")
    args = p.parse_args()
    out = args.output_dir or args.sweep_dir
    out.mkdir(parents=True, exist_ok=True)

    sweeps = load_sweeps(args.sweep_dir)
    per_key, curve_rows = {}, []
    for s in sweeps:
        key = (s["label"], s["channel"])
        for ei, ev in enumerate(s["events"]):
            m = metrics_from_counts(s["counts"][ei])
            tau = s["tau"]
            kn = int(np.argmin(np.abs(tau - ev)))
            per_key.setdefault((key, round(float(ev), 4)), []).append((m, tau, kn))
            for k in range(len(tau)):
                curve_rows.append(dict(model=key[0], channel=key[1], run=s["run"], event_thr=round(float(ev), 4),
                                       tau=float(tau[k]), POD=m["POD"][k], FAR=m["FAR"][k], SR=m["SR"][k],
                                       CSI=m["CSI"][k], bias=m["bias"][k], defined=bool(m["defined"][k])))
    pd.DataFrame(curve_rows).to_csv(out / "sweep_curves.csv", index=False)

    keys = sorted({k for (k, _) in per_key}, key=lambda k: ORDER.index(k) if k in ORDER else 99)

    def stat(vals):
        vals = np.asarray(vals, dtype=float)
        ok = np.isfinite(vals)
        mean = float(vals[ok].mean()) if ok.any() else np.nan
        std = float(vals[ok].std(ddof=1)) if ok.sum() > 1 else np.nan
        return mean, std, int(ok.sum())

    rows, panels = [], []
    for ev in [round(float(e), 4) for e in args.thresholds]:
        native_sr = {}
        for key in keys:
            runs = per_key.get((key, ev), [])
            if runs:
                native_sr[key] = np.mean([m["SR"][kn] for m, _, kn in runs])
        targets = [k for k in (("Quantile", "q90"), ("Quantile", "q95")) if k in native_sr]
        curves = []
        for key in keys:
            runs = per_key.get((key, ev), [])
            if not runs:
                continue
            row = {"event_thr": ev, "output": DISPLAY.get(key, f"{key[0]}:{key[1]}"), "n_runs": len(runs)}
            for name in ("POD", "FAR", "CSI", "bias"):
                row[f"{name}_native_mean"], row[f"{name}_native_std"], _ = stat([m[name][kn] for m, _, kn in runs])
            csi_max = [m["CSI"][t <= args.tau_max].max() for m, t, _ in runs]
            tau_best = [t[t <= args.tau_max][m["CSI"][t <= args.tau_max].argmax()] for m, t, _ in runs]
            row["CSI_best_mean"], row["CSI_best_std"], _ = stat(csi_max)
            row["tau_best_mean"], _, _ = stat(tau_best)
            for t in targets:
                tag = SHORT[t]
                vals = [pod_at_sr(tt, m["SR"], m["POD"], native_sr[t], args.tau_max) for m, tt, _ in runs]
                row[f"FAR_target_{tag}"] = 1 - native_sr[t]
                (row[f"POD_at_FAR_{tag}_mean"], row[f"POD_at_FAR_{tag}_std"],
                 row[f"POD_at_FAR_{tag}_n"]) = stat(vals)
            rows.append(row)

            tau = runs[0][1]
            sr = np.mean([m["SR"] for m, _, _ in runs], axis=0)
            pod = np.mean([m["POD"] for m, _, _ in runs], axis=0)
            key_, tau_, sr_, pod_, kn_, _ = display_curve(key, tau, sr, pod, ev, args.tau_max)
            ci = None
            if args.ci and len(runs) > 1:
                t_crit = t_crit_95(len(runs) - 1)
                ci = tuple(t_crit * np.std([m[n][kn] for m, _, kn in runs], ddof=1) / np.sqrt(len(runs))
                           for n in ("SR", "POD"))
            curves.append((key_, tau_, sr_, pod_, kn_, ci))
        if curves:
            panels.append((f"≥ {ev:g} mm/h", curves))

    table = pd.DataFrame(rows)
    table.to_csv(out / "matched_far.csv", index=False)

    md = []
    for ev in [round(float(e), 4) for e in args.thresholds]:
        t = table[table.event_thr == ev]
        if t.empty:
            continue
        tags = [c.replace("FAR_target_", "") for c in t.columns if c.startswith("FAR_target_")]
        cols = ["Output", "POD", "FAR", "CSI", "Bias", "best CSI over tau (tau)"]
        cols += [f"POD @ FAR of {tag} ({t[f'FAR_target_{tag}'].iloc[0]:.3f})" for tag in tags]
        md += [f"### >= {ev:g} mm/h (mean ± std over runs; metrics averaged over lead times)\n",
               "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in t.iterrows():
            cells = [r.output] + [f"{r[n + '_native_mean']:.3f} ± {r[n + '_native_std']:.3f}"
                                  for n in ("POD", "FAR", "CSI", "bias")]
            cells.append(f"{r.CSI_best_mean:.3f} ± {r.CSI_best_std:.3f} ({r.tau_best_mean:.3g} mm/h)")
            for tag in tags:
                v, s, n = r[f"POD_at_FAR_{tag}_mean"], r[f"POD_at_FAR_{tag}_std"], r[f"POD_at_FAR_{tag}_n"]
                cells.append("n/a" if not np.isfinite(v) else
                             f"{v:.3f} ± {s:.3f}" + ("" if n == r.n_runs else f" (n={n})"))
            md.append("| " + " | ".join(cells) + " |")
        md.append("")
    (out / "matched_far.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

    make_figure(panels, out, tuple(args.figsize))
    print(f"Saved {out / 'performance_diagram.pdf'}")


if __name__ == "__main__":
    main()
