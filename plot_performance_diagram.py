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
    return key, tau[:end], sr[:end], pod[:end], kn


def nice_step(xmax):
    for step in (0.01, 0.02, 0.025, 0.05, 0.1, 0.2):
        if xmax / step <= 7:
            return step
    return 0.2


def draw_panel(ax, curves, title, xmax=None):
    """curves: list of (key, tau, SR, POD, k_native) with mean-over-runs SR/POD (already truncated)."""
    if xmax is None:
        xmax = max(float(np.nanmax(sr)) for _, _, sr, _, _ in curves) * 1.08
    sr = np.linspace(1e-4, xmax, 300)
    pod = np.linspace(1e-4, 1, 300)
    SR, POD = np.meshgrid(sr, pod)
    CSI = 1.0 / (1.0 / SR + 1.0 / POD - 1.0)
    step = nice_step(xmax)
    levels = np.arange(step, xmax, step)
    cs = ax.contour(SR, POD, CSI, levels=levels, colors="0.7", linewidths=0.5)
    y_lab = 0.93  # put CSI labels in a row near the top, where the isolines are near-vertical
    manual = [(1.0 / (1.0 / c - 1.0 / y_lab + 1.0), y_lab) for c in levels]
    ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=5, inline=True, inline_spacing=1, manual=manual)
    for b in (0.5, 1, 2, 4):
        x = np.array([0, min(xmax, 1 / b)])
        ax.plot(x, b * x, ls=(0, (3, 2)), color="0.55", lw=0.5, zorder=1)
        xl, yl = (xmax, b * xmax) if b * xmax <= 1 else (1 / b, 1.0)
        ax.text(xl, yl, f" {b:g}", fontsize=5, color="0.4",
                ha="right" if b * xmax <= 1 else "center", va="bottom")
    for key, tau, s, p, kn in curves:
        st = STYLE.get(key, dict(color="k", marker="x"))
        ax.plot(s, p, color=st["color"], lw=1.0, zorder=3)
        ax.plot(s[kn], p[kn], ls="none", ms=4.5, mec="k", mew=0.4, zorder=4, **st)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_xlabel("Success ratio (1 − FAR)")
    ax.grid(False)


def make_figure(panels, out, width=3.5):
    """panels: list of (title, curves) for draw_panel."""
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6})
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 2.35), sharey=True)
    axes = np.atleast_1d(axes)
    keys = []
    for ax, (title, curves) in zip(axes, panels):
        draw_panel(ax, curves, title)
        keys += [c[0] for c in curves if c[0] not in keys]
    axes[0].set_ylabel("POD")
    keys = [k for k in ORDER if k in keys] + [k for k in keys if k not in ORDER]
    h = [Line2D([], [], lw=1.0, ms=4, mec="k", mew=0.4, **STYLE.get(k, dict(color="k", marker="x"))) for k in keys]
    l = [DISPLAY.get(k, f"{k[0]}:{k[1]}").replace("Quantile (", "").replace(")", "") for k in keys]
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=5.5, frameon=False, handlelength=1.4,
               columnspacing=0.8, handletextpad=0.4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"performance_diagram.{ext}", dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", type=pathlib.Path, required=True)
    p.add_argument("--output-dir", type=pathlib.Path, default=None)
    p.add_argument("--thresholds", type=float, nargs="+", default=[10.0, 20.0])
    p.add_argument("--tau-max", type=float, default=40.0,
                   help="ignore decision thresholds above this (too few forecasts, noisy SR)")
    p.add_argument("--width", type=float, default=3.5, help="figure width in inches (single column ~3.5)")
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
            curves.append(display_curve(key, tau, sr, pod, ev, args.tau_max))
        if curves:
            panels.append((rf"$\geq${ev:g} mm/h", curves))

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

    make_figure(panels, out, args.width)
    print(f"Saved {out / 'performance_diagram.pdf'}")


if __name__ == "__main__":
    main()
