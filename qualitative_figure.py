"""
Compact, quantitative qualitative-example figure (reviewer R3 / R5).

Two sub-commands:

  scan    Run MSE, MAE and quantile checkpoints over the whole test set once.
          Writes per-sample statistics (GT peak, area >= 10 mm/h, per-model
          CSI@10 and peak), a stratified summary (do the gains concentrate in
          convective samples?), a ranked shortlist of convective candidates,
          and optional quick-look PNGs of the top candidates.

  render  Draw the compact figure for one sample: GT / MSE / MAE / q0.50 /
          q0.95 at a few lead times, GT 10 mm/h contour on every panel and
          per-panel CSI@10 + peak intensity.

Example
-------
python qualitative_figure.py scan \
    --data-file data/precipitation_dataset.h5 \
    --ckpt-dir checkpoints/paper --out-dir results/qualitative --quicklooks 12

python qualitative_figure.py render \
    --data-file data/precipitation_dataset.h5 \
    --ckpt-dir checkpoints/paper --out-dir results/qualitative --sample 1321

The paper checkpoints have no config.json and the quantile checkpoint pickles
a loss class from an old module path, so the weights are loaded directly from
the state dict (quantiles 0.5, 0.9, 0.95 in that head order).
"""

import argparse
import pathlib
import pickle
import types

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from models.SmaAt_UNet.model import SmaAt_UNet

SCALING = 47.83          # normalized -> mm / 5 min
TO_MMH = SCALING * 12    # normalized -> mm / h
EXTENT = [3.38, 7.84, 50.82, 53.48]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output name -> (checkpoint folder, quantile head or None)
OUTPUTS = {
    "MSE": ("MSE", None),
    "MAE": ("MAE", None),
    "q0.50": ("Quantile", 0),
    "q0.90": ("Quantile", 1),
    "q0.95": ("Quantile", 2),
}


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _LenientUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return _Stub


_PICKLE = types.SimpleNamespace(__name__="lenient_pickle", Unpickler=_LenientUnpickler, load=pickle.load)


def load_backbone(ckpt_path: pathlib.Path, n_in: int) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, pickle_module=_PICKLE)
    sd = {k[len("backbone."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("backbone.")}
    n_out = sd["outc.conv.weight"].shape[0]
    net = SmaAt_UNet(in_channels=n_in, out_channels=n_out, bilinear=True, reduction_ratio=16)
    net.load_state_dict(sd, strict=True)
    return net.to(DEVICE).eval()


def load_models(ckpt_dir: pathlib.Path, n_in: int) -> dict:
    return {
        folder: load_backbone(ckpt_dir / folder / "model.ckpt", n_in)
        for folder in sorted({f for f, _ in OUTPUTS.values()})
    }


@torch.no_grad()
def predict(models: dict, x: torch.Tensor, n_out: int) -> dict:
    """x: (B, T_in, H, W) normalized. Returns name -> (B, T_out, H, W) in mm/h."""
    raw = {f: m(x.to(DEVICE)) for f, m in models.items()}
    preds = {}
    for name, (folder, head) in OUTPUTS.items():
        y = raw[folder]
        if head is not None:
            y = y.view(y.shape[0], -1, n_out, y.shape[-2], y.shape[-1])[:, head]
        preds[name] = (y * TO_MMH).float().cpu().numpy()
    return preds


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
class H5Test:
    def __init__(self, path: pathlib.Path, n_in: int, n_out: int):
        self.f = h5py.File(path, "r")["test"]
        self.n_in, self.n_out = n_in, n_out

    def __len__(self):
        return self.f["images"].shape[0]

    def batch(self, idx):
        imgs = self.f["images"][idx[0]:idx[-1] + 1] if _contiguous(idx) else np.stack([self.f["images"][i] for i in idx])
        x = torch.from_numpy(imgs[:, :self.n_in].astype(np.float32))
        y = imgs[:, self.n_in:self.n_in + self.n_out].astype(np.float32) * TO_MMH
        return x, y

    def timestamp(self, i, lead=0):
        try:
            t = self.f["timestamps"][i, self.n_in + lead, 0]
            return t.decode() if isinstance(t, bytes) else str(t)
        except Exception:
            return ""


def _contiguous(idx):
    return len(idx) == idx[-1] - idx[0] + 1


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def contingency(pred, true, thr):
    """pred/true: (..., H, W) mm/h -> tp, fp, fn summed over the last two axes."""
    p, t = pred > thr, true > thr
    return (p & t).sum((-2, -1)), (p & ~t).sum((-2, -1)), (~p & t).sum((-2, -1))


def csi(tp, fp, fn):
    den = tp + fp + fn
    return np.where(den > 0, tp / np.maximum(den, 1), np.nan)


# --------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------
def scan(args):
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    ds = H5Test(args.data_file, args.n_input_imgs, args.n_output_imgs)
    models = load_models(args.ckpt_dir, args.n_input_imgs)
    thr = args.threshold
    leads = args.lead_time_indices

    rows, counts = [], []   # counts: per sample, per output, summed over all leads
    n = len(ds) if args.max_samples is None else min(len(ds), args.max_samples)
    for start in tqdm(range(0, n, args.batch_size), desc="Scanning test set"):
        idx = np.arange(start, min(start + args.batch_size, n))
        x, y = ds.batch(idx)
        preds = predict(models, x, args.n_output_imgs)

        for b, i in enumerate(idx):
            g = y[b]
            r = {
                "sample": int(i),
                "time": ds.timestamp(i),
                "gt_peak": float(g.max()),
                "gt_area10_mean": float((g > thr).sum((-2, -1)).mean()),   # pixels >= thr, mean over leads
                "gt_area20_mean": float((g > 20).sum((-2, -1)).mean()),
                "gt_wet_frac": float((g > 0.5).mean()),
            }
            for L in leads:
                r[f"gt_peak_t{L}"] = float(g[L].max())
            c = {"sample": int(i)}
            for name, p in preds.items():
                pb = p[b]
                for t_ in (thr, 20.0, 0.5):
                    tp, fp, fn = contingency(pb, g, t_)
                    c[f"{name}_tp{t_:g}"], c[f"{name}_fp{t_:g}"], c[f"{name}_fn{t_:g}"] = tp.sum(), fp.sum(), fn.sum()
                tp, fp, fn = contingency(pb, g, thr)
                r[f"{name}_csi10_all"] = float(csi(tp.sum(), fp.sum(), fn.sum()))
                r[f"{name}_peak"] = float(pb.max())
                r[f"{name}_mse"] = float(((pb - g) ** 2).mean())
                for L in leads:
                    r[f"{name}_csi10_t{L}"] = float(csi(tp[L], fp[L], fn[L]))
                    r[f"{name}_peak_t{L}"] = float(pb[L].max())
            rows.append(r)
            counts.append(c)

    df = pd.DataFrame(rows)
    cdf = pd.DataFrame(counts)
    df.to_csv(out / "per_sample_stats.csv", index=False)
    cdf.to_csv(out / "per_sample_counts.csv", index=False)

    write_stratified(df, cdf, out, thr)
    cand = rank_candidates(df, leads)
    cand.head(50).to_csv(out / "candidates.csv", index=False)
    print("\nTop convective candidates (see candidates.csv):")
    show = ["sample", "time", "gt_peak", "gt_area10_mean", "MSE_csi10_all", "MAE_csi10_all",
            "q0.50_csi10_all", "q0.95_csi10_all", "gain_q50_vs_best"]
    print(cand[show].head(15).to_string(index=False, float_format="%.3f"))

    # Quick-looks skip cartopy: no coastline download needed on compute nodes.
    args.cartopy = False
    for s in cand["sample"].head(args.quicklooks):
        render_sample(ds, models, int(s), args, out / "quicklooks", quick=True)


def write_stratified(df, cdf, out, thr):
    """Pooled CSI/POD/FAR per GT-convection stratum: is the gain in convective samples?"""
    a = df["gt_area10_mean"]
    strata = pd.Series("no >=10 mm/h", index=df.index)
    conv = a > 0
    if conv.sum() >= 3:
        q = a[conv].quantile([1 / 3, 2 / 3]).values
        strata[conv & (a <= q[0])] = "low convective"
        strata[conv & (a > q[0]) & (a <= q[1])] = "mid convective"
        strata[conv & (a > q[1])] = "high convective"
    else:
        strata[conv] = "convective"
    cdf = cdf.assign(stratum=strata.values)

    lines = []
    order = ["no >=10 mm/h", "low convective", "mid convective", "high convective", "convective", "all"]
    for t_ in (thr, 20.0, 0.5):
        for s in order:
            sub = cdf if s == "all" else cdf[cdf.stratum == s]
            if len(sub) == 0:
                continue
            row = {"threshold": t_, "stratum": s, "n": len(sub)}
            if s != "all":
                row["area10_range"] = f"{a[cdf.stratum == s].min():.0f}-{a[cdf.stratum == s].max():.0f}"
            for name in OUTPUTS:
                tp, fp, fn = (sub[f"{name}_{k}{t_:g}"].sum() for k in ("tp", "fp", "fn"))
                row[f"{name}_CSI"] = tp / max(tp + fp + fn, 1)
                row[f"{name}_POD"] = tp / max(tp + fn, 1)
                row[f"{name}_FAR"] = fp / max(tp + fp, 1)
            lines.append(row)
    st = pd.DataFrame(lines)
    st.to_csv(out / "stratified_summary.csv", index=False)

    md = ["# Pooled scores per convection stratum", "",
          "Strata: samples by mean number of GT pixels >= 10 mm/h per lead time "
          "(none, then terciles of the rest). Scores pooled over samples and all 12 lead times.", ""]
    for t_ in (thr, 20.0, 0.5):
        sub = st[st.threshold == t_]
        md += [f"## CSI at {t_:g} mm/h", "",
               "| Stratum | n | " + " | ".join(OUTPUTS) + " |",
               "|---|---|" + "---|" * len(OUTPUTS)]
        for _, r in sub.iterrows():
            md.append(f"| {r.stratum} | {r.n} | " + " | ".join(f"{r[f'{m}_CSI']:.3f}" for m in OUTPUTS) + " |")
        md.append("")
    (out / "stratified_summary.md").write_text("\n".join(md))
    print("\n".join(md))


def rank_candidates(df, leads):
    """Strongly convective samples, ranked by area >= 10 mm/h. Reports the q0.50 gain too,
    so we can pick a case that is typical of its stratum rather than the most flattering one."""
    d = df[df.gt_area10_mean > 0].copy()
    d["gain_q50_vs_best"] = d["q0.50_csi10_all"] - d[["MSE_csi10_all", "MAE_csi10_all"]].max(axis=1)
    d["gain_q95_vs_best"] = d["q0.95_csi10_all"] - d[["MSE_csi10_all", "MAE_csi10_all"]].max(axis=1)
    cut = d.gt_area10_mean.quantile(2 / 3) if len(d) > 3 else 0
    top = d[d.gt_area10_mean >= cut]
    if len(top):
        med = top["gain_q50_vs_best"].median()
        d["typicality"] = (d["gain_q50_vs_best"] - med).abs()
    # Drop near-duplicates: consecutive samples overlap heavily in time.
    d = d.sort_values("gt_area10_mean", ascending=False)
    keep, taken = [], []
    for _, r in d.iterrows():
        if all(abs(r["sample"] - t) > 12 for t in taken):
            keep.append(r)
            taken.append(r["sample"])
    return pd.DataFrame(keep)


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------
RAIN_BOUNDS = [0.1, 0.5, 1, 2, 5, 10, 20, 30, 50]
RAIN_COLORS = ["#c6dbef", "#6baed6", "#2171b5", "#41ab5d", "#fed976",
               "#fd8d3c", "#e31a1c", "#800026"]


def _rain_cmap():
    cmap = ListedColormap(RAIN_COLORS)
    cmap.set_under((1, 1, 1, 0))
    cmap.set_over("#4a0018")
    return cmap, BoundaryNorm(RAIN_BOUNDS, cmap.N)


def _crop_slices(gt, thr, pad, shape):
    """Bounding box (in pixels) around GT >= thr over the shown leads, padded, made square-ish."""
    m = (gt > thr).any(0)
    if not m.any():
        return slice(None), slice(None)
    r, c = np.where(m)
    r0, r1 = max(r.min() - pad, 0), min(r.max() + pad, shape[0])
    c0, c1 = max(c.min() - pad, 0), min(c.max() + pad, shape[1])
    return slice(r0, r1), slice(c0, c1)


def _crop_extent(rs, cs, shape):
    H, W = shape
    lon0, lon1, lat0, lat1 = EXTENT
    r0, r1 = rs.start or 0, rs.stop or H
    c0, c1 = cs.start or 0, cs.stop or W
    return [lon0 + (lon1 - lon0) * c0 / W, lon0 + (lon1 - lon0) * c1 / W,
            lat1 - (lat1 - lat0) * r1 / H, lat1 - (lat1 - lat0) * r0 / H]


def render_sample(ds, models, sample, args, out_dir, quick=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y = ds.batch(np.array([sample]))
    preds = predict(models, x, args.n_output_imgs)
    gt = y[0]
    leads = args.lead_time_indices
    names = args.outputs
    thr = args.threshold

    fields = {"Observed": gt}
    fields.update({n: preds[n][0] for n in names})
    labels = {"Observed": "Observed", "MSE": "MSE", "MAE": "MAE",
              "q0.50": r"$q_{0.50}$", "q0.90": r"$q_{0.90}$", "q0.95": r"$q_{0.95}$"}

    H, W = gt.shape[-2:]
    if args.crop == "auto":
        rs, cs = _crop_slices(gt[leads], thr, args.crop_pad, (H, W))
    elif args.crop == "none":
        rs, cs = slice(None), slice(None)
    else:  # r0,r1,c0,c1
        r0, r1, c0, c1 = map(int, args.crop.split(","))
        rs, cs = slice(r0, r1), slice(c0, c1)
    ext = _crop_extent(rs, cs, (H, W))

    use_cartopy = args.cartopy
    if use_cartopy:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
        except ImportError:
            use_cartopy = False

    rows = list(fields)
    cols = leads
    if args.layout == "models-as-columns":
        nr, nc = len(cols), len(rows)
    else:
        nr, nc = len(rows), len(cols)

    aspect = (ext[3] - ext[2]) / (ext[1] - ext[0])  # panel height / width (PlateCarree, as in the paper)
    pw = args.width / (nc + 0.35)
    ph = pw * aspect
    fig_h = ph * nr + 0.55

    kw = {"subplot_kw": {"projection": ccrs.PlateCarree()}} if use_cartopy else {}
    fig, axes = plt.subplots(nr, nc, figsize=(args.width, fig_h), squeeze=False, **kw)
    cmap, norm = _rain_cmap()
    fs = args.fontsize

    gt_c = gt[:, rs, cs]
    for ri, name in enumerate(rows):
        f = fields[name][:, rs, cs]
        for ci, L in enumerate(cols):
            ax = axes[ci, ri] if args.layout == "models-as-columns" else axes[ri, ci]
            if use_cartopy:
                ax.set_extent(ext, crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f2f2f2", zorder=0)
                ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dde8f0", zorder=0)
                ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.4, edgecolor="#555555", zorder=3)
                ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.3, edgecolor="#777777", zorder=3)
                tr = {"transform": ccrs.PlateCarree()}
            else:
                ax.set_facecolor("#f2f2f2")
                tr = {}
            im = ax.imshow(f[L], extent=ext, origin="upper", cmap=cmap, norm=norm,
                           interpolation="nearest", zorder=1, **tr)
            ax.contour(np.linspace(ext[0], ext[1], f.shape[-1]), np.linspace(ext[3], ext[2], f.shape[-2]),
                       gt_c[L], levels=[thr], colors="k", linewidths=0.6, zorder=4, **tr)
            ax.set_xticks([]); ax.set_yticks([])
            # Rasterize background, rain field and coastlines in the PDF (vector
            # versions are ~100 MB); the GT contour and labels stay vector.
            ax.set_rasterization_zorder(3.5)
            if not use_cartopy:
                ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("auto")

            if not getattr(args, "annotate", False):
                continue
            peak = fields[name][L].max()           # full-domain peak
            if name == "Observed":
                txt = f"max {peak:.0f}"
            else:
                tp, fp, fn = contingency(fields[name][L], gt[L], thr)  # full-domain CSI
                c = csi(tp, fp, fn)
                txt = f"CSI {c:.2f}\nmax {peak:.0f}" if np.isfinite(c) else f"CSI –\nmax {peak:.0f}"
            ax.text(0.03, 0.04, txt, transform=ax.transAxes, fontsize=fs - 1.5, va="bottom", ha="left",
                    zorder=5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    lead_lbl = [f"+{(L + 1) * args.pred_interval} min" for L in cols]
    model_lbl = [labels.get(n, n) for n in rows]
    if args.layout == "models-as-columns":
        top, left = model_lbl, lead_lbl
    else:
        top, left = lead_lbl, model_lbl
    for j, t in enumerate(top):
        axes[0, j].set_title(t, fontsize=fs, pad=3)
    for i, t in enumerate(left):
        axes[i, 0].set_ylabel(t, fontsize=fs, labelpad=3)

    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.02, top=1 - 0.3 / fig_h, wspace=0.03, hspace=0.04)
    cax = fig.add_axes([0.915, 0.08, 0.015, 0.84 - 0.3 / fig_h])
    cb = fig.colorbar(im, cax=cax, extend="max", ticks=RAIN_BOUNDS)
    cb.ax.tick_params(labelsize=fs - 1.5)
    cb.set_label("mm/h", fontsize=fs - 1)
    if quick:
        fig.suptitle(f"sample {sample}  {ds.timestamp(sample)}", fontsize=fs, y=1.0)

    stem = f"{'quick_' if quick else ''}sample{sample}"
    for ext_ in (["png"] if quick else ["png", "pdf"]):
        fig.savefig(out_dir / f"{stem}.{ext_}", dpi=args.dpi if not quick else 110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / stem}.*")


def render(args):
    ds = H5Test(args.data_file, args.n_input_imgs, args.n_output_imgs)
    models = load_models(args.ckpt_dir, args.n_input_imgs)
    for s in args.sample:
        render_sample(ds, models, s, args, args.out_dir)


# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-file", type=pathlib.Path, required=True)
    common.add_argument("--ckpt-dir", type=pathlib.Path, default=pathlib.Path("checkpoints/paper"))
    common.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("results/qualitative"))
    common.add_argument("--n-input-imgs", type=int, default=18)
    common.add_argument("--n-output-imgs", type=int, default=12)
    common.add_argument("--pred-interval", type=int, default=5)
    common.add_argument("--threshold", type=float, default=10.0, help="Contour / CSI threshold (mm/h).")
    common.add_argument("--lead-time-indices", type=int, nargs="+", default=[2, 5, 11],
                        help="0-based output frames; default +15, +30, +60 min.")
    # rendering options (also used by scan quick-looks)
    common.add_argument("--outputs", nargs="+", default=["MSE", "MAE", "q0.50", "q0.95"], choices=list(OUTPUTS))
    common.add_argument("--layout", choices=["models-as-columns", "models-as-rows"], default="models-as-columns")
    common.add_argument("--crop", default="none", help="'none', 'auto' (box around GT >= threshold) or 'r0,r1,c0,c1'.")
    common.add_argument("--crop-pad", type=int, default=25)
    common.add_argument("--cartopy", action=argparse.BooleanOptionalAction, default=True)
    common.add_argument("--width", type=float, default=7.0, help="Figure width in inches (PRL full width ~7.1).")
    common.add_argument("--fontsize", type=float, default=8)
    common.add_argument("--dpi", type=int, default=300)
    common.add_argument("--annotate", action=argparse.BooleanOptionalAction, default=False,
                        help="Write per-panel CSI and peak-intensity labels (off by default).")

    s = sub.add_parser("scan", parents=[common])
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--max-samples", type=int, default=None)
    s.add_argument("--quicklooks", type=int, default=12)

    r = sub.add_parser("render", parents=[common])
    r.add_argument("--sample", type=int, nargs="+", required=True)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    scan(a) if a.cmd == "scan" else render(a)
