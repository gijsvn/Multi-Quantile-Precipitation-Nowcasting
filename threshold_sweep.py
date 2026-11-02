"""
Decision-threshold sweep for the POD/FAR trade-off analysis (R2 / editor).

For every checkpoint and every output channel (MSE, MAE, or q50/q90/q95 of a
quantile model) this script thresholds the forecast at a grid of *decision*
thresholds tau (mm/h) while the *event* threshold on the observation stays
fixed (e.g. 10 and 20 mm/h). It stores the full contingency table
(TN, FP, FN, TP) per event threshold, decision threshold and lead time, so
performance diagrams (POD vs. success ratio = 1 - FAR = precision),
precision-recall curves and POD at matched FAR can be derived afterwards
with `plot_performance_diagram.py`.

At tau == event threshold the counts are identical to those of eval.py
(same "> threshold" convention, same mm/h conversion), so the native operating
points of Tables 1-2 lie on the curves.

Usage:
    python threshold_sweep.py --data-file data.h5 --output-dir sweep_results \
        --run MSE=/path/to/mse_run_1 --run MSE=/path/to/mse_run_2 ... \
        --run Quantile=/path/to/quantile_run_1 ...

Each --run is LABEL=PATH where PATH is a run directory (the checkpoint with
the lowest val_loss in PATH/checkpoints is used, as in the paper) or a .ckpt
file. Quantile models are detected from the number of output channels and
their quantile levels are read from the checkpoint hyper-parameters.
"""

import argparse
import io
import json
import pathlib
import pickle
import re
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.SmaAt_UNet.model import SmaAt_UNet
from util.load_dataset import PrecipitationDataset

MM_5MIN_SCALE = 47.83  # same denormalisation as util/evaluate_model.py


# ----------------------------------------------------------------------------
# Checkpoint loading (robust to pickled objects from the original thesis code)
# ----------------------------------------------------------------------------
class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state


class _LenientUnpickler(pickle.Unpickler):
    """Replaces classes that cannot be imported (e.g. util.custom_losses) by a stub."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return _Stub


class _LenientPickleModule:
    Unpickler = _LenientUnpickler
    load = pickle.load
    __name__ = "lenient_pickle"


def resolve_checkpoint(path: pathlib.Path) -> pathlib.Path:
    if path.is_file():
        return path
    ckpts = sorted(path.glob("checkpoints/*.ckpt")) + sorted(path.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt found in {path}")
    with_loss = [c for c in ckpts if "val_loss=" in c.name]
    if with_loss:
        return min(with_loss, key=lambda c: float(re.search(r"val_loss=([0-9.eE+-]+?)\.ckpt$", c.name).group(1)))
    return max(ckpts, key=lambda c: c.stat().st_mtime)


def load_model(ckpt_path: pathlib.Path, n_input_imgs: int, n_output_imgs: int, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, pickle_module=_LenientPickleModule)
    state = ckpt.get("state_dict", ckpt)
    state = {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith("loss.")}

    out_w = [v for k, v in state.items() if k.startswith("outc.") and k.endswith("weight")]
    if not out_w:
        raise KeyError(f"{ckpt_path}: cannot find output layer (outc.*) in state dict")
    out_channels = out_w[0].shape[0]
    n_q = out_channels // n_output_imgs

    hp = ckpt.get("hyper_parameters", {}) or {}
    quantiles = hp.get("quantiles")
    if quantiles is not None:
        quantiles = [float(q) for q in quantiles]
    if n_q == 1:
        quantiles = None
    elif quantiles is None or len(quantiles) != n_q:
        raise ValueError(f"{ckpt_path}: {n_q} output groups but quantiles in hparams = {quantiles}")

    backbone = SmaAt_UNet(
        in_channels=n_input_imgs,
        out_channels=out_channels,
        kernels_per_layer=2,
        bilinear=True,
        reduction_ratio=16,
    )
    backbone.load_state_dict(state, strict=True)
    backbone.eval().to(device)
    return backbone, quantiles


# ----------------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------------
@torch.no_grad()
def sweep(model, quantiles, loader, n_output_imgs, event_thrs, tau, device, max_batches=None):
    """Returns counts[Q, E, T, K, 4] with order (TN, FP, FN, TP)."""
    n_q = 1 if quantiles is None else len(quantiles)
    E, K, T = len(event_thrs), len(tau), n_output_imgs
    counts = np.zeros((n_q, E, T, K, 4), dtype=np.int64)
    tau_t = torch.tensor(tau, dtype=torch.float32, device=device)
    offsets = (torch.arange(T, device=device) * (K + 1)).view(1, T, 1, 1)

    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x = x.to(device).float()
        y = y.to(device).float()
        pred = model(x)
        B, _, H, W = pred.shape
        pred = pred.view(B, n_q, T, H, W) * MM_5MIN_SCALE * 12  # mm/h
        obs = y * MM_5MIN_SCALE * 12

        for e, thr in enumerate(event_thrs):
            pos = obs > thr  # [B, T, H, W]
            n_pos = pos.sum(dim=(0, 2, 3)).cpu().numpy()
            n_tot = B * H * W
            for q in range(n_q):
                # bucket index i means tau[i-1] < p <= tau[i]; p > tau[k] <=> i > k
                idx = torch.bucketize(pred[:, q].contiguous(), tau_t, right=False) + offsets
                hist_pos = torch.bincount(idx[pos], minlength=T * (K + 1)).view(T, K + 1)
                hist_all = torch.bincount(idx.view(-1), minlength=T * (K + 1)).view(T, K + 1)
                # count of p > tau[k] = sum over buckets k+1..K
                above_pos = hist_pos.flip(1).cumsum(1).flip(1)[:, 1:].cpu().numpy()  # [T, K]
                above_all = hist_all.flip(1).cumsum(1).flip(1)[:, 1:].cpu().numpy()
                tp = above_pos
                fp = above_all - above_pos
                fn = n_pos[:, None] - tp
                tn = (n_tot - n_pos)[:, None] - fp
                counts[q, e] += np.stack([tn, fp, fn, tp], axis=-1)
    return counts


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-file", type=pathlib.Path, required=True)
    p.add_argument("--output-dir", type=pathlib.Path, required=True)
    p.add_argument("--run", action="append", required=True, help="LABEL=PATH (run dir or .ckpt); repeat")
    p.add_argument("--event-thresholds", type=float, nargs="+", default=[0.5, 10.0, 20.0])
    p.add_argument("--n-taus", type=int, default=80, help="log-spaced decision thresholds in [tau-min, tau-max]")
    p.add_argument("--tau-min", type=float, default=0.05)
    p.add_argument("--tau-max", type=float, default=100.0)
    p.add_argument("--n-input-imgs", type=int, default=18)
    p.add_argument("--n-output-imgs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=None, help="for quick tests only")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, torch.get_num_threads()))

    # Log grid plus the event thresholds themselves (native operating points)
    tau = np.unique(np.concatenate([
        np.geomspace(args.tau_min, args.tau_max, args.n_taus),
        np.array(args.event_thresholds),
    ])).astype(np.float32)

    ds = PrecipitationDataset(args.data_file, args.n_input_imgs, args.n_output_imgs, test=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Test samples: {len(ds)}, device: {device}, {len(tau)} decision thresholds", flush=True)

    # Group runs per label to number them
    run_index = {}
    for spec in args.run:
        if "=" not in spec:
            sys.exit(f"--run must be LABEL=PATH, got {spec}")
        label, path = spec.split("=", 1)
        run_index[label] = run_index.get(label, 0) + 1
        run_no = run_index[label]
        out_file = args.output_dir / f"{label}_run{run_no}.npz"
        if out_file.exists():
            print(f"Skipping {out_file.name} (exists)", flush=True)
            continue

        ckpt = resolve_checkpoint(pathlib.Path(path))
        model, quantiles = load_model(ckpt, args.n_input_imgs, args.n_output_imgs, device)
        channels = ["point"] if quantiles is None else [f"q{int(round(q * 100))}" for q in quantiles]
        print(f"[{label} run {run_no}] {ckpt} -> channels {channels}", flush=True)

        t0 = time.time()
        counts = sweep(model, quantiles, loader, args.n_output_imgs, args.event_thresholds, tau, device,
                       args.max_batches)
        np.savez_compressed(
            out_file,
            counts=counts,
            tau=tau,
            event_thresholds=np.array(args.event_thresholds, dtype=np.float32),
            channels=np.array(channels),
            label=label,
            checkpoint=str(ckpt),
        )
        print(f"[{label} run {run_no}] done in {time.time() - t0:.0f} s -> {out_file}", flush=True)

    (args.output_dir / "sweep_args.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()}, indent=2))


if __name__ == "__main__":
    main()
