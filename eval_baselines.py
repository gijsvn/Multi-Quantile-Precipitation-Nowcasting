"""
Evaluate non-learned nowcasting baselines on the same test set, lead times,
thresholds and metrics as the trained models (Tables 1-2 of the paper):

* persistence:   Eulerian persistence, the last input frame repeated for all lead times.
* extrapolation: Lagrangian extrapolation with pySTEPS, i.e. a Lucas-Kanade
                 motion field estimated from the last input frames and
                 semi-Lagrangian advection of the last input frame.

Both baselines are deterministic, no ensemble/quantile output is produced.
Results are written as <output-dir>/<method>/<evaluation-set>_results.json in
the same format as eval.py, so they can be added to the comparison tables.

Example:
    python eval_baselines.py --data-file data/dataset.h5 --methods persistence extrapolation --num-procs 16
"""

import argparse
import json
import pathlib
import time
import warnings
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch

from util.load_dataset import PrecipitationDataModule
from util.evaluate_model import Evaluator

# Normalized pixel value -> mm/5min (see README); x12 for mm/h
NORM_TO_MM_5MIN = 47.83
NORM_TO_MM_H = NORM_TO_MM_5MIN * 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate persistence and pySTEPS extrapolation baselines.")

    parser.add_argument("--data-file", type=pathlib.Path, required=True, help="Path to the HDF5 precipitation dataset.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("results/baselines"),
                        help="Results go to <output-dir>/<method>/<evaluation-set>_results.json.")
    parser.add_argument("--methods", type=str, nargs="+", choices=["persistence", "extrapolation"],
                        default=["persistence", "extrapolation"], help="Baselines to evaluate.")
    parser.add_argument("--evaluation-set", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--precipitation-thresholds", type=float, nargs="+", default=[0.5, 10.0, 20.0],
                        help="Rainfall thresholds in mm/h used for verification metrics.")
    parser.add_argument("--n-input-imgs", type=int, default=18, help="Number of input frames (paper: 18).")
    parser.add_argument("--n-output-imgs", type=int, default=12, help="Number of lead times (paper: 12).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--val-fraction", type=float, default=0.1)

    # pySTEPS options
    parser.add_argument("--num-procs", type=int, default=1,
                        help="Parallel processes for pySTEPS (one sample per process).")
    parser.add_argument("--motion-frames", type=int, default=4,
                        help="Number of most recent input frames used to estimate the motion field "
                             "(0 = all input frames; the thesis code used all).")
    parser.add_argument("--motion-transform", type=str, choices=["db", "none"], default="db",
                        help="Estimate motion on dB-transformed rain rates (pySTEPS convention) or on raw mm/h.")
    parser.add_argument("--db-threshold", type=float, default=0.1,
                        help="Rain-rate threshold (mm/h) for the dB transform.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Only evaluate the first N samples (quick test run).")

    return parser.parse_args()


def get_dataloader(data_module: PrecipitationDataModule, evaluation_set: str):
    if evaluation_set == "val":
        data_module.setup("fit")
        return data_module.val_dataloader()
    data_module.setup("test")
    return data_module.test_dataloader()


# ----------------------------------------------------------------------------
# pySTEPS Lagrangian extrapolation
# ----------------------------------------------------------------------------
def extrapolate_sample(
    x_norm: np.ndarray,
    n_leadtimes: int,
    motion_frames: int,
    motion_transform: str,
    db_threshold: float,
) -> np.ndarray:
    """x_norm: [T_in, H, W] normalized input -> [n_leadtimes, H, W] normalized forecast."""
    from pysteps import motion, nowcasts
    from pysteps.utils import transformation

    rain_rate = np.clip(x_norm.astype(np.float64), 0.0, None) * NORM_TO_MM_H  # mm/h

    if motion_frames > 0:
        motion_input = rain_rate[-motion_frames:]
    else:
        motion_input = rain_rate

    if motion_transform == "db":
        motion_input, _ = transformation.dB_transform(
            motion_input,
            metadata={"transform": None, "zerovalue": 0.0, "threshold": db_threshold},
            threshold=db_threshold,
            zerovalue=-15.0,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            velocity = motion.get_method("lucaskanade")(motion_input)
        except Exception:
            # e.g. no trackable features in (almost) dry fields: fall back to zero motion
            velocity = np.zeros((2,) + rain_rate.shape[1:])

        forecast = nowcasts.get_method("extrapolation")(
            rain_rate[-1],
            velocity,
            n_leadtimes,
            extrap_kwargs={"allow_nonfinite_values": True},
        )

    # Pixels advected in from outside the domain are NaN: treat as no rain (as in the thesis code)
    forecast = np.nan_to_num(forecast, nan=0.0, posinf=0.0, neginf=0.0)
    forecast = np.clip(forecast, 0.0, None)

    return (forecast / NORM_TO_MM_H).astype(np.float32)


class PySTEPSExtrapolation:
    """Callable baseline for Evaluator: f(x [B, T_in, H, W], n_out) -> [B, n_out, H, W]."""

    def __init__(self, num_procs: int, **sample_kwargs) -> None:
        self.num_procs = num_procs
        self.sample_kwargs = sample_kwargs
        self.pool = Pool(num_procs) if num_procs > 1 else None

    def __call__(self, x: torch.Tensor, n_output_imgs: int) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        fn = partial(extrapolate_sample, n_leadtimes=n_output_imgs, **self.sample_kwargs)

        if self.pool is not None:
            preds = self.pool.map(fn, list(x_np))
        else:
            preds = [fn(sample) for sample in x_np]

        return torch.from_numpy(np.stack(preds))

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


# ----------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    data_module = PrecipitationDataModule(
        file_path=args.data_file,
        n_input_imgs=args.n_input_imgs,
        n_output_imgs=args.n_output_imgs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
    )
    dataloader = get_dataloader(data_module, args.evaluation_set)

    if args.max_samples is not None:
        subset = torch.utils.data.Subset(dataloader.dataset, range(min(args.max_samples, len(dataloader.dataset))))
        dataloader = torch.utils.data.DataLoader(
            subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )

    print(f"Evaluating on {len(dataloader.dataset)} {args.evaluation_set} samples")

    for method in args.methods:
        start = time.time()

        if method == "persistence":
            model = "persistence"
            settings = {}
        else:
            model = PySTEPSExtrapolation(
                num_procs=args.num_procs,
                motion_frames=args.motion_frames,
                motion_transform=args.motion_transform,
                db_threshold=args.db_threshold,
            )
            settings = {
                "motion_method": "lucaskanade",
                "extrapolation": "semilagrangian",
                "motion_frames": args.motion_frames if args.motion_frames > 0 else args.n_input_imgs,
                "motion_transform": args.motion_transform,
                "db_threshold_mm_h": args.db_threshold,
            }

        evaluator = Evaluator(model=model, dataloader=dataloader)
        results = evaluator.compute_metrics(rain_thresholds=args.precipitation_thresholds)

        if isinstance(model, PySTEPSExtrapolation):
            model.close()

        results["parameters"] = {"total": 0, "trainable": 0}
        results["baseline"] = {
            "method": method,
            "n_samples": len(dataloader.dataset),
            "n_input_imgs": args.n_input_imgs,
            "n_output_imgs": args.n_output_imgs,
            "runtime_s": round(time.time() - start, 1),
            **settings,
        }

        out_dir = args.output_dir / method
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{args.evaluation_set}_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        summary = f"{method}: MSE {np.mean(results['MSE']):.4f}  MAE {np.mean(results['MAE']):.4f}"
        for thr in args.precipitation_thresholds:
            summary += f"  CSI@{thr:g} {np.mean(results[str(thr)]['CSI']):.3f}"
        print(summary)
        print(f"Saved {out_file} ({results['baseline']['runtime_s']} s)")


if __name__ == "__main__":
    main(parse_args())
