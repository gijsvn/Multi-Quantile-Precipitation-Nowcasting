"""
Evaluate a checkpoint on the test set and write the metrics in the same JSON
format as the five-run results on the laptop
(five_run_aggregates/five_run_loss_experiments/<loss>/<run>_test_results.json):
MSE, MAE, pred_distribution and, per threshold, precision, recall, accuracy,
f1, csi, csi_n, far, hss and mcc, plus parameters and checkpoint.

The metric code mirrors util/evaluate_model.py of the thesis code used for the
paper's MSE/MAE/quantile tables, so the new rows are computed the same way.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from util.load_dataset import PrecipitationDataModule
from util.evaluate_model import count_parameters, safe_div
from models.lightning_base import LightningBaseModel
from models.SmaAt_UNet.model import SmaAt_UNet

# Histogram bins in mm/5min, as in the thesis Evaluator
BIN_EDGES = np.array([0.00, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 3.20, 6.40], dtype=np.float32)
NEIGHBORHOOD_RADIUS = 4

def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    x = mask.float().unsqueeze(1)
    k = 2 * radius + 1
    return torch.nn.functional.max_pool2d(x, kernel_size=k, stride=1, padding=radius).squeeze(1) > 0

@torch.no_grad()
def compute_metrics(model, dataloader, thresholds, device) -> dict:
    T = None
    hist = np.zeros(len(BIN_EDGES) - 1, dtype=np.int64)
    n = 0

    for x, y_true in tqdm(dataloader, desc="Evaluating model"):
        x, y_true = x.to(device), y_true.to(device)
        y_pred = model(x)
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred {y_pred.shape}, y_true {y_true.shape}")

        B, T_, _, _ = y_pred.shape
        if T is None:
            T = T_
            counts = {thr: np.zeros((T, 4)) for thr in thresholds}   # tn, fp, fn, tp
            counts_n = {thr: np.zeros((T, 3)) for thr in thresholds} # tp, fp, fn (neighborhood)
            mse_sums = np.zeros(T)
            mae_sums = np.zeros(T)

        pred_5 = y_pred * 47.83
        true_5 = y_true * 47.83

        hist += np.histogram(np.clip(pred_5.cpu().numpy().ravel(), 0.0, None), bins=BIN_EDGES)[0]

        mse_sums += ((pred_5 - true_5) ** 2).mean(dim=(0, 2, 3)).cpu().numpy() * B
        mae_sums += torch.abs(pred_5 - true_5).mean(dim=(0, 2, 3)).cpu().numpy() * B
        n += B

        pred_h = pred_5 * 12
        true_h = true_5 * 12
        for thr in thresholds:
            for h in range(T):
                p = pred_h[:, h] > thr
                t = true_h[:, h] > thr

                p_dil, t_dil = dilate(p, NEIGHBORHOOD_RADIUS), dilate(t, NEIGHBORHOOD_RADIUS)
                counts_n[thr][h] += [
                    (t & p_dil).sum().item(),
                    (p & ~t_dil).sum().item(),
                    (t & ~p_dil).sum().item(),
                ]

                codes = (t.int() * 2 + p.int()).view(-1).cpu().numpy()
                counts[thr][h] += np.bincount(codes, minlength=4).astype(np.float64)

    results = {
        "MSE": (mse_sums / n).tolist(),
        "MAE": (mae_sums / n).tolist(),
        "pred_distribution": {"bin_edges": BIN_EDGES.tolist(), "counts": hist.tolist()},
    }

    for thr in thresholds:
        tn, fp, fn, tp = counts[thr].T
        tp_n, fp_n, fn_n = counts_n[thr].T

        precision = [safe_div(a, a + b) for a, b in zip(tp, fp)]
        recall = [safe_div(a, a + b) for a, b in zip(tp, fn)]
        results[str(thr)] = {
            "precision": precision,
            "recall": recall,
            "accuracy": [safe_div(a + d, a + b + c + d) for a, b, c, d in zip(tp, fp, fn, tn)],
            "f1": [safe_div(2 * p_ * r_, p_ + r_) for p_, r_ in zip(precision, recall)],
            "csi": [safe_div(a, a + b + c) for a, b, c in zip(tp, fn, fp)],
            "csi_n": [safe_div(a, a + b + c) for a, b, c in zip(tp_n, fn_n, fp_n)],
            "far": [safe_div(b, a + b) for a, b in zip(tp, fp)],
            "hss": [
                safe_div(a * d - c * b, (a + c) * (c + d) + (a + b) * (b + d))
                for a, b, c, d in zip(tp, fp, fn, tn)
            ],
            "mcc": [
                safe_div(a * d - b * c, np.sqrt((a + b) * (a + c) * (d + b) * (d + c)))
                for a, b, c, d in zip(tp, fp, fn, tn)
            ],
        }

    return results

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=pathlib.Path, required=True)
    parser.add_argument("--data-file", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--precipitation-thresholds", type=float, nargs="+", default=[0.5, 10.0, 20.0])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    with open(args.model_path.parent / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = SmaAt_UNet(
        in_channels=config["n_input_imgs"],
        out_channels=config["n_output_imgs"],
        kernels_per_layer=2,
        bilinear=True,
        reduction_ratio=16,
    )
    model = LightningBaseModel.load_from_checkpoint(
        checkpoint_path=args.model_path,
        backbone=backbone,
        weights_only=False,
        strict=True,
    ).to(device)
    model.eval()

    data = PrecipitationDataModule(
        file_path=args.data_file,
        n_input_imgs=config["n_input_imgs"],
        n_output_imgs=config["n_output_imgs"],
        batch_size=args.batch_size,
        num_workers=0,
    )
    data.setup("test")

    results = compute_metrics(model, data.test_dataloader(), args.precipitation_thresholds, device)
    total, trainable = count_parameters(backbone)
    results["parameters"] = {"total": total, "trainable": trainable}
    results["checkpoint"] = str(args.model_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
