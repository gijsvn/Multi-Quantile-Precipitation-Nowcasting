from typing import List, Dict
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from models.lightning_base import LightningBaseModel
from util.visualization import visualize_precipitation_maps

def count_parameters(model: LightningBaseModel):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def safe_div(num, den):
    return float(num / den) if den != 0 else 0.0

class Evaluator:
    def __init__(
        self, 
        model: LightningBaseModel | str, 
        dataloader: DataLoader,
        distribution_bin_edges: np.ndarray = np.array(
            [0.0, 0.6, 1.2, 2.4, 4.8, 9.6, 19.2, 38.4, 76.8],
            dtype=np.float32
        ) # bin edges for predicted precipitation distribution histogram (in mm/h)
    ) -> None:
        
        self.model = model
        if isinstance(self.model, LightningBaseModel):
            self.model.eval()
            self.device = next(self.model.parameters()).device
        elif self.model != "persistence":
            raise ValueError("Model must be a LightningBaseModel or 'persistence' baseline")

        self.dataloader = dataloader
        self.bin_edges = distribution_bin_edges

        self.mse_loss = torch.nn.MSELoss(reduction="mean")
        self.mae_loss = torch.nn.L1Loss(reduction="mean")

    @torch.no_grad()
    def compute_metrics(self, rain_thresholds: List[float]) -> dict:
        threshold_metrics = None

        bin_counts = np.zeros(len(self.bin_edges) - 1, dtype=np.int64)

        mse_sums = None
        mae_sums = None
        total_samples = 0

        # ------------------------------------------------------
        # Iterate over dataset
        # ------------------------------------------------------
        for x, y_true in tqdm(self.dataloader, desc="Evaluating model"):
            if isinstance(self.model, LightningBaseModel):
                x = x.to(self.device)
                y_true = y_true.to(self.device)

                y_pred = self.model(x)

                if isinstance(self.model, LightningBaseModel):
                    # Convert output from quantile models
                    if y_pred.shape[1] == y_true.shape[1] * 3:
                        y_pred = y_pred.view(y_pred.shape[0], 3, y_true.shape[1], y_pred.shape[2], y_pred.shape[3])

                        y_pred = y_pred[:, 0, :, :, :] # Take 50th percentile prediction
                        # y_pred = y_pred[:, 1, :, :, :] # Take 90th percentile prediction
                        # y_pred = y_pred[:, 2, :, :, :] # Take 95th percentile prediction
            elif self.model == "persistence":
                # Repeat last input image across all output horizons
                y_pred = x[:, -1, :, :].unsqueeze(1).repeat(1, y_true.shape[1], 1, 1)

            if y_pred.shape != y_true.shape:
                raise ValueError(f"Shape mismatch: y_pred {y_pred.shape}, y_true {y_true.shape}")

            B, T, H, W = y_pred.shape

            # Initialize accumulators
            if threshold_metrics is None:
                threshold_metrics: Dict[float, np.ndarray] = {
                    thr: np.zeros((T, 4), dtype=np.float64) for thr in rain_thresholds
                }
                mse_sums = np.zeros(T, dtype=np.float64)
                mae_sums = np.zeros(T, dtype=np.float64)

            # Denormalize predictions and ground truth to mm/5min
            y_pred_mm_5min = y_pred * 47.83
            y_true_mm_5min = y_true * 47.83

            # Compute MSE and MAE per horizon in original units (mm/5min)
            mse_per_horizon = ((y_pred_mm_5min - y_true_mm_5min) ** 2).mean(dim=(0, 2, 3))   # [T]
            mae_per_horizon = torch.abs(y_pred_mm_5min - y_true_mm_5min).mean(dim=(0, 2, 3)) # [T]

            mse_sums += mse_per_horizon.detach().cpu().numpy() * B
            mae_sums += mae_per_horizon.detach().cpu().numpy() * B
            total_samples += B

            # Extrapolate to mm/hour for threshold-based metrics
            y_pred_mm_hour = y_pred_mm_5min * 12
            y_true_mm_hour = y_true_mm_5min * 12

            # Update predicted precipitation distribution histogram
            predicted_values = y_pred_mm_hour.detach().cpu().numpy().ravel()
            counts, _ = np.histogram(predicted_values, bins=self.bin_edges)
            bin_counts += counts

            # Update threshold-based metrics
            for thr in rain_thresholds:
                for h_idx in range(T):
                    y_pred_mask = y_pred_mm_hour[:, h_idx] > thr
                    y_true_mask = y_true_mm_hour[:, h_idx] > thr

                    # 0: tn, 1: fp, 2: fn, 3: tp
                    codes = (y_true_mask.int() * 2 + y_pred_mask.int()).view(-1).cpu().numpy()
                    tn, fp, fn, tp = np.bincount(codes, minlength=4).astype(np.float64)

                    threshold_metrics[thr][h_idx] += np.array([tn, fp, fn, tp], dtype=np.float64)

        # ------------------------------------------------------
        # Process results
        # ------------------------------------------------------
        results = {}

        results["MSE"] = (mse_sums / total_samples).tolist()
        results["MAE"] = (mae_sums / total_samples).tolist()
        results["pred_distribution"] = {
            "bin_edges": self.bin_edges.tolist(),
            "counts": bin_counts.tolist()
        }

        for threshold in threshold_metrics:
            tn = threshold_metrics[threshold][:, 0]
            fp = threshold_metrics[threshold][:, 1]
            fn = threshold_metrics[threshold][:, 2]
            tp = threshold_metrics[threshold][:, 3]

            csi = [safe_div(tp_i, tp_i + fn_i + fp_i) for tp_i, fn_i, fp_i in zip(tp, fn, fp)]
            pod = [safe_div(tp_i, tp_i + fn_i) for tp_i, fn_i in zip(tp, fn)]
            far = [safe_div(fp_i, tp_i + fp_i) for tp_i, fp_i in zip(tp, fp)]

            mcc = []
            for tn_i, fp_i, fn_i, tp_i in zip(tn, fp, fn, tp):
                mcc_den = np.sqrt((tp_i + fp_i) * (tp_i + fn_i) * (tn_i + fp_i) * (tn_i + fn_i))
                mcc.append(safe_div(tp_i * tn_i - fp_i * fn_i, mcc_den))

            results[str(threshold)] = {
                "CSI": csi,
                "POD": pod,
                "FAR": far,
                "MCC": mcc
            }

        return results
    
    @torch.no_grad()
    def visualize_predictions(
        self, 
        indices: List[int]=[222, 444, 777, 1337], 
        title: str|None=None,  
        save_path: str|None=None
    ) -> None:
        xs, ys = [], []
        for idx in indices:
            x, y = self.dataloader.dataset[idx]
            xs.append(x)
            ys.append(y)
        
        x = torch.stack(xs).to(self.device)
        y_true = torch.stack(ys).to(self.device)

        if isinstance(self.model, LightningBaseModel):
            y_pred = self.model(x)
            # Seclect final frame from sequence for visualization
            y_pred = y_pred[:, -1, :, :]
        elif self.model == "persistence":
            y_pred = x[:, -1, :, :]

        visualize_precipitation_maps(
            precipitation_maps=torch.stack([x[:, -1, :, :], y_pred, y_true[:, -1, :, :]]).cpu().numpy(),
            row_labels=[f"Sample {idx}" for idx in indices],
            column_labels=["Input (Persistence)", "Prediction", "Ground Truth"],
            suptitle=title,
            save_path=save_path
        )
