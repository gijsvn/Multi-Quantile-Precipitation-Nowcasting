import torch
import torch.nn as nn

# Normalized precipitation * 47.83 = mm/5min, * 12 = mm/h
MM_PER_HOUR_PER_UNIT = 47.83 * 12

class BalancedLoss(nn.Module):
    """
    Balanced MSE / MAE of Shi et al. (2017): every pixel's error is weighted
    by the rain-rate class of its ground truth,

        w(y) = 1 (y < 2), 2 (2 <= y < 5), 5 (5 <= y < 10), 10 (10 <= y < 30), 30 (y >= 30)

    with y in mm/h. The thresholds are converted to the normalized scale of the
    data, so the error itself is computed on the same scale as the MSE/MAE runs.

    pred:   (B, T, H, W)
    target: (B, T, H, W)
    """
    def __init__(
        self,
        kind: str = "mse",
        thresholds_mm_h: list[float] = [2.0, 5.0, 10.0, 30.0],
        weights: list[float] = [1.0, 2.0, 5.0, 10.0, 30.0],
        reduction: str = "sum",
    ) -> None:
        super().__init__()

        if kind not in ("mse", "mae"):
            raise ValueError(f"Invalid kind: {kind} (valid: 'mse', 'mae')")
        if len(weights) != len(thresholds_mm_h) + 1:
            raise ValueError("weights must have one more entry than thresholds_mm_h")
        if list(thresholds_mm_h) != sorted(thresholds_mm_h):
            raise ValueError("thresholds_mm_h must be increasing")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction: {reduction} (valid: 'mean', 'sum', 'none')")

        self.kind = kind
        self.thresholds_mm_h = list(thresholds_mm_h)
        self.reduction = reduction

        self.register_buffer(
            "thresholds",
            torch.tensor(thresholds_mm_h, dtype=torch.float32) / MM_PER_HOUR_PER_UNIT
        )
        self.register_buffer(
            "weights",
            torch.tensor(weights, dtype=torch.float32)
        )

    def pixel_weights(self, target: torch.Tensor) -> torch.Tensor:
        # right=True: thresholds[i-1] <= y < thresholds[i] gets weights[i]
        idx = torch.bucketize(target.float(), self.thresholds, right=True)
        return self.weights[idx]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred and target shapes differ: {pred.shape} vs {target.shape}")

        error = pred - target
        error = error ** 2 if self.kind == "mse" else torch.abs(error)
        loss = self.pixel_weights(target) * error

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.mean()
        return loss

    def _get_name(self) -> str:
        return (
            f"Balanced{self.kind.upper()}Loss(thresholds_mm_h={self.thresholds_mm_h}, "
            f"weights={self.weights.tolist()}, reduction={self.reduction})"
        )
