import torch
import torch.nn as nn

class MultiQuantilePinballLoss(nn.Module):
    """
    pred:   (B, Q, T, H, W)
    target: (B, T, H, W)
    """
    def __init__(
        self,
        quantiles: list[float],
        quantile_weights: list[float] | None = None,
        reduction: str = "sum",
    ) -> None:
        super().__init__()

        if len(quantiles) == 0:
            raise ValueError("quantiles must not be empty")
        if any(q <= 0.0 or q >= 1.0 for q in quantiles):
            raise ValueError(f"All quantiles must be in (0,1), got {quantiles}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction: {reduction} (valid: 'mean', 'sum', 'none')")

        self.register_buffer(
            "quantiles",
            torch.tensor(quantiles, dtype=torch.float32)
        )

        if quantile_weights is None:
            quantile_weights = [1.0] * len(quantiles)
        if len(quantile_weights) != len(quantiles):
            raise ValueError("quantile_weights must match quantiles length")

        self.register_buffer(
            "quantile_weights",
            torch.tensor(quantile_weights, dtype=torch.float32)
        )

        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 5:
            raise ValueError(f"pred must be (B,Q,T,H,W), got {pred.shape}")
        if target.ndim != 4:
            raise ValueError(f"target must be (B,T,H,W), got {target.shape}")

        b, q, t, h, w = pred.shape
        if target.shape != (b, t, h, w):
            raise ValueError(
                f"target should match pred except quantile dim, "
                f"got pred={pred.shape}, target={target.shape}"
            )

        target = target.unsqueeze(1)  # (B,1,T,H,W)
        qvals = self.quantiles.view(1, q, 1, 1, 1).to(pred.device)
        qweights = self.quantile_weights.view(1, q, 1, 1, 1).to(pred.device)

        error = target - pred
        loss = torch.maximum(qvals * error, (qvals - 1.0) * error) # Compute pinball losses
        loss = loss * qweights

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            return loss.mean()
        return loss

    def _get_name(self) -> str:
        return f"MultiQuantilePinballLoss(quantiles={self.quantiles}, weights={self.quantile_weights}, reduction={self.reduction})"