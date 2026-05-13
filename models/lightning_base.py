from typing import Tuple
import torch
from torch import nn, optim
import pytorch_lightning as pl

class LightningBaseModel(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        learning_rate: float=1e-3,
        lr_patience: int=5,
        loss: nn.Module=nn.MSELoss(reduction="sum"),
        quantiles: list[float]|None=None,
        n_output_imgs: int=12
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["backbone"])

        self.backbone = backbone
        self.loss = loss
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles) if quantiles is not None else 1
        self.n_output_imgs = n_output_imgs

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        return self.backbone(model_input)
        
    def _reshape_quantile_output(self, y_hat: torch.Tensor) -> torch.Tensor:
        """
        Convert model output from (B, Q*T, H, W) to (B, Q, T, H, W)
        if quantiles are being used.
        """
        if self.quantiles is None:
            return y_hat

        if self.n_output_imgs is None:
            raise ValueError("n_output_imgs must be set when using quantiles")

        if y_hat.ndim != 4:
            raise ValueError(
                f"Expected raw quantile model output of shape (B,Q*T,H,W), got {y_hat.shape}"
            )

        b, c, h, w = y_hat.shape
        expected_c = self.n_quantiles * self.n_output_imgs
        if c != expected_c:
            raise ValueError(
                f"Expected {expected_c} output channels (=Q*T), got {c}. "
                f"Q={self.n_quantiles}, T={self.n_output_imgs}"
            )

        y_hat = y_hat.view(b, self.n_quantiles, self.n_output_imgs, h, w)

        return y_hat

    def _get_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Convert model output to (B, Q, T, H, W) for quantile loss
        if self.quantiles is not None:
            y_hat = self._reshape_quantile_output(y_hat)

        return (self.loss(y_hat, y) / y.size(0)) # Normalize by batch size

    def training_step(self, batch: Tuple[torch.tensor, torch.tensor], batch_idx: int) -> torch.Tensor:
        x, y_true = batch
        y_pred = self(x)

        # NOTE: I don't think I need this, but keep it for now, remove later
        # if y_hat.dim() == y.dim() + 1 and y_hat.size(1) == 1:
        #     y_hat = y_hat.squeeze(1)

        loss = self._get_loss(y_pred, y_true)

        self.log(f"train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch: Tuple[torch.tensor, torch.tensor], batch_idx: int) -> None:
        x, y_true = batch
        y_pred = self(x)

        # NOTE: I don't think I need this, but keep it for now, remove later
        # if y_hat.dim() == y.dim() + 1 and y_hat.size(1) == 1:
        #     y_hat = y_hat.squeeze(1)

        loss = self._get_loss(y_pred, y_true)

        self.log(f"val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch: Tuple[torch.tensor, torch.tensor], batch_idx: int) -> None:
        x, y_true = batch
        y_pred = self(x)

        # NOTE: I don't think I need this, but keep it for now, remove later
        # if y_hat.dim() == y.dim() + 1 and y_hat.size(1) == 1:
        #     y_hat = y_hat.squeeze(1)

        loss = self._get_loss(y_pred, y_true)

        self.log(f"test_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self) -> dict:
        opt = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.1,
            patience=self.hparams.lr_patience,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss",
            },
        }