import datetime
import pathlib
import random
import json
import os

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
import numpy as np

from util.load_dataset import PrecipitationDataModule
from util.callbacks import LossPlotCallback, VisualizationCallback
from util.quantile_loss import MultiQuantilePinballLoss
from models.lightning_base import LightningBaseModel
from models.SmaAt_UNet.model import SmaAt_UNet

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
seed_everything(SEED, workers=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def main():
    result_log_dir = pathlib.Path("results")
    os.makedirs(result_log_dir, exist_ok=True)

    data_file_path = "../hybrid-nowcasting-thesis-v0/data/kevin_18_18.h5" # Replace with your actual data file path
    num_workers = 0
        
    n_input_imgs = 18
    n_output_imgs = 12

    batch_size = 2 # NOTE: change back to 16
    learning_rate = 1e-3
    lr_patience = 4

    max_epochs = 300
    es_patience = 15

    # NOTE: loss fixen met een argument
    quantiles = [0.5, 0.9, 0.95]
    higher_q_weights = 0.5

    loss = MultiQuantilePinballLoss(
       quantiles=quantiles,
       quantile_weights=[1.0, higher_q_weights, higher_q_weights],
       reduction="sum"
    )

    # loss = torch.nn.MSELoss(reduction="sum")
    # loss = torch.nn.L1Loss(reduction="sum")

    data = PrecipitationDataModule(
        file_path=data_file_path,
        n_input_imgs=n_input_imgs,
        n_output_imgs=n_output_imgs,
        batch_size=batch_size,
        num_workers=num_workers,
        val_fraction=0.1
    )

    backbone = SmaAt_UNet(
        in_channels=n_input_imgs,
        out_channels=n_output_imgs*3, # NOTE: dit fixen met een argument
        kernels_per_layer=2,
        bilinear=True,
        reduction_ratio=16
    )

    model = LightningBaseModel(
        backbone=backbone,
        learning_rate=learning_rate,
        lr_patience=lr_patience,
        loss=loss,
        quantiles=quantiles,
        n_output_imgs=n_output_imgs
    )

    run_nr = 1
    logging_file_name = backbone.name + "_" + datetime.date.today().strftime("%Y-%m-%d") + f"_run-{run_nr}"
    while os.path.exists(result_log_dir / logging_file_name):
        run_nr += 1
        logging_file_name = backbone.name + "_" + datetime.date.today().strftime("%Y-%m-%d") + f"_run-{run_nr}"

    logger = CSVLogger(
        save_dir=result_log_dir,
        name=logging_file_name,
        version="."
    )

    checkpoint = ModelCheckpoint(
        monitor="val_loss", 
        mode="min", 
        save_top_k=1, 
        save_last=False,
        filename='{epoch}-{val_loss:.5f}'
    )

    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=es_patience,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    loss_plotter = LossPlotCallback()
    prediction_visualizer = VisualizationCallback(val_indices=[50, 150, 333])

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=logger,
        callbacks=[loss_plotter, prediction_visualizer, checkpoint, early_stop_cb, lr_monitor],
        log_every_n_steps=100,
        enable_progress_bar=True
    )

    config_dict = {
        "data_file": data_file_path,
        "n_input_imgs": n_input_imgs,
        "n_output_imgs": n_output_imgs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lr_patience": lr_patience,
        "max_epochs": max_epochs,
        "es_patience": es_patience,
        "model_backbone": backbone.name,
        "loss": loss._get_name(),
        "quantiles": quantiles,
        "quantile_weights": loss.quantile_weights.tolist() if hasattr(loss, "quantile_weights") else None
    }

    os.makedirs(trainer.logger.log_dir, exist_ok=True)
    with open(os.path.join(trainer.logger.log_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    trainer.fit(model, datamodule=data)

if __name__ == "__main__":
    main()