"""
Compare performance different precipitation nowcasting models.

The script print tables comparing performance metrics, and generates
plots and visualizes predictions for all models in a specified folder.
"""

import argparse
import pathlib
import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

from util.load_dataset import PrecipitationDataModule
from util.visualization import visualize_precipitation_maps
from models.lightning_base import LightningBaseModel
from models.SmaAt_UNet.model import SmaAt_UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare trained precipitation nowcasting models."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["val", "test"],
        default="test",
        help="Dataset split to evaluate: 'val' or 'test'.",
    )

    parser.add_argument(
        "--model-folder",
        type=pathlib.Path,
        required=True,
        help="Directory containing one subfolder per trained model.",
    )

    parser.add_argument(
        "--data-file",
        type=pathlib.Path,
        required=True,
        help="Path to the HDF5 precipitation dataset.",
    )

    parser.add_argument(
        "--output-folder",
        type=pathlib.Path,
        default=None,
        help="Directory where generated figures are saved. Defaults to model-folder.",
    )

    parser.add_argument(
        "--n-input-imgs",
        type=int,
        default=18,
        help="Number of input precipitation frames.",
    )

    parser.add_argument(
        "--n-output-imgs",
        type=int,
        default=12,
        help="Number of output precipitation frames.",
    )

    parser.add_argument(
        "--thresholds",
        type=str,
        nargs="+",
        default=["0.5", "10.0", "20.0"],
        help="Precipitation thresholds used evaluation.",
    )

    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=[444, 750, 777, 1321, 1333],
        help="Sample indices to visualize.",
    )

    parser.add_argument(
        "--lead-time-indices",
        type=int,
        nargs="+",
        default=[2, 5, 8, 11], # 15, 30, 45 & 60 minutes ahead at 5-minute intervals
        help="Output frame indices to visualize.",
    )

    parser.add_argument(
        "--pred-interval",
        type=int,
        default=5,
        help="Time interval between predicted frames, in minutes.",
    )

    parser.add_argument(
        "--graph-labels",
        type=str,
        nargs="+",
        default=None,
        help="Optional display labels for models.",
    )

    parser.add_argument(
        "--prediction-labels",
        type=str,
        nargs="+",
        default=None,
        help="Optional display labels for models.",
    )

    return parser.parse_args()

def print_metrics_tables(
        model_folder: pathlib.Path,
        dataset: str,
        threshold_keys: list[str],
    ) -> None:

    rows_by_threshold: dict[str, list[dict]] = {
        threshold: [] for threshold in threshold_keys
    }

    for cur_model in sorted(model_folder.iterdir()):
        if not cur_model.is_dir():
            continue

        results_path = cur_model / f"{dataset}_results.json"

        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
        else:
            print(f"Could not find results: '{results_path}' for {cur_model}. Skipping model!")
            continue

        for threshold in threshold_keys:
            row = {
                "Model": cur_model.name,
                "CSI": np.mean(results[threshold]["CSI"]),
                "POD": np.mean(results[threshold]["POD"]),
                "FAR": np.mean(results[threshold]["FAR"]),
                "MCC": np.mean(results[threshold]["MCC"]),
            }

            if threshold == threshold_keys[0]:
                row["MSE"] = np.mean(results["MSE"])
                row["# params"] = results["parameters"]["trainable"]

            rows_by_threshold[threshold].append(row)
        
    print("\n\n"+"#"*100+"\n\n")
    for threshold in threshold_keys:
        table = pd.DataFrame(rows_by_threshold[threshold])

        if threshold == threshold_keys[0]:
            table = table[["Model", "MSE", "CSI", "POD", "FAR", "MCC", "# params"]]
        else:
            table = table[["Model", "CSI", "POD", "FAR", "MCC"]]

        print(f"Precipitation >= {threshold} mm/h\n")
        print(table)
        print("\n" + "#" * 100 + "\n")

def plot_csi_across_lead_times(
        model_folder: pathlib.Path,
        dataset: str,
        n_output_imgs: int,
        threshold_keys: list[str],
        output_path: pathlib.Path | None = None,
        pred_interval: int = 5,
        labels: list[str] | None = None,
    ) -> None:

    if output_path is None:
        output_path = model_folder / "csi_across_lead_times.png"

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "legend.title_fontsize": 13,
    })

    n_thresholds = len(threshold_keys)
    fig, ax = plt.subplots(nrows=1, ncols=n_thresholds, figsize=(18, 5))

    forecasting_times = np.arange(
        start=pred_interval, 
        stop=(n_output_imgs+1)*pred_interval, 
        step=pred_interval
    )

    markers = ["o", "s", "^", "^", "D", "x", "v"]

    model_idx = 0
    for cur_model in sorted(model_folder.iterdir()):
        if not cur_model.is_dir() or cur_model.name == "persistence":
            continue

        results_path = cur_model / f"{dataset}_results.json"

        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
        else:
            print(f"Could not find results: '{results_path}' for {cur_model}. Skipping model!")
            continue

        if labels is not None:
            if model_idx >= len(labels):
                raise ValueError("Number of graph labels is smaller than number of plotted models.")
            label = labels[model_idx]
        else:
            label = cur_model.name

        for idx, threshold in enumerate(threshold_keys):
            ax[idx].plot(
                forecasting_times, 
                results[threshold]["CSI"], 
                marker=markers[model_idx % len(markers)], 
                markersize=8,
                label=label
            )

        for idx in range(n_thresholds):
            ax[idx].set_title(f"CSI (≥ {threshold_keys[idx]} mm/h)")
            ax[idx].set_xlabel("Nowcasting Time (Minutes)")
            ax[idx].set_ylabel("CSI")
            ax[idx].grid(True, alpha=.5)
            ax[idx].legend()

        model_idx += 1

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

def visualize_predictions_across_lead_times(
        model_folder: pathlib.Path,
        data_file_path: pathlib.Path,
        dataset: str,
        n_input_imgs: int,
        n_output_imgs: int,
        output_folder: pathlib.Path,
        sample_indices: list[int],
        lead_time_indices: list[int],
        pred_interval: int = 5,
        labels: list[str] | None = None,
    ) -> None:

    output_folder.mkdir(parents=True, exist_ok=True)

    models = []
    for cur_model in sorted(model_folder.iterdir()):
        if not cur_model.is_dir() or cur_model.name == "persistence":
            continue

        model_path = cur_model / "model.ckpt"
        if not model_path.exists():
            print(f"Could not find checkpoint: '{model_path}' for {cur_model}. Skipping model!")
            continue

        config_path = cur_model / "config.json"
        if not config_path.exists():
            print(f"Could not find config: '{config_path}' for {cur_model}. Skipping model!")
            continue

        with open(config_path, 'r') as f:
            config = json.load(f)

        backbone = SmaAt_UNet(
            in_channels=n_input_imgs,
            out_channels=n_output_imgs*(1 if "quantile" not in config['loss'].lower() else 3),
            bilinear=True,
            reduction_ratio=16,
        )

        model = LightningBaseModel.load_from_checkpoint(
            checkpoint_path=model_path,
            backbone=backbone,
            weights_only=False,
            strict=True
        ).to(device).eval()

        models.append({
            "model": model,
            "name": cur_model.name,
            "config": config
        })

    data_module = PrecipitationDataModule(
        file_path=data_file_path,
        n_input_imgs=n_input_imgs,
        n_output_imgs=n_output_imgs,
        batch_size=1,
        num_workers=0,
        val_fraction=0.1
    )

    if dataset == "val":
        data_module.setup("fit")
        dataloader = data_module.val_dataloader()
    elif dataset == "test":
        data_module.setup("test")
        dataloader = data_module.test_dataloader()

    for sample_nr in sample_indices:
        x, y_true = dataloader.dataset[sample_nr]

        x = x.to(device)
        x = x.unsqueeze(0) # add batch dimension

        auto_labels = ["Ground Truth"]

        precipitation_maps = [y_true.numpy()[lead_time_indices, :, :]]
        with torch.no_grad():
            for model in models:
                y_pred = model["model"](x)

                is_quantile_model = "quantile" in model['config'].get("loss", "").lower()
                n_quantiles = len(model['config'].get("quantiles", [])) if is_quantile_model else 1

                # For quantile models, include all quantiles
                if is_quantile_model:
                    y_pred = y_pred.view(y_pred.shape[0], n_quantiles, n_output_imgs, y_pred.shape[-2], y_pred.shape[-1])
                    for i in range(n_quantiles):
                        cur_y_pred = y_pred[:, i, :, :, :]
                        cur_y_pred = cur_y_pred.detach().cpu().numpy()
                        cur_y_pred = cur_y_pred[0, lead_time_indices, :, :]
                        precipitation_maps.append(cur_y_pred)
                        auto_labels.append(f"{model['name']} q{i+1}")
                else:
                    y_pred = y_pred.detach().cpu().numpy()
                    y_pred = y_pred[0, lead_time_indices, :, :]
                    precipitation_maps.append(y_pred)
                    auto_labels.append(model["name"])

        precipitation_maps = np.array(precipitation_maps)
        precipitation_maps = np.transpose(precipitation_maps, (1, 0, 2, 3)) # reorder to (time, model, height, width)

        current_labels = labels if labels is not None else auto_labels

        output_path = output_folder / f"{dataset}_{sample_nr}_predictions.png"
        visualize_precipitation_maps(
            precipitation_maps=precipitation_maps,
            row_labels=current_labels,
            column_labels=[
                f"t + {(idx + 1) * pred_interval} minutes"
                for idx in lead_time_indices
            ],
            save_path=output_path,
        )

        print(f"Sample {sample_nr} saved at: {output_path}")

if __name__ == "__main__":
    args = parse_args()

    output_folder = args.output_folder
    if output_folder is None:
        output_folder = args.model_folder

    output_folder.mkdir(parents=True, exist_ok=True)

    print_metrics_tables(
        model_folder=args.model_folder,
        dataset=args.dataset,
        threshold_keys=args.thresholds,
    )

    plot_csi_across_lead_times(
        model_folder=args.model_folder,
        dataset=args.dataset,
        n_output_imgs=args.n_output_imgs,
        threshold_keys=args.thresholds,
        output_path=output_folder / "csi_across_lead_times.png",
        pred_interval=args.pred_interval,
        labels=args.graph_labels,
    )

    visualize_predictions_across_lead_times(
        model_folder=args.model_folder,
        data_file_path=args.data_file,
        dataset=args.dataset,
        n_input_imgs=args.n_input_imgs,
        n_output_imgs=args.n_output_imgs,
        output_folder=output_folder,
        sample_indices=args.sample_indices,
        lead_time_indices=args.lead_time_indices,
        pred_interval=args.pred_interval,
        labels=args.prediction_labels,
    )