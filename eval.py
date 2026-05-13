import torch
import json

from util.load_dataset import PrecipitationDataModule
from util.evaluate_model import Evaluator, count_parameters
from models.lightning_base import LightningBaseModel
from models.SmaAt_UNet.model import SmaAt_UNet

def main():
    model_folder = "path/to/model/folder" # Replace with your actual model folder path
    model_folder = "results/SmaAt-UNet_2026-05-12_run-1" # NOTE: remove
    model_name = "model"

    evaluation_set = "test"  # "test" or "val"
    precipitation_thresholds = [0.5, 10.0, 20.0] # in mm/h

    data_file_path = "path/to/data/file.h5" # Replace with your actual data file path
    data_file_path = "../hybrid-nowcasting-thesis-v0/data/kevin_18_18.h5" # NOTE: remove

    model_path = f"{model_folder}/{model_name}.ckpt"

    with open(f"{'/'.join(model_path.split('/')[:-1])}/config.json", "r") as f:
        model_config = json.load(f)

    n_input_imgs = model_config['n_input_imgs']
    n_output_imgs = model_config['n_output_imgs']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = SmaAt_UNet(
        in_channels=n_input_imgs,
        out_channels=n_output_imgs*(3 if "quantile" in model_config.get("loss", "").lower() else 1),
        kernels_per_layer=2,
        bilinear=True,
        reduction_ratio=16
    )

    data = PrecipitationDataModule(
        file_path=data_file_path,
        n_input_imgs=n_input_imgs,
        n_output_imgs=n_output_imgs,
        batch_size=model_config.get("batch_size", 16),
        num_workers=0,
        val_fraction=0.1
    )

    if evaluation_set == "val":
        data.setup("fit")
        dataloader = data.val_dataloader()
    elif evaluation_set == "test":
        data.setup("test")
        dataloader = data.test_dataloader()

    model = LightningBaseModel.load_from_checkpoint(
        checkpoint_path=model_path,
        backbone=backbone,
        strict=True
    ).to(device)

    evaluator = Evaluator(
        model=model,
        dataloader=dataloader
    )

    # results = evaluator.compute_metrics(
    #     rain_thresholds=precipitation_thresholds
    # )

    # total_params, trainable_params = count_parameters(backbone)
    # results['parameters'] = {"total": total_params, "trainable": trainable_params}

    # output_file = f"{'/'.join(model_path.split('/')[:-1])}/{evaluation_set}_results.json"
    # with open(output_file, "w") as f:
    #     json.dump(results, f, indent=4)
    
    output_file = f"{'/'.join(model_path.split('/')[:-1])}/{evaluation_set}_predictions.png"
    evaluator.visualize_predictions(
        title=f"{model_folder}/{model_name}",
        save_path=output_file
    )

if __name__ == "__main__":
    main()