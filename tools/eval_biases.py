import os
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from .dino_saliency_map import load_model, load_image

DAYS_IN_YEAR = 365.25


def build_path(row):
    prefix = "MANDRILLS BKB_"
    indiv_id = str(row["id"])
    photo_path = row["photo_path"]
    photo_path = photo_path.split(prefix + f"{indiv_id}_")[-1]
    sub_path = os.path.join(indiv_id, photo_path)
    return indiv_id, sub_path


def foreground_contribution_ratio(attribution, binary_mask):
    total_attributions = np.sum(attribution)

    foreground_attribution = np.where(binary_mask == 1, attribution, 0)
    total_foreground_attributions = np.sum(foreground_attribution)

    return total_foreground_attributions / total_attributions


@click.command()
@click.option(
    "--regression_model",
    required=True,
    help="Path to the pretrained age regression model in torch format.",
)
@click.option(
    "--segmentation_folder",
    required=True,
    help="Path to the foreground/background segmentation folder.",
)
@click.option(
    "--im_folder",
    required=True,
    help="Path to the source image folder.",
)
@click.option(
    "--csv_data",
    required=True,
    help="Path to the result csv data.",
)
@click.option(
    "--device",
    required=True,
    help="Which device to use (cpu, cuda:0).",
)
def main(regression_model, segmentation_folder, im_folder, csv_data, device):
    # Load the regression model
    model = load_model(regression_model, device, dino_type="large")
    model = model.half()
    model.eval()

    # Read csv data
    df = pd.read_csv(csv_data, sep=",")

    # For each image
    for _, row in tqdm(df.iterrows()):
        sub_dir, sub_path = build_path(row)
        im_path = os.path.join(im_folder, sub_path)

        im_name = Path(sub_path).stem
        no_bg_im = os.path.join(segmentation_folder, sub_dir, "masks", f"{im_name}.png")
        assert os.path.exists(no_bg_im), f"Path does not exist: {no_bg_im}"

        # im_path and its binary mask no_bg_im
        std_im, im_tensor = load_image(im_path, device, im_size=(224, 224))

        print(im_path)
        print(row["y_pred"])
        print(row["y_true"])
        # 1. How does the background affect the prediction results ?
        # 1.1 Predict on original image
        pred = model(im_tensor)[0]
        pred_in_days = int(DAYS_IN_YEAR * pred)
        print(pred_in_days)
        input()

        # 1.2 Make background-less image and predict

        # 2. How much background pixels contribute to the prediction ?
        # Compute attribution on a single image

        # Compute (sum foreground contribution) / 1.0

        print(sub_path)
    # def inference_one_image(model, image_path):


if __name__ == "__main__":
    main()
