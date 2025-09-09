import os
import time
import torch
from pathlib import Path
import cv2

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from .dino_saliency_map import load_model, load_image, compute_raw_gradients, get_attribution

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
    "--csv_data",
    required=True,
    help="Path to the result csv data.",
)
@click.option(
    "--device",
    required=True,
    help="Which device to use (cpu, cuda:0).",
)
def main(regression_model, segmentation_folder, csv_data, device):
    # Load the regression model
    model = load_model(regression_model, device, dino_type="large")
    model.eval()

    # Read csv data
    df = pd.read_csv(csv_data, sep=",")

    data = []

    # For each image
    for _, row in tqdm(df.iterrows(), total=len(df)):
        new_row = {}
        sub_dir, sub_path = build_path(row)
        im_path = os.path.join(segmentation_folder, sub_path)
        new_row["relative_image_path"] = sub_path
        new_row["y_true"] = row["y_true"]
        new_row["y_pred"] = row["y_pred"]

        im_name = Path(sub_path).stem
        no_bg_im = os.path.join(segmentation_folder, sub_dir, "masks", f"{im_name}.png")
        assert os.path.exists(no_bg_im), f"Path does not exist: {no_bg_im}"

        # im_path and its binary mask no_bg_im
        std_im, im_tensor = load_image(im_path, device, im_size=(224, 224))

        # 1. How does the background affect the prediction results ?
        # 1.1 Predict on original image
        # with torch.no_grad():
        #     pred = model(im_tensor)[0]
        #     pred_in_days = np.round(DAYS_IN_YEAR * pred.detach().cpu().numpy())
        # pred_in_days = int(row["y_pred"])

        # 1.2 Make background-less image and predict
        binary_im = cv2.imread(no_bg_im)
        std_im_no_bg = np.where(binary_im == 0, 0, std_im)

        # Unnormalize image and save it
        im_no_bg = (std_im_no_bg * 255).astype(np.uint8)
        im_no_bg_dir = os.path.join(segmentation_folder, sub_dir, "no_bg")
        os.makedirs(im_no_bg_dir, exist_ok=True)
        cv2.imwrite(os.path.join(im_no_bg_dir, f"{im_name}.png"), im_no_bg)

        # Create a torch tensor
        input_data_no_bg = torch.from_numpy(std_im_no_bg).unsqueeze(dim=0)

        # Make it channel first
        input_data_no_bg = input_data_no_bg.movedim(-1, 1)

        # Move it to the device
        input_data_no_bg = input_data_no_bg.to(device)

        with torch.no_grad():
            pred_no_bg = model(input_data_no_bg)[0]
            pred_no_bg_in_days = np.round(DAYS_IN_YEAR * pred_no_bg.detach().cpu().numpy())
            new_row["y_pred_no_bg"] = pred_no_bg_in_days

        # 2. How much background pixels contribute to the prediction ?
        # Compute fg/bg ratio
        fg_bg_ratio = np.count_nonzero(binary_im) / np.prod(binary_im.shape)
        new_row["fg_bg_ratio"] = fg_bg_ratio

        # Compute attribution on a single image
        attrs = get_attribution(compute_raw_gradients, model, im_tensor)
        attr_dir = os.path.join(segmentation_folder, sub_dir, "attrs")
        os.makedirs(attr_dir, exist_ok=True)
        cv2.imwrite(os.path.join(attr_dir, f"{im_name}.png"), attrs)

        # Compute (sum foreground contribution) / (sum of all)
        foreground_attribution_ratio = foreground_contribution_ratio(attrs, binary_im)
        new_row["foreground_attribution_ratio"] = foreground_attribution_ratio

        data.append(new_row)

        print(new_row)
        input()

        if len(data) > 10:
            break

    cols = list(data[0].keys())
    df = pd.DataFrame(data, cols)
    df.to_csv("eval_biases_results.csv")


if __name__ == "__main__":
    main()
