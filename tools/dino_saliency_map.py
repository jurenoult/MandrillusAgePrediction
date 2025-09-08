import torch
import numpy as np
import click
from mandrillage.models import DinoV2, RegressionHead
from skimage import io
import matplotlib.pyplot as plt
import time
import os
from glob import glob
from tqdm import tqdm
import cv2
import torchvision.transforms.functional as TF
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# Follow https://github.com/facebookresearch/dinov2/issues/19
# To update your vision transformer

blue_black_red = LinearSegmentedColormap.from_list(
    "blue_black_red", [(0.0, "blue"), (0.5, "black"), (1.0, "red")], N=256
)


def blur(input_tensor):

    blurred = TF.gaussian_blur(input_tensor, kernel_size=11, sigma=5)
    return blurred


def _cumulative_sum_threshold(values, percentile) -> float:
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id: int = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]


def _normalize_scale(attr, scale_factor):
    # signed attr rescaled
    attr_norm = attr / scale_factor
    # Force [-1; 1]
    attr_norm = np.clip(attr_norm, -1, 1)
    return attr_norm


def _normalize_attr(attr, outlier_perc=0.1, reduction_axis=0):
    # Compute threshold to normalize without outliers
    attr_combined = np.sum(attr, axis=reduction_axis)
    threshold = _cumulative_sum_threshold(np.abs(attr_combined), 100.0 - outlier_perc)

    # Rescale attribution map
    return _normalize_scale(attr_combined, threshold)


def attr_to_gray(attr):
    attr = attr.squeeze().detach().cpu().numpy()

    signed_attr = _normalize_attr(attr)

    # Between -1.0 and 1.0
    assert signed_attr.min() >= -1.0 and attr.max() <= 1.0

    # Take absolute value
    attr_gray = (np.abs(signed_attr) * 255).astype(np.uint8)  # Convert to [0, 255] shape: [H, W]
    return attr_gray, signed_attr


def get_attribution(attr_method, model, input_tensor, normalized=True):
    attr = attr_method(model, input_tensor)
    if normalized:
        return attr_to_gray(attr)
    return attr


def compute_saliency(model, input_tensor):
    from captum.attr import Saliency, NoiseTunnel

    saliency = Saliency(model)
    nt = NoiseTunnel(saliency)
    # blurred = blur(input_tensor)
    attr = nt.attribute(input_tensor, nt_type="smoothgrad", nt_samples=50, stdevs=0.2, target=None)
    return attr


def compute_raw_gradients(model, input_tensor):
    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(model)
    blurred = blur(input_tensor)
    attr = ig.attribute(
        input_tensor, baselines=blurred, target=None, n_steps=50
    )  # Regression → target=None
    return attr


def plot_signed_attribution(ax, im, attr, alpha=0.6, title="Signed Attribution"):
    """
    Visualizes areas that increase or decrease the regression output.
    - Red: increases (e.g., looks older)
    - Blue: decreases (e.g., looks younger)
    """

    # Normalize attribution around 0
    vmax = np.percentile(np.abs(attr), 99)  # robust normalization
    vmin = -vmax


def display_all_methods(model, input_tensor, im, methods_dict):
    n_methods = len(methods_dict)

    fig, axes = plt.subplots(2, n_methods + 1, figsize=(15, 5))

    for i, (method_name, method) in enumerate(methods_dict.items()):
        if i == 0:
            # Original
            axes[0, 0].imshow(im)
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis("off")

        # Display abs and signed attribution map
        attr, attr_signed = get_attribution(method, model, input_tensor)
        attr_signed = gaussian_filter(attr_signed, sigma=1)
        vmax = np.percentile(np.abs(attr_signed), 99)  # robust normalization
        vmin = -vmax

        ax_abs = axes[0, i + 1]
        im_attr = ax_abs.imshow(attr, cmap="hot")
        cbar = plt.colorbar(im_attr, ax=ax_abs, orientation="vertical")
        ax_abs.set_title(f"{method_name}_abs")
        ax_abs.axis("off")

        ax_signed = axes[1, i + 1]
        im_attr_signed = ax_signed.imshow(attr_signed, cmap="seismic", vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im_attr_signed, ax=ax_signed, orientation="vertical")
        ax_signed.set_title(f"{method_name}_signed")
        ax_signed.axis("off")

        ax_overlay = axes[1, 0]
        ax_overlay.imshow(im)
        ax_overlay.imshow(attr_signed, cmap="seismic", alpha=0.6, vmin=vmin, vmax=vmax)
        ax_overlay.set_title("Signed Attribution")
        ax_overlay.axis("off")

    plt.savefig("out.png")


def get_all_method_dict():
    return {
        "Integrated Gradients": compute_raw_gradients,
        # "Saliency": compute_saliency,
    }


def load_model(model_path, device, dino_type="small"):
    # Load a default backbone model
    baseline_backbone_model = DinoV2(dino_type).to("cpu")

    # Load model
    model = torch.load(model_path, weights_only=False).to("cpu")
    backbone_statedict = model.backbone.state_dict()
    baseline_backbone_model.load_state_dict(backbone_statedict)
    model.backbone = baseline_backbone_model

    model = model.to(device)
    print("Model loaded")
    model.eval()

    return model


def load_image(im_path, device, im_size=(224, 224)):
    im = io.imread(im_path)

    # Make sure the image is 224, 224
    if im.shape[0:2] != im_size:
        im = cv2.resize(im, im_size, interpolation=cv2.INTER_AREA)

    # Normalize image to range [0, 1]
    std_im = (im / 255.0).astype(np.float16)

    # Create a torch tensor
    input_data = torch.from_numpy(std_im).unsqueeze(dim=0)

    # Make it channel first
    input_data = input_data.movedim(-1, 1)

    # Move it to the device
    input_data = input_data.to(device)

    return std_im, input_data


@click.command()
@click.option(
    "--model_path",
    required=True,
    help="Path to model to load.",
)
@click.option(
    "--im_path",
    required=False,
    default=None,
    help="Path to the image to process.",
)
@click.option(
    "--im_folder",
    required=False,
    default=None,
    help="Path to the folder to process.",
)
@click.option(
    "--dino_type",
    required=False,
    default="large",
    help="Dino model type (small,medium,large)",
)
@click.option(
    "--device",
    required=True,
    default="cuda:0",
    help="The device to use. Usually either cuda:0 for gpu or cpu",
)
def main(model_path, im_path, im_folder, dino_type, device):
    # Xor
    assert bool(im_path) != bool(
        im_folder
    ), "Expected either image path or folder path but not both"
    print("Starting !")

    # Load model
    t0 = time.time()
    model = load_model(model_path, device, dino_type)
    print(f"Model loaded in {time.time() - t0} s")

    if im_path:
        # Load image
        t0 = time.time()
        std_im, input_tensor = load_image(im_path, device)
        print(f"Image loaded in {time.time() - t0} s")

        # Compute the saliency map
        t0 = time.time()
        saliency_im = get_attribution(compute_saliency, model, input_tensor, normalized=True)
        print(f"Saliency map computed in {time.time() - t0} s")

        # Display and write results
        # display_overlay(saliency_im, std_im)
        display_all_methods(model, input_tensor, std_im, methods_dict=get_all_method_dict())

        output = model(input_tensor)
        print(f"Predicted age is {output}")

    else:
        assert os.path.exists(im_folder)

        files = glob(os.path.join(im_folder, "*", "*.jpg"))
        for im_path in tqdm(files):
            # Skip if selected image is a saliency image
            if im_path.endswith("saliency.jpg"):
                continue
            # Or if saliency has already been computed
            output_path = im_path.replace(".jpg", "_saliency.jpg")
            if os.path.exists(output_path):
                continue

            std_im, input_tensor = load_image(im_path, device)
            gray_saliency_im = get_attribution(compute_raw_gradients, model, input_tensor)
            io.imsave(output_path, gray_saliency_im)


if __name__ == "__main__":
    main()
