import click
import os
import torch
from .dataset import SAM2LikeTransform
from .module import SegmentationLightningModule
from .napari_viz import ImageViewer, infer_im
from glob import glob
from pathlib import Path
from tqdm import tqdm

os.environ["PROJECT_ROOT"] = ".."


def load_top_folder(top_folder_path):
    sub_folders = glob(os.path.join(top_folder_path, "*"))
    all_paths = []
    for folder in sub_folders:
        paths = load_im_folder(folder)
        all_paths += paths
    return all_paths


def get_paths(im_folder, im_path):
    im_name = Path(im_path).stem
    groundtruth_folder = os.path.join(im_folder, "masks")
    groundtruth_path = os.path.join(groundtruth_folder, f"{im_name}.png")
    return im_folder, groundtruth_folder, im_path, groundtruth_path


def load_im_folder(im_folder):
    paths = []
    ims = glob(os.path.join(im_folder, "*.jpg"))
    for im in ims:
        paths.append(get_paths(im_folder, im))
    return paths


def load_model(ckpt_path):
    print(f"Loading model: {ckpt_path}")
    model = SegmentationLightningModule(ckpt_path=None, num_classes=1)
    state_dict = torch.load(ckpt_path, weights_only=False)["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded")
    return model


def infer_all_images(model, paths):
    transforms = SAM2LikeTransform(is_train=False)
    for im_folder, gt_folder, im_path, gt_path in tqdm(paths):
        if os.path.exists(gt_path):
            continue
        infer_im(im_path, gt_path, gt_folder, transforms, model)


@click.command()
@click.option(
    "--model_folder",
    required=True,
    help="Path to the pretrained model.",
)
@click.option(
    "--top_folder",
    required=True,
    help="Path to the image dataset top folder.",
)
@click.option(
    "--interactive_mode",
    type=bool,
    default=True,
    show_default=True,
    help="Whether to use the interactive mode (image viewer) or infer silently.",
)
@click.option(
    "--device",
    required=True,
    default="cpu",
    help="The device to use for inference.",
)
def main(model_folder, top_folder, interactive_mode, device):
    # Load model
    model = load_model(os.path.join(model_folder, "checkpoints", "last.ckpt"))
    model = model.to(device)
    # A dataset is formatted as follow:
    # top_folder/
    #   <id>/
    #       masks/img0.jpg
    #       img0.jpg
    all_paths = load_top_folder(top_folder)
    print(f"Found {len(all_paths)} images")

    if interactive_mode:
        # In interactive mode, we start a napari viewer that display all image with a mask if it exists
        #   You can go through all images
        #   Perform the inference on an image
        #   Delete a mask (if it is not good enough)
        ImageViewer(model, all_paths)
    else:
        # In offline mode we perform the inference on all the images and replace existing masks
        infer_all_images(model, all_paths)


if __name__ == "__main__":
    main()
