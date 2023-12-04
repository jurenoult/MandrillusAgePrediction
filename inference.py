import numpy as np
import torch
from mandrillage.models import DinoV2
import click
from skimage import io


def preload_backbone():
    DinoV2("small")


def load_model(backbone_name, weights_path):
    preload_backbone()

    model = torch.load(weights_path)
    model.eval()
    return model


def preprocess(image):
    image = image.astype(np.float32) / 255.0
    return image


def inference(model, image, device):
    image = preprocess(image)
    image = np.moveaxis(image, -1, 0)
    image = torch.tensor(image)
    image = image.to(device)
    image = torch.unsqueeze(image, dim=0)

    prediction = model(image)
    prediction = prediction[0]

    return prediction.detach().cpu().numpy()


@click.command()
@click.option(
    "--model_path",
    required=True,
    help="Path to model to load.",
)
@click.option(
    "--image_path",
    required=True,
    help="Path to image to use.",
)
def main(model_path, image_path):
    model = load_model("dinov2_medium", model_path)
    image = io.imread(image_path)
    age = inference(model, image, torch.device("cuda"))
    print(f"Predicted age: {age}")


if __name__ == "__main__":
    main()
