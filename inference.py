import numpy as np
import torch
from mandrillage.models import DinoV2, SequentialModel, RegressionHead


def preload_backbone():
    DinoV2("medium")


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


def test():
    model = load_model(
        "dinov2_medium",
        "/home/rkarpins/projects/mandrill/mandrillagerecognition/logs/experiments/runs/regression_freeze/checkpoints/regression_0_regression_freeze.h5",
    )
    from skimage import io

    image = io.imread(
        "/home/rkarpins/projects/mandrill/mandrillagerecognition/data/db2/Images/MANDRILLS CIRMF/17B7B/20180802_id17B7B_malsub_(1)_ORIGINAL.jpg"
    )

    print(model)

    age = inference(model, image, torch.device("cuda"))
    print(f"Predicted age: {age}")


if __name__ == "__main__":
    test()
