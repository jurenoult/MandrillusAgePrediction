import numpy as np
import click
from skimage import io
import time
import onnxruntime as ort


def load_model(model_path):
    print(f"Using device : {ort.get_device()}")
    ort_sess = ort.InferenceSession(model_path)
    return ort_sess


def preprocess(image):
    image = image.astype(np.float32) / 255.0
    image = np.moveaxis(image, -1, 0)
    image = np.expand_dims(image, axis=0)
    return image


def inference(model, image):
    image = preprocess(image)
    start_time = time.time()
    outputs = model.run(None, {"input": image})
    end_time = time.time()
    print(f"Inference time took: {(end_time - start_time):.3f} sec")
    return outputs[0]


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
    model = load_model(model_path)
    image = io.imread(image_path)
    age = inference(model, image)
    print(f"Predicted age: {age}")


if __name__ == "__main__":
    main()
