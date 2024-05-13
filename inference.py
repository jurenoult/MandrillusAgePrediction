import os
import numpy as np
import click
from skimage import io
import time
import onnxruntime as ort
from glob import glob

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
    
    outputs_d = {}
    if len(outputs) == 1: # May be either age or face id vector
        if outputs[0].ndim > 1: # face id
            outputs_d["face_id_vector"] = outputs[0][0].tolist()
        else:
            outputs_d["age"] = outputs[0]
    else: # Multi objective (age, sex, quality)
        outputs_d["age"] = outputs[0][0].tolist()
        outputs_d["sex"] = outputs[1].tolist()
        outputs_d["quality"] = outputs[2].tolist()
    return outputs_d

def find_images(folder, types=["tif","png", "jpg", "jpeg"]):
    files_grabbed = []
    for ext in types:
        files_grabbed.extend(glob(os.path.join(folder, f"*.{ext}")))
    return files_grabbed

def inference_one_image(model, image_path):
    image = io.imread(image_path)
    output = inference(model, image)
    return output

def inference_folder(model, folder_path):
    images_paths = find_images(folder_path)
    outputs = [inference(model, io.imread(image_path)) for image_path in images_paths]
    return outputs

@click.command()
@click.option(
    "--model_path",
    required=True,
    help="Path to model to load.",
)
@click.option(
    "--image_path",
    required=False,
    default=None,
    help="Path to the image to process. Set either image_path or folder_path but not both",
)
@click.option(
    "--folder_path",
    required=False,
    default=None,
    help="Path to the folder to process. Set either image_path or folder_path but not both",
)
def main(model_path, image_path, folder_path):
    image_path_set = image_path is not None
    folder_path_set = folder_path is not None

    # image xor folder
    assert image_path_set != folder_path_set, "Expected either image_path or folder_path to be set but not both."

    model = load_model(model_path)
    if image_path_set:
        output = inference_one_image(model, image_path)
    else:
        output = inference_folder(model, folder_path)
    
    print(f"Predicted : {output}")
    return output

if __name__ == "__main__":
    main()
