import json
from flask import Flask, request
import os
import onnx
from onnx import numpy_helper
from json import JSONEncoder
import numpy as np
from inference import preprocess, inference, load_model

app = Flask(__name__)

PATH_MODEL = os.path.join(os.getcwd(), "model/")


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


@app.route("/api/prediction")
def prediction():
    data = request.get_json()
    model_url = os.path.join(PATH_MODEL, data['nom'])
    model = load_model(model_url)
    img = preprocess(data['image'])
    output = inference(model, img)
    return json.dumps(output.to_json())


@app.route("/api/infos", methods=['get'])
def infos():
    data = request.get_json()
    model = onnx.load(os.path.join(PATH_MODEL, data['nom']))
    INTIALIZERS = model.graph.initializer
    w1 = numpy_helper.to_array(INTIALIZERS[0])
    numpyData = {"Weight": w1}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
    return json.dumps(encodedNumpyData)


@app.route("/api/models")
def labels():
    list_model = []
    for file in os.listdir(PATH_MODEL):
        filename = os.fsdecode(file)
        if filename.endswith(".onnx"):
            list_model.append(file)
    return json.dumps(list_model)
