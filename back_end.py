import json
from flask import Flask, request
import os

app = Flask(__name__)
import pandas as pd
import numpy as np

PATH_MODEL = os.path.join(os.getcwd(), "model/")

train = pd.read_csv(
    "https://raw.githubusercontent.com/pkhetland/Facies-prediction-TIP160/master/datasets/facies_vectors.csv"
)
train = train.rename(columns={"Well Name": "WELL"})


@app.route("/api/data")
def data():
    selector = request.args.get("selector")
    if not selector:
        selector = "SHRIMPLIN"
    # print(selector)
    data = train[train["WELL"].isin([selector])]
    # print(data)
    return json.dumps(data.to_json())


@app.route("/api/models")
def labels():
    list = []
    for file in os.listdir(PATH_MODEL):
        filename = os.fsdecode(file)
        if filename.endswith(".h5"):
            list.append(file)
    return json.dumps(list)

