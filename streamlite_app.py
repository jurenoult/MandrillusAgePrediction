# -*- coding: utf-8 -*-
from json import JSONEncoder
import json
import pandas as pd
import numpy as np
import streamlit as st
import os
from PIL import Image
import cv2
import shutil
import subprocess
import onnx
from onnx import numpy_helper

INPUT_EXTENSIONS = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp")
PATH_MODEL = os.path.join(os.getcwd(), "model/")  # Folder to find prediction model


def to_img_pil(img_open_cv):
    return Image.fromarray(cv2.cvtColor(img_open_cv, cv2.COLOR_BGR2RGB))


def load_image(image_file):
    img = Image.open(image_file)
    return img


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def clean_streamlit_folder():
    st.legacy_caching.clear_cache()
    if os.path.exists("file.zip"):
        os.remove("file.zip")
    if os.path.isdir(os.getcwd() + "/__pycache__"):
        shutil.rmtree(os.getcwd() + "/__pycache__")
    for root, dirs, files in os.walk(os.getcwd() + "/tmp"):
        for file in files:
            os.unlink(os.path.join(root, file))
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir))


COLOR_BR_r = ['#00CC96', '#EF553B']  # ['dodgerblue', 'indianred']
COLOR_BR = ['indianred', 'dodgerblue']


# Create button to upload an image
def button_image(multiple=False):
    st.subheader("Image")
    image_file = st.file_uploader("Upload Images", type=INPUT_EXTENSIONS, accept_multiple_files=multiple)
    return image_file


# Get models' weight values
def get_info_model(modele):
    # info_model = {'nom': modele}
    # infos = requests.get("http://localhost:5000/api/infos", json=info_model).json()
    model = onnx.load(os.path.join(PATH_MODEL, modele))
    INTIALIZERS = model.graph.initializer
    w1 = numpy_helper.to_array(INTIALIZERS[0])
    numpyData = {"Weight": w1}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
    return json.dumps(encodedNumpyData)


# run prediction on a image with the model in parameters; return prediction's text
def get_inference(modele, image):
    # info_model = {'nom': modele}
    # inference = requests.get("http://localhost:5000/api/prediction", files=image, json=info_model).json()
    # return st.text('\n'.join(inference.split('\n')))
    command = ["python3", 'inference.py', f'--model_path=model/{modele}', f'--image_path={image}']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    stdout, stderr = process.communicate()
    return st.text('\n'.join(stdout.split('\n')[1:][:-1]))


# run prediction on multiple image with the model in parameters and return a dataframe
def get_inference_multiple(model_select, list_path):
    listimage = []
    listime = []
    listprediction = []

    for img in list_path:
        command = ["python3", 'inference.py', f'--model_path=model/{model_select}', f'--image_path={img}']
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        stdout, stderr = process.communicate()
        listimage.append(img)
        listime.append(stdout.split('\n')[1:2][0].split(':')[-1])
        listprediction.append(stdout.split('\n')[2:][0].split(':')[-1])

    df = pd.DataFrame({'Image': listimage, 'Time': listime, 'Prediction': listprediction})
    return df


#######################################################################################


# Convert a Dataframe in csv to download
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Get list of model found in variable PATH_MODEL.
@st.cache_data
def get_liste_model():
    # labels = requests.get("http://localhost:5000/api/models").json()
    # return labels
    list_model = []
# TODO
    for file in os.listdir(PATH_MODEL):
        filename = os.fsdecode(file)
        if filename.endswith(".onnx"):
            print(list_model)
            list_model.append(file)
    return list_model


#######################################################################################


st.set_page_config(page_title="Projet Mar",  # Must be 1st st statement
                   page_icon="🐒",
                   initial_sidebar_state="expanded", layout="wide")

########################################################################################################################

list_model = get_liste_model()
sb_image = 'https://upload.wikimedia.org/wikipedia/commons/9/93/Mandril.jpg'
titre_data = "Prédiction d'age chez les Mandrills"
sb = st.sidebar  # add a side bar
imageLocation = sb.empty()
imageLocation.image(sb_image, width=300)
rad_who = sb.radio('', ['🏠 Accueil', 'Prediction'])  # two versions of the app
# the two versions of the app will have different options, home is common to all
if rad_who == 'Prediction':
    sb.markdown(
        '<h3 style=\'text-align: center;\'> Modèle à utiliser</h3>',
        unsafe_allow_html=True)
    np.random.seed(13)  # one major change is that client is directly asked as input since sidebar
    if not list_model:
        model_select = sb.markdown('Aucun modèle présent dans l\'application')
        rad = sb.radio('', ['🏠 Accueil'])
    else:
        model_select = sb.selectbox('Selectionnez le modèle', list_model)
        rad = sb.radio('', ['🐒 Single Prediction', '🐒 🐒 Multi Prediction',
                            '🔎 Informations du modèle'])

else:
    rad = ''
# defining containers of the app
header = st.container()
dataset = st.container()
eda = st.container()
#######################################################################################
# Implementing containers
#######################################################################################
if not rad:
    with header:
        st.markdown("<h1 style='text-align: center;'>Prediction de l'âge </h1>",
                    unsafe_allow_html=True)
        colT1, colT2 = st.columns([1, 3])
        with colT2:
            st.markdown("**Tout commence dans le menu de gauche :**")
            st.markdown("- Pour prédire à partir d'une image, choisissez prediction à gauche")

#######################################################################################


if rad == '🐒 Single Prediction':

    colT1, colT2 = st.columns([1, 3])
    with colT2:
        image_file = button_image(multiple=False)
        if image_file is not None:
            pic_names = []
            file = image_file.read()  # Read the data
            image_result = open(image_file.name,
                                'wb')  # creates a writable image and later we can write the decoded result
            image_result.write(file)  # Saves the file with the name uploaded_file.name to the root path('./')
            pic_names.append(image_file.name)  # Append the name of image to the list
            image_result.close()  # Close the file pointer
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            # opencv_image = cv2.imdecode(file_bytes, 1)
            for i in range(len(pic_names)):  # Iterating over each file name
                name = pic_names[i]  # Getting the name of current file
                path = './' + pic_names[i]  # Creating path string which is basically ["./image.jpg"]
                test = image_file.getvalue()
                # print(test)
            colT2.image(path, channels="BGR")
            if st.button("Prédire l'age", type="primary"):
                inference = get_inference(model_select, path)

if rad == '🐒 🐒 Multi Prediction':
    df_csv = None
    colT1, colT2 = st.columns([1, 3])
    with colT2:
        image_file = button_image(multiple=True)
        if image_file is not None:
            pic_names = []
            list_path = []
            for uploaded_img in image_file:  # Iterating over each file name
                file = uploaded_img.read()  # Read the data
                image_result = open(uploaded_img.name,
                                    'wb')  # creates a writable image and later we can write the decoded result
                image_result.write(file)  # Saves the file with the name uploaded_file.name to the root path('./')
                pic_names.append(uploaded_img.name)  # Append the name of image to the list
                image_result.close()  # Close the file pointer
                file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
                name = uploaded_img.name  # Getting the name of current file
                path = './' + name  # Creating path string which is basically ["./image.jpg"]
                list_path.append(path)
                test = uploaded_img.getvalue()
                colT2.image(path, channels="BGR") # Comment this line to desactivate visualisation
            if st.button("Générer les résultat", type="primary"):
                df_csv = get_inference_multiple(model_select, list_path)

            if df_csv is not None:
                csv = convert_df(df_csv)
                st.download_button(label="Téléchargez", data=csv, file_name=f'prediction_{model_select}.csv',
                                   mime='text/csv', key='download-csv')

if rad == '🔎 Informations du modèle':
    with eda:
        st.header("**Données du modele" + "** \n ----")
        sous_titre = f"le nom du modèle: {model_select}"
        st.subheader(sous_titre)
        infos = get_info_model(modele=model_select)
        st.json(infos, expanded=False)
        st.markdown("Téléchargez les poids du modele")
        st.download_button(label="Téléchargez", data=infos, file_name='weight.json',
                           mime='text/json', )

#######################################################################################

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#######################################################################################
if __name__ == "__main__":
    print("Script runned directly")
