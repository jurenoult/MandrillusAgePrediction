# -*- coding: utf-8 -*-
import numpy as np
import streamlit as st
import joblib
import os
from PIL import Image
import cv2
import shutil
import requests

INPUT_EXTENSIONS = ("jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp")


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


def button_image():
    st.subheader("Image")
    image_file = st.file_uploader("Upload Images", type=INPUT_EXTENSIONS)
    return image_file

#######################################################################################

@st.cache_data
def get_data(finename):
    df = joblib.load(finename)
    return df

#######################################################################################


st.set_page_config(page_title="Projet Mar",  # Must be 1st st statement
                   page_icon="🐒",
                   initial_sidebar_state="expanded", layout="wide")


########################################################################################################################
list_model = []

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
        input_client = sb.markdown('Aucun modèle présent dans l\'application')
        rad = sb.radio('', ['🏠 Accueil'])
    else:
        input_client = sb.selectbox('Selectionnez le modèle', list_model)
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
        image_file = button_image()
        if image_file is not None:
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            # Now do something with the image! For example, let's display it:
            colT2.image(opencv_image, channels="BGR")


if rad == '🔎 Informations générales':
    with eda:
        st.header("**Données du modele" + "** \n ----")
        st.subheader("le nom du modèle:")

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