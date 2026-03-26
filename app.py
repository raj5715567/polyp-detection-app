import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import base64
import os
from huggingface_hub import hf_hub_download
import shutil

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="AI Polyp Detection", layout="wide")

MODEL_PATH = "model.weights.h5"

# -------------------------
# DOWNLOAD MODEL (ONLY ADDITION)
# -------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading AI Model..."):

            file_path = hf_hub_download(
                repo_id="raj571556/model.weights.h5",
                filename="model.weights.h5"
            )

            shutil.copy(file_path, MODEL_PATH)

# -------------------------
# UI STYLE (UNCHANGED)
# -------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #141e30, #243b55);
    color: white;
}
.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
}
.card {
    padding: 25px;
    border-radius: 20px;
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
}
.high { color: #ff4b5c; font-size: 24px; font-weight: bold; }
.low { color: #00e676; font-size: 24px; font-weight: bold; }
.medium { color: #ffd54f; font-size: 24px; font-weight: bold; }
img {
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 AI Polyp Risk Detection</div>', unsafe_allow_html=True)

# -------------------------
# MODEL (EXACT SAME)
# -------------------------
def DeeplabV3(input_shape=(256,256,3), num_classes=1):

    base_model = tf.keras.applications.ResNet50(
        weights=None,
        include_top=False,
        input_tensor=Input(shape=input_shape)
    )

    layer_names = ['conv4_block6_out','conv5_block3_out']
    layers = [base_model.get_layer(name).output for name in layer_names]

    b4 = tf.keras.layers.GlobalAveragePooling2D()(layers[-1])
    b4 = tf.keras.layers.Reshape((1,1,b4.shape[-1]))(b4)
    b4 = tf.keras.layers.Conv2D(256,1,padding='same',use_bias=False)(b4)
    b4 = tf.keras.layers.BatchNormalization()(b4)
    b4 = tf.keras.layers.Activation('relu')(b4)
    b4 = tf.keras.layers.UpSampling2D(size=(layers[-1].shape[1],layers[-1].shape[2]), interpolation='bilinear')(b4)

    b0 = tf.keras.layers.Conv2D(256,1,padding='same',use_bias=False)(layers[-1])
    b0 = tf.keras.layers.BatchNormalization()(b0)
    b0 = tf.keras.layers.Activation('relu')(b0)

    b1 = tf.keras.layers.Conv2D(256,3,padding='same',dilation_rate=6,use_bias=False)(layers[-1])
    b1 = tf.keras.layers.BatchNormalization()(b1)
    b1 = tf.keras.layers.Activation('relu')(b1)

    b2 = tf.keras.layers.Conv2D(256,3,padding='same',dilation_rate=12,use_bias=False)(layers[-1])
    b2 = tf.keras.layers.BatchNormalization()(b2)
    b2 = tf.keras.layers.Activation('relu')(b2)

    b3 = tf.keras.layers.Conv2D(256,3,padding='same',dilation_rate=18,use_bias=False)(layers[-1])
    b3 = tf.keras.layers.BatchNormalization()(b3)
    b3 = tf.keras.layers.Activation('relu')(b3)

    x = tf.keras.layers.Concatenate()([b4,b0,b1,b2,b3])
    x = tf.keras.layers.Conv2D(256,1,padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.UpSampling2D(size=(4,4), interpolation='bilinear')(x)

    x = tf.keras.layers.Conv2D(1,(1,1))(x)
    x = tf.keras.layers.Activation('sigmoid')(x)
    x = tf.keras.layers.UpSampling2D(size=(8,8), interpolation='bilinear')(x)

    return Model(inputs=base_model.input, outputs=x)

# -------------------------
# LOAD MODEL (EXACT SAME + BUILD FIX)
# -------------------------
@st.cache_resource
def load_model_weights():
    download_model()

    model = DeeplabV3()

    # IMPORTANT: build model (cloud fix)
    model(np.zeros((1,256,256,3)))

    # STRICT loading (same as your local)
    model.load_weights(MODEL_PATH)

    return model

model = load_model_weights()
