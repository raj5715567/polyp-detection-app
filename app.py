import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import cv2
from PIL import Image
import gdown
import os

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Polyp Detection AI", layout="wide")

st.title("🧠 AI Polyp Detection")

# -------------------------
# DOWNLOAD MODEL
# -------------------------
WEIGHTS_PATH = "model.weights.h5"

def download_model():
    if not os.path.exists(WEIGHTS_PATH):
        url = "https://drive.google.com/uc?id=1JCO8bi5W1RPUu6xJKVp3m0D-e02cZhrp"
        gdown.download(url, WEIGHTS_PATH, quiet=False)

# -------------------------
# MODEL ARCHITECTURE (SAME AS TRAINING)
# -------------------------
def DeeplabV3(input_shape=(256, 256, 3), num_classes=1):

    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=Input(shape=input_shape)
    )

    layer_names = ['conv4_block6_out', 'conv5_block3_out']
    layers = [base_model.get_layer(name).output for name in layer_names]

    # ASPP
    b4 = tf.keras.layers.GlobalAveragePooling2D()(layers[-1])
    b4 = tf.keras.layers.Reshape((1, 1, b4.shape[-1]))(b4)
    b4 = tf.keras.layers.Conv2D(256, 1, padding='same', use_bias=False)(b4)
    b4 = tf.keras.layers.BatchNormalization()(b4)
    b4 = tf.keras.layers.Activation('relu')(b4)
    b4 = tf.keras.layers.UpSampling2D(
        size=(layers[-1].shape[1] // b4.shape[1],
              layers[-1].shape[2] // b4.shape[2]),
        interpolation='bilinear')(b4)

    # branches
    def conv_block(x, filters, rate=1):
        x = tf.keras.layers.Conv2D(filters, 3 if rate > 1 else 1,
                                  padding='same',
                                  dilation_rate=rate,
                                  use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.Activation('relu')(x)

    b0 = conv_block(layers[-1], 256)
    b1 = conv_block(layers[-1], 256, 6)
    b2 = conv_block(layers[-1], 256, 12)
    b3 = conv_block(layers[-1], 256, 18)

    x = tf.keras.layers.Concatenate()([b4, b0, b1, b2, b3])
    x = tf.keras.layers.Conv2D(256, 1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

    x = tf.keras.layers.Conv2D(num_classes, 1, name='output_layer')(x)
    x = tf.keras.layers.Activation('sigmoid')(x)
    x = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(x)

    return Model(inputs=base_model.input, outputs=x)

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model_weights():
    download_model()
    model = DeeplabV3()

    # 🔥 FIX (IMPORTANT)
    model.load_weights(
        WEIGHTS_PATH,
        by_name=True,
        skip_mismatch=True
    )

    return model

model = load_model_weights()

# -------------------------
# PREPROCESS
# -------------------------
def preprocess(image):
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# -------------------------
# UI
# -------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Original Image", use_column_width=True)

    # prediction
    pred = model.predict(preprocess(image))
    mask = (pred > 0.5).astype(np.uint8).squeeze()

    st.image(mask * 255, caption="Predicted Mask", use_column_width=True)
