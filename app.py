import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import gdown
import os

st.set_page_config(page_title="AI Polyp Detection", layout="wide")

# -------------------------
# DOWNLOAD MODEL
# -------------------------
WEIGHTS_PATH = "model.weights.h5"

def download_model():
    if not os.path.exists(WEIGHTS_PATH):
        url = "https://drive.google.com/uc?id=1JCO8bi5W1RPUu6xJKVp3m0D-e02cZhrp"
        gdown.download(url, WEIGHTS_PATH, quiet=False)

# -------------------------
# MODEL (FIXED)
# -------------------------
def DeeplabV3(input_shape=(256,256,3)):

    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',   # ✅ FIX
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
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    download_model()
    model = DeeplabV3()
    model.load_weights(WEIGHTS_PATH)
    return model

model = load_model()

# -------------------------
# PREPROCESS
# -------------------------
def preprocess(img):
    img = img.resize((256,256))
    img = np.array(img)/255.0
    return np.expand_dims(img,0)

# -------------------------
# UI
# -------------------------
st.title("🧠 AI Polyp Detection")

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)

    pred = model.predict(preprocess(image))
    mask = (pred > 0.5).astype(np.uint8).squeeze()

    mask_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))

    overlay = img_np.copy()
    overlay[mask_resized==1] = [255,0,0]

    blended = cv2.addWeighted(img_np,0.7,overlay,0.3,0)

    st.image(image, caption="Original")
    st.image(blended, caption="Prediction")
    st.image(mask*255, caption="Mask")
