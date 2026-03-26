import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import gdown
import os

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Polyp Detection", layout="wide")

# -------------------------
# DOWNLOAD MODEL
# -------------------------
MODEL_PATH = "model.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        url = "https://drive.google.com/uc?id=1JCO8bi5W1RPUu6xJKVp3m0D-e02cZhrp"
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

# -------------------------
# CUSTOM METRICS (IMPORTANT)
# -------------------------
def dice_coefficient(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-7) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-7)

def iou_score(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_bin = K.round(K.clip(y_true_f, 0, 1))
    y_pred_bin = K.round(K.clip(y_pred_f, 0, 1))
    intersection = K.sum(y_true_bin * y_pred_bin)
    union = K.sum(y_true_bin) + K.sum(y_pred_bin) - intersection
    return (intersection + 1e-7) / (union + 1e-7)

def precision(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pp = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return tp / (pp + 1e-7)

def recall(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pos = K.sum(K.round(K.clip(y_true, 0, 1)))
    return tp / (pos + 1e-7)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + 1e-7))

# -------------------------
# LOAD MODEL (FINAL FIX)
# -------------------------
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "dice_coefficient": dice_coefficient,
            "iou_score": iou_score,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        },
        compile=False
    )

model = load_model()

# -------------------------
# PREPROCESS
# -------------------------
def preprocess(img):
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# -------------------------
# UI
# -------------------------
st.title("🧠 AI Polyp Detection System")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:

    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original", use_container_width=True)

    with st.spinner("Analyzing..."):

        pred = model.predict(preprocess(image))
        mask = (pred > 0.5).astype(np.uint8).squeeze()

        mask_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))

        overlay = img_np.copy()
        overlay[mask_resized == 1] = [255, 0, 0]
        blended = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)

        area = np.sum(mask) / (mask.shape[0] * mask.shape[1])

        if area > 0.15:
            risk = "HIGH RISK"
        elif area > 0.05:
            risk = "MODERATE RISK"
        else:
            risk = "LOW RISK"

    with col2:
        st.image(blended, caption="Prediction", use_container_width=True)

        st.subheader("Results")
        st.write(f"Risk: {risk}")
        st.write(f"Area: {area*100:.2f}%")

    st.subheader("Mask")
    st.image(mask * 255)

st.markdown("---")
st.markdown("Final Year Project")
