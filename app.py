import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import gdown
import os

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="AI Polyp Detection", layout="wide")

# -------------------------
# DOWNLOAD MODEL
# -------------------------
MODEL_PATH = "model.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading AI Model..."):
        url = "https://drive.google.com/uc?id=1IDbuJqZ5way9b1TPk-W7tTp8iYKUbTJ-"
        gdown.download(url, MODEL_PATH, quiet=False)

# -------------------------
# LOAD MODEL (CORRECT WAY)
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# -------------------------
# PREPROCESS
# -------------------------
def preprocess(img):
    img = img.resize((256,256))
    img = np.array(img)/255.0
    return np.expand_dims(img, axis=0)

# -------------------------
# UI
# -------------------------
st.title("🧠 AI Polyp Risk Detection")

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded:

    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)

    col1, col2 = st.columns(2)

    with st.spinner("🔍 AI Analyzing..."):

        pred = model.predict(preprocess(image))
        mask = (pred > 0.5).astype(np.uint8).squeeze()

        pixels = np.sum(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        relative_area = pixels / total_pixels

        if relative_area > 0.15:
            risk = "HIGH RISK"
        elif relative_area > 0.05:
            risk = "MODERATE RISK"
        else:
            risk = "LOW RISK"

        # Overlay
        mask_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))

        overlay = img_np.copy()
        overlay[mask_resized == 1] = [255, 0, 0]

        blended = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)

    with col1:
        st.image(image, caption="Original Image")

    with col2:
        st.image(blended, caption="Detection Output")

    st.write(f"Area: {relative_area*100:.2f}%")
    st.write(f"Risk: {risk}")

    st.subheader("Segmentation Mask")
    st.image(mask*255)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("🚀 Final Year Project • AI Medical Assistant")
