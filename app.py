import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
import gdown

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="AI Polyp Detection", layout="wide")

MODEL_PATH = "final_model.h5"

# -------------------------
# DOWNLOAD MODEL
# -------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):

        url = "https://drive.google.com/uc?id=1FLxIMcIvtLNdgjNlsRxSNCwFB4b2weYc"

        with st.spinner("📥 Downloading model..."):
            gdown.download(url, MODEL_PATH, quiet=False)

        size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        st.write(f"Downloaded size: {size:.2f} MB")

        if size < 100:
            st.error("❌ Model file corrupted")
            st.stop()

# -------------------------
# LOAD MODEL (FINAL FIX)
# -------------------------
@st.cache_resource
def load_model():

    download_model()

    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"❌ Loading failed: {e}")
        st.stop()

    return model

# -------------------------
# INIT MODEL
# -------------------------
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
st.title("🧠 AI Polyp Detection")

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded:

    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)

    col1, col2 = st.columns(2)

    with st.spinner("🔍 AI Analyzing..."):

        pred = model.predict(preprocess(image))
        mask = (pred > 0.5).astype(np.uint8).squeeze()

        mask_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))

        overlay = img_np.copy()
        overlay[mask_resized == 1] = [255,0,0]

        blended = cv2.addWeighted(img_np,0.7,overlay,0.3,0)

    with col1:
        st.image(image, caption="Original")

    with col2:
        st.image(blended, caption="Detection")

    st.image(mask*255, caption="Mask")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("🚀 Final Year Project")
