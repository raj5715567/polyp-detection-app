import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import gdown
import os

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="AI Polyp Detection", layout="wide")

# -------------------------
# MODEL DOWNLOAD (FIXED)
# -------------------------
WEIGHTS_PATH = "model.weights.h5"

if not os.path.exists(WEIGHTS_PATH):
    with st.spinner("📥 Downloading model... Please wait"):
        url = "https://drive.google.com/uc?id=1JCO8bi5W1RPUu6xJKVp3m0D-e02cZhrp"
        gdown.download(url, WEIGHTS_PATH, quiet=False, fuzzy=True)

# -------------------------
# MODEL ARCHITECTURE (DeeplabV3)
# -------------------------
def DeeplabV3(input_shape=(256, 256, 3), num_classes=1):

    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=Input(shape=input_shape)
    )

    x = base_model.get_layer('conv4_block6_out').output
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

    x = tf.keras.layers.Conv2D(num_classes, 1, activation='sigmoid')(x)

    return Model(inputs=base_model.input, outputs=x)

# -------------------------
# LOAD MODEL
# -------------------------
def load_model():
    model = DeeplabV3()
    model.load_weights(WEIGHTS_PATH)
    return model

model = load_model()

# -------------------------
# PREPROCESS
# -------------------------
def preprocess(img):
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# -------------------------
# UI DESIGN
# -------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #2c5364);
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.1);
}
.high { color: red; font-size: 24px; }
.low { color: lightgreen; font-size: 24px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 AI Polyp Risk Detection</div>', unsafe_allow_html=True)

# -------------------------
# IMAGE UPLOAD
# -------------------------
uploaded = st.file_uploader("📤 Upload Medical Image", type=["jpg", "png", "jpeg"])

if uploaded:

    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    with st.spinner("🔍 Analyzing..."):

        pred = model.predict(preprocess(image))
        mask = (pred > 0.5).astype(np.uint8).squeeze()

        # Resize mask
        mask_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))

        # Overlay
        overlay = img_np.copy()
        overlay[mask_resized == 1] = [255, 0, 0]
        blended = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)

        # Area calculation
        positive_pixels = np.sum(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        area = positive_pixels / total_pixels

        # Risk logic
        if area > 0.15:
            risk = "HIGH RISK"
        elif area > 0.05:
            risk = "MODERATE RISK"
        else:
            risk = "LOW RISK"

    with col2:
        st.image(blended, caption="Detection Output", use_container_width=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)

        if "HIGH" in risk:
            st.markdown(f'<div class="high">⚠️ {risk}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="low">✅ {risk}</div>', unsafe_allow_html=True)

        st.metric("📏 Area (%)", f"{area*100:.2f}")
        st.metric("🧮 Positive Pixels", int(positive_pixels))

        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("🧬 Segmentation Mask")
    st.image(mask * 255)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("🚀 Final Year Project | AI Medical Assistant")
