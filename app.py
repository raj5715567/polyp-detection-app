import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K
import gdown
import os

# -------------------------
# IMPORTANT FIX (KERAS ISSUE)
# -------------------------
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Polyp Detection", layout="wide")

# -------------------------
# CUSTOM METRICS (FROM TRAINING)
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
# DOWNLOAD MODEL
# -------------------------
MODEL_PATH = "model.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model..."):
        url = "https://drive.google.com/uc?id=1IDbuJqZ5way9b1TPk-W7tTp8iYKUbTJ-"
        gdown.download(url, MODEL_PATH, quiet=False)

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
        compile=False,
        safe_mode=False
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
# UI DESIGN
# -------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #2c5364);
}
.title {
    text-align: center;
    font-size: 40px;
    color: white;
    font-weight: bold;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.1);
}
.high { color: red; font-size: 25px; }
.low { color: lightgreen; font-size: 25px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 AI Polyp Detection System</div>', unsafe_allow_html=True)

# -------------------------
# UPLOAD IMAGE
# -------------------------
uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded:

    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Analyzing..."):

        pred = model.predict(preprocess(image))
        mask = (pred > 0.5).astype(np.uint8).squeeze()

        # AREA
        positive_pixels = np.sum(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        area = positive_pixels / total_pixels

        # RISK
        if area > 0.15:
            risk = "HIGH RISK"
        elif area > 0.05:
            risk = "MODERATE RISK"
        else:
            risk = "LOW RISK"

        # OVERLAY
        mask_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))
        overlay = img_np.copy()
        overlay[mask_resized == 1] = [255, 0, 0]
        blended = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)

    with col2:
        st.image(blended, caption="Detection Output", use_container_width=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)

        if "HIGH" in risk:
            st.markdown(f'<div class="high">⚠️ {risk}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="low">✅ {risk}</div>', unsafe_allow_html=True)

        st.metric("Area (%)", f"{area*100:.2f}")
        st.metric("Positive Pixels", int(positive_pixels))

        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("Segmentation Mask")
    st.image(mask * 255)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("🚀 Final Year Project | AI Medical Assistant")
