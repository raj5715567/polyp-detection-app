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

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="AI Polyp Detection", layout="wide")

MODEL_PATH = "model.weights.h5"

# -------------------------
# DOWNLOAD MODEL (SAFE HF)
# -------------------------
from huggingface_hub import hf_hub_download
import shutil

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading AI Model..."):

            file_path = hf_hub_download(
                repo_id="raj571556/model.weights.h5",
                filename="model.weights.h5"
            )

            # ✅ FIX: use copy instead of rename
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
# MODEL (UNCHANGED)
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
# LOAD MODEL (FIXED)
# -------------------------
@st.cache_resource
def load_model_weights():
    download_model()

    model = DeeplabV3()

    # ✅ VERY IMPORTANT: BUILD MODEL FIRST
    model(np.zeros((1,256,256,3)))

    # ✅ SAFE LOAD
    model.load_weights(
        MODEL_PATH,
        by_name=True,
        skip_mismatch=True
    )

    return model

# -------------------------
# LOAD MODEL SAFE
# -------------------------
try:
    model = load_model_weights()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    model = None

# -------------------------
# PARAMETERS (UNCHANGED)
# -------------------------
pixel_per_cm = 100
threshold_cm2 = 1

# -------------------------
# PREPROCESS (UNCHANGED)
# -------------------------
def preprocess(img):
    img = img.resize((256,256))
    img = np.array(img)/255.0
    return np.expand_dims(img, axis=0)

# -------------------------
# UPLOAD (UNCHANGED)
# -------------------------
uploaded = st.file_uploader("📤 Upload Medical Image", type=["jpg","png","jpeg"])

if uploaded and model is not None:

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

        mask_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))

        overlay = img_np.copy()
        overlay[mask_resized==1] = [255,0,0]

        blended = cv2.addWeighted(img_np,0.7,overlay,0.3,0)

        contours,_ = cv2.findContours(mask_resized.astype(np.uint8),
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended,contours,-1,(0,255,0),2)

    with col1:
        st.image(image, caption="Original Image", width=400)

    with col2:
        st.image(blended, caption="AI Detection Output", width=400)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    c1.metric("📊 Relative Area (%)", f"{relative_area*100:.2f}%")
    c2.metric("🧮 Pixels", int(pixels))
    c3.metric("⚠️ Risk", risk)

    if risk == "HIGH RISK":
        st.markdown('<p class="high">⚠️ Immediate attention needed</p>', unsafe_allow_html=True)
    elif risk == "MODERATE RISK":
        st.markdown('<p class="medium">⚠️ Moderate risk detected</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="low">✅ Low risk detected</p>', unsafe_allow_html=True)

    st.progress(min(relative_area,1.0))

    st.markdown('</div>', unsafe_allow_html=True)

    report = f"""
Polyp Detection Report

Relative Area: {relative_area*100:.2f} %
Pixels: {pixels}
Risk Level: {risk}
"""

    b64 = base64.b64encode(report.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="report.txt">📥 Download Report</a>'
    st.markdown(href, unsafe_allow_html=True)

    st.subheader("🧬 Segmentation Mask")
    st.image(mask*255, width=400)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("🚀 Final Year Project • AI Medical Assistant")
