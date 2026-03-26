import streamlit as st
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
import gdown
import os
import base64

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="AI Polyp Detection", layout="wide")

# -------------------------
# DOWNLOAD MODEL FROM DRIVE
# -------------------------
MODEL_PATH = "model.onnx"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading AI Model..."):
        url = "https://drive.google.com/file/d/1G1mavoCNuxyZSRPwRapdaS8RygdxMYHF/view?usp=sharing"
        gdown.download(url, MODEL_PATH, quiet=False)

# -------------------------
# LOAD ONNX MODEL
# -------------------------
@st.cache_resource
def load_model():
    session = ort.InferenceSession(MODEL_PATH)
    return session

session = load_model()

input_name = session.get_inputs()[0].name

# -------------------------
# UI STYLE
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
}
.high { color: #ff4b5c; font-size: 24px; font-weight: bold; }
.medium { color: #ffd54f; font-size: 24px; font-weight: bold; }
.low { color: #00e676; font-size: 24px; font-weight: bold; }
img { border-radius: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 AI Polyp Risk Detection</div>', unsafe_allow_html=True)

# -------------------------
# PREPROCESS
# -------------------------
def preprocess(img):
    img = img.resize((256,256))
    img = np.array(img)/255.0
    img = img.astype(np.float32)
    return np.expand_dims(img, axis=0)

# -------------------------
# UPLOAD
# -------------------------
uploaded = st.file_uploader("📤 Upload Medical Image", type=["jpg","png","jpeg"])

if uploaded:

    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)

    col1, col2 = st.columns(2)

    with st.spinner("🔍 AI Analyzing..."):

        input_data = preprocess(image)

        # ONNX prediction
        output = session.run(None, {input_name: input_data})
        pred = output[0]

        mask = (pred > 0.5).astype(np.uint8).squeeze()

        pixels = np.sum(mask)

        # -------------------------
        # RELATIVE AREA
        # -------------------------
        total_pixels = mask.shape[0] * mask.shape[1]
        relative_area = pixels / total_pixels

        if relative_area > 0.15:
            risk = "HIGH RISK"
        elif relative_area > 0.05:
            risk = "MODERATE RISK"
        else:
            risk = "LOW RISK"

        # -------------------------
        # VISUALIZATION
        # -------------------------
        mask_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))

        overlay = img_np.copy()
        overlay[mask_resized == 1] = [255, 0, 0]

        blended = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)

        contours, _ = cv2.findContours(mask_resized.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (0,255,0), 2)

    # -------------------------
    # DISPLAY
    # -------------------------
    with col1:
        st.image(image, caption="Original Image", width=400)

    with col2:
        st.image(blended, caption="AI Detection Output", width=400)

    # -------------------------
    # RESULTS
    # -------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    c1.metric("📊 Relative Area (%)", f"{relative_area*100:.2f}%")
    c2.metric("🧮 Pixels", int(pixels))
    c3.metric("⚠️ Risk", risk)

    if risk == "HIGH RISK":
        st.error("⚠️ Immediate attention needed")
    elif risk == "MODERATE RISK":
        st.warning("⚠️ Moderate risk detected")
    else:
        st.success("✅ Low risk detected")

    st.progress(min(relative_area, 1.0))

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------
    # REPORT
    # -------------------------
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