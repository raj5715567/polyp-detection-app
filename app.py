import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import base64
import os
import gdown

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="AI Polyp Detection", layout="wide")

MODEL_PATH = "model.weights.h5"

# -------------------------
# DOWNLOAD MODEL
# -------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):

        url = "https://drive.google.com/uc?id=1JCO8bi5W1RPUu6xJKVp3m0D-e02cZhrp"

        with st.spinner("📥 Downloading model..."):
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

        size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        st.write(f"Downloaded size: {size:.2f} MB")

        if size < 300:
            st.error("❌ Model file corrupted")
            os.remove(MODEL_PATH)
            st.stop()

# -------------------------
# MODEL ARCHITECTURE
# -------------------------
def DeeplabV3(input_shape=(256,256,3), num_classes=1):

    base_model = tf.keras.applications.ResNet50(
        weights=None,
        include_top=False,
        input_tensor=tf.keras.layers.Input(shape=input_shape)
    )

    layer_names = ['conv4_block6_out','conv5_block3_out']
    layers = [base_model.get_layer(name).output for name in layer_names]

    def conv_block(x, filters, rate=1):
        x = tf.keras.layers.Conv2D(filters, 3 if rate > 1 else 1,
                                  padding='same',
                                  dilation_rate=rate,
                                  use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.Activation('relu')(x)

    b4 = tf.keras.layers.GlobalAveragePooling2D()(layers[-1])
    b4 = tf.keras.layers.Reshape((1,1,b4.shape[-1]))(b4)
    b4 = conv_block(b4, 256)
    b4 = tf.keras.layers.UpSampling2D(size=(layers[-1].shape[1], layers[-1].shape[2]), interpolation='bilinear')(b4)

    b0 = conv_block(layers[-1], 256)
    b1 = conv_block(layers[-1], 256, 6)
    b2 = conv_block(layers[-1], 256, 12)
    b3 = conv_block(layers[-1], 256, 18)

    x = tf.keras.layers.Concatenate()([b4, b0, b1, b2, b3])
    x = conv_block(x, 256)
    x = tf.keras.layers.UpSampling2D(size=(4,4), interpolation='bilinear')(x)

    x = tf.keras.layers.Conv2D(1,1)(x)
    x = tf.keras.layers.Activation('sigmoid')(x)
    x = tf.keras.layers.UpSampling2D(size=(8,8), interpolation='bilinear')(x)

    return tf.keras.models.Model(inputs=base_model.input, outputs=x)

# -------------------------
# SMART MODEL LOADER
# -------------------------
def load_model():

    download_model()

    # 🔁 TRY 1: FULL MODEL
    try:
        st.write("🔄 Trying full model load...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.success("✅ Loaded as FULL model")
        return model
    except Exception as e:
        st.warning(f"❌ Full model failed: {e}")

    # 🔁 TRY 2: STRICT WEIGHTS
    try:
        st.write("🔄 Trying strict weights load...")
        model = DeeplabV3()
        model(np.zeros((1,256,256,3)))
        model.load_weights(MODEL_PATH)
        st.success("✅ Loaded as WEIGHTS (strict)")
        return model
    except Exception as e:
        st.warning(f"❌ Strict weights failed: {e}")

    # 🔁 TRY 3: FLEXIBLE WEIGHTS
    try:
        st.write("🔄 Trying flexible weights load...")
        model = DeeplabV3()
        model(np.zeros((1,256,256,3)))
        model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
        st.success("⚠️ Loaded with partial weights")
        return model
    except Exception as e:
        st.warning(f"❌ Flexible weights failed: {e}")

    # ❌ FINAL FAIL
    st.error("🚨 All loading methods failed")
    st.stop()

# -------------------------
# LOAD MODEL
# -------------------------
model = load_model()

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
.high { color: #ff4b5c; font-size: 24px; }
.medium { color: #ffd54f; font-size: 24px; }
.low { color: #00e676; font-size: 24px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 AI Polyp Risk Detection</div>', unsafe_allow_html=True)

# -------------------------
# PREPROCESS
# -------------------------
def preprocess(img):
    img = img.resize((256,256))
    img = np.array(img)/255.0
    return np.expand_dims(img, axis=0)

# -------------------------
# UPLOAD
# -------------------------
uploaded = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

if uploaded:

    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)

    col1, col2 = st.columns(2)

    with st.spinner("🔍 AI Analyzing..."):

        pred = model.predict(preprocess(image))
        mask = (pred > 0.5).astype(np.uint8).squeeze()

        pixels = np.sum(mask)
        total_pixels = mask.size
        relative_area = pixels / total_pixels

        if relative_area > 0.15:
            risk = "HIGH RISK"
        elif relative_area > 0.05:
            risk = "MODERATE RISK"
        else:
            risk = "LOW RISK"

        mask_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))

        overlay = img_np.copy()
        overlay[mask_resized == 1] = [255,0,0]

        blended = cv2.addWeighted(img_np,0.7,overlay,0.3,0)

        contours,_ = cv2.findContours(mask_resized.astype(np.uint8),
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended,contours,-1,(0,255,0),2)

    with col1:
        st.image(image, caption="Original Image")

    with col2:
        st.image(blended, caption="AI Output")

    st.metric("Risk", risk)
    st.metric("Area %", f"{relative_area*100:.2f}")

    st.subheader("Mask")
    st.image(mask*255)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("🚀 Final Year Project")
