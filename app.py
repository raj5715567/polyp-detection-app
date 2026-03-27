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

MODEL_PATH = "final.weights.h5"

# -------------------------
# DOWNLOAD MODEL
# -------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):

        url = "https://drive.google.com/uc?id=13t3y8TkT7H85p_2UpXikKz9ybxUWrzWq"

        with st.spinner("📥 Downloading model..."):
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

        size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        st.write(f"Downloaded size: {size:.2f} MB")

        if size < 300:
            st.error("❌ Model file corrupted")
            st.stop()

# -------------------------
# MODEL ARCHITECTURE
# -------------------------
def DeeplabV3(input_shape=(256,256,3)):

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
    b4 = tf.keras.layers.UpSampling2D(
        size=(layers[-1].shape[1], layers[-1].shape[2]),
        interpolation='bilinear')(b4)

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
# LOAD MODEL
# -------------------------
def load_model():

    download_model()

    model = DeeplabV3()

    # build model
    model(np.zeros((1,256,256,3)))

    # ✅ LOAD WEIGHTS (FINAL FIX)
    model.load_weights(MODEL_PATH)

    st.success("✅ Model loaded successfully")

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

uploaded = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

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
        st.image(image, caption="Original Image")

    with col2:
        st.image(blended, caption="AI Detection")

    st.subheader("Segmentation Mask")
    st.image(mask*255)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("🚀 Final Year Project • AI Medical Assistant")
