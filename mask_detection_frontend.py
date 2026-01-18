import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mask_detector_model.h5")

model = load_model()

st.set_page_config(page_title="Face Mask Detector", layout="centered")

st.title("😷 Face Mask Detection System")
st.write("Upload an image to check whether the person is wearing a mask.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image = image.resize((128, 128))
    image = np.array(image)
    image = image / 255.0
    image = np.reshape(image, (1, 128, 128, 3))

    # Prediction
if st.button('Predict'):
    prediction = model.predict(image)

    if prediction[0][0] > 0.5:
        st.success("✅ The person is WEARING a mask")
    else:
        st.error("❌ The person is NOT wearing a mask")

#python -m streamlit run mask_detection_frontend.py