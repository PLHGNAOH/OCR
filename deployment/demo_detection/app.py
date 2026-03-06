import json
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# =========================
# Config
# =========================
MODEL_PATH = "best.pt"

st.set_page_config(
    page_title="OCR Text Detection Demo",
    layout="wide"
)

st.title("Text Detection (YOLO)")
st.write("Upload an image and click Run to detect text regions.")

# =========================
# Load model (cache)
# =========================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# =========================
# UI Controls
# =========================
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

run_button = st.button("Run Detection")

# =========================
# Inference
# =========================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(img_np, use_column_width=True)

    if run_button:
        with st.spinner("Running inference..."):
            results = model(img_np, verbose=False)
            annotated_img = results[0].plot()
            predictions = json.loads(results[0].to_json())

        with col2:
            st.subheader("Detection Result")
            st.image(annotated_img, use_column_width=True)

            st.subheader("Raw Predictions (JSON)")
            st.json(predictions)