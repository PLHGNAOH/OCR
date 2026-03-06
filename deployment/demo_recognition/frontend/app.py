import json
import requests
import streamlit as st
from PIL import Image
from io import BytesIO

st.set_page_config(layout="wide")
st.title("OCR Text Detection & Recognition")

# ==============================
# Sidebar Config
# ==============================
API_URL = st.sidebar.text_input(
    "Backend API URL",
    "http://localhost:8000"
)

st.sidebar.markdown("---")
st.sidebar.info("Ensure backend is running before using this app.")

# ==============================
# Upload Section
# ==============================
uploaded = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(uploaded, use_container_width=True)

    if st.button("Run OCR"):

        with st.spinner("Processing..."):

            try:
                response = requests.post(
                    f"{API_URL}/ocr/upload",
                    files={
                        "file": (
                            uploaded.name,
                            uploaded.getvalue(),
                            uploaded.type
                        )
                    },
                    timeout=60
                )

            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")
                st.stop()

            # ==============================
            # Check Status Code
            # ==============================
            if response.status_code != 200:
                st.error("Backend Error:")
                st.text(response.text)
                st.stop()

            # ==============================
            # Validate Content Type
            # ==============================
            content_type = response.headers.get("content-type", "")
            if "image" not in content_type:
                st.error("Server did not return an image.")
                st.text(response.text)
                st.stop()

            # ==============================
            # Load Image Safely
            # ==============================
            try:
                processed_image = Image.open(BytesIO(response.content))
            except Exception as e:
                st.error(f"Image decoding error: {e}")
                st.stop()

            # ==============================
            # Parse Predictions
            # ==============================
            try:
                predictions = json.loads(
                    response.headers.get("X-Predictions", "[]")
                )
            except json.JSONDecodeError:
                predictions = []

        # ==============================
        # Display Results
        # ==============================
        with col2:
            st.subheader("Processed Image")
            st.image(processed_image, use_container_width=True)

        st.subheader("Detected Text Results")
        st.json(predictions)