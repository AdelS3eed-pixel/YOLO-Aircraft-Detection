import streamlit as st
import requests
from PIL import Image
import io

st.title("✈️ Aircraft Type Detection (SAM + FastAPI)")

uploaded_file = st.file_uploader("Upload an aircraft image...", type=["jpg", "jpeg", "png"])

API_URL = "https://seminomadically-diachronic-belia.ngrok-free.dev/predict"

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Detect Aircraft Type"):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        files = {"file": img_byte_arr.getvalue()}
        
        with st.spinner('Waiting for API response...'):
            try:
                response = requests.post(API_URL, files=files)
                result = response.json()
                st.success(f"Result: {result['class']} (Confidence: {result['confidence']})")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")