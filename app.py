import os
import sqlite3
from huggingface_hub import snapshot_download
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import pickle
import tensorflow as tf 
from PIL import Image
 





def get_base64_of_bin_file(bin_file):
     with open(bin_file, 'rb') as f:
        data = f.read()
     return base64.b64encode(data).decode()

image_file = "Crop.jpg"  # your local image file path
img_base64 = get_base64_of_bin_file(image_file)

page_bg_img = f"""
     <style>
     [data-testid="stAppViewContainer"] {{
     background-image: url("data:image/jpg;base64,{img_base64}");
     background-size: cover;
     background-position: center;
     background-repeat: no-repeat;
     background-attachment: fixed;
     }}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center; color:black; '> 🌿 Crop Disease Prediction </h1>",
    unsafe_allow_html=True)


st.markdown("<h3 style='color: black;'>🧪 Enter Test Inputs</h3>", unsafe_allow_html=True)

with open('crop_model.pkl', 'rb') as f:
    crop_model = pickle.load(f)

col1, col2 = st.columns(2)

with col1:
    input1 = st.number_input("Nitrogen N")
    input2 = st.number_input("Phosphorus P")
    input3 = st.number_input("Potassium K")
    input4 = st.number_input(" Temperature")

with col2:
    input5 = st.number_input(" Humidity")
    input6 = st.number_input(" Soil pH")
    input7 = st.number_input(" Rainfall")
st.write("")  
col1, col2 = st.columns([0.8,1])
with col2 :
 Predict = st.button("🔍 Predict")

if Predict:
    input_array = [[input1, input2, input3, input4, input5, input6, input7]]
    result = crop_model.predict(input_array)
    st.success(f"Prediction : {result} ")
st.markdown("------------------------------")


uploaded_image = st.file_uploader("Upload a crop image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image")
else: 
    st.write("Please upload an image of the crop.")

# Load model
@st.cache_resource
def load_model():
    model_dir = snapshot_download(
        repo_id="Tusharyadav/satellite-image-classifier",
        repo_type="model"
    )
    saved_model_path = os.path.join(model_dir, "satellite_cnn_savedmodel")
    model = tf.keras.models.load_model(saved_model_path)
    return model

# Class names
class_names = [
    "Apple Apple scab", "Apple Black rot", "Apple Cedar apple rust", "Apple healthy",
    "Blueberry healthy", "Cherry (including sour) Powdery mildew", "Cherry (including sour) healthy",
    "Corn (maize) Cercospora leaf spot Gray leaf spot", "Corn (maize) Common rust",
    "Corn (maize) Northern Leaf Blight", "Corn (maize) healthy", "Grape Black rot",
    "Grape Esca (Black Measles)", "Grape Leaf blight (Isariopsis Leaf Spot)", "Grape healthy",
    "Orange Haunglongbing (Citrus greening)", "Peach Bacterial spot", "Peach healthy",
    "Pepper, bell Bacterial spot", "Pepper, bell healthy", "Potato Early blight",
    "Potato Late blight", "Potato healthy", "Raspberry healthy", "Soybean healthy",
    "Squash Powdery mildew", "Strawberry Leaf scorch", "Strawberry healthy",
    "Tomato Bacterial spot", "Tomato Early blight", "Tomato Late blight",
    "Tomato Leaf Mold", "Tomato Septoria leaf spot", "Tomato Spider mites Two-spotted spider mite",
    "Tomato Target Spot", "Tomato Tomato Yellow Leaf Curl Virus",
    "Tomato Tomato mosaic virus", "Tomato healthy"
]


# Classification function
def classify(model, image: Image.Image):
    image = image.resize((64, 64))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = float(np.max(predictions))
    class_label = class_names[predicted_index]
    return class_label, confidence

# Run classification
if uploaded_image is not None:
    uploaded_image.seek(0)
    image = Image.open(uploaded_image)
    model = load_model()
    label, confidence = classify(model, image)
    st.success(f"Prediction: {label} ({confidence:.2%} confidence)")