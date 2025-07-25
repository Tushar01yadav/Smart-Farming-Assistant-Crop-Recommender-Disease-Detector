import base64
import pickle

import numpy as np
import streamlit as st
from PIL import Image

st.title("🌾 Smart Farming Assistant:")

# Set background image using custom CSS from local file

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image = get_base64_of_bin_file("background.jpg")
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

def load_model():
    with open("crop_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()
col1, col2 = st.columns(2)


with col1:
    input1 = st.text_input("Nitrogen N")
    input3 = st.text_input("Phosphorus P")
    input5 = st.text_input("Potassium K")
    input7= st.text_input("Rainfall")

with col2:
    input2 = st.text_input("Temperature")
    input4 = st.text_input("Humidity")
    input6 = st.text_input("PH")

col_btn1, col_btn2, col_btn3 = st.columns([1.5, 1, 1])
with col_btn2:
    Predict = st.button("Predict")
if Predict:
     if (input7.strip() and input6.strip() and input5.strip() and input4.strip() and input3.strip() and input2.strip() and input1.strip()):
         features = np.array([[input1, input2, input3, input4, input5, input6, input7]])
         prediction = model.predict(features)
         st.success(f"Recommended Crop: {prediction[0]}")
     else :
         st.warning("Fields cannot be Empty please enter some value ")
# Image upload
uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True,height=150)

# Display inputs
