
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Define the class labels
CLASSES = {0: "Mild Dementia", 1: "Moderate Dementia", 2: "Non Dementia", 3: "Very Mild Dementia"}

# Load your pre-trained deep learning model
model = load_model('VGG.hdf5')  


dim = (176, 208) 

# Function to preprocess the image
def preprocess_image(img_path):
    img = load_img(img_path, target_size=dim)  
    img_array = img_to_array(img) 
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

# Function to make predictions
def predict(img_path):
    processed_image = preprocess_image(img_path)
    predictions = model.predict(processed_image)
    print("Raw Predictions:", predictions)  
    predicted_class = np.argmax(predictions)
    pred_class_label = CLASSES[predicted_class]
    confidence = np.max(predictions) * 100
    return pred_class_label, confidence

# Streamlit app
st.title("Alzheimers Disease Stage Classification")

uploaded_file = st.file_uploader("Upload an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image.', use_container_width=True)
    st.write("")

    temp_file_path = "temp_image.jpg"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Classifying...")
    pred_class_label, confidence = predict(temp_file_path)

    st.write(f"**Predicted Stage:** {pred_class_label}")
    st.write(f"**Confidence:** {confidence:.2f}%")