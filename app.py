import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Define functions
def edge_mask(img, line_size, blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value
    )
    return edges

def color_quantization(img, k):
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, label, center = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

def cartoonize_image(image, line_size=7, blur_value=7, k=9):
    # Convert PIL image to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    edges = edge_mask(img, line_size, blur_value)
    img_quantized = color_quantization(img, k)
    blurred = cv2.bilateralFilter(img_quantized, d=7, sigmaColor=200, sigmaSpace=200)
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
    return cartoon

# Streamlit app
st.title("Image Cartoonization App")
st.write("Upload an image, and we'll turn it into a cartoon!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Cartoonize button
    if st.button("Cartoonize"):
        cartoon_image = cartoonize_image(image)
        
        # Display cartoonized image
        st.image(cartoon_image, caption="Cartoonized Image", use_column_width=True)
