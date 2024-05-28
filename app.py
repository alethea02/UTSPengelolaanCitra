import streamlit as st
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

# Konversi RGB ke HSV
def convert_rgb_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Menghitung dan menampilkan histogram
def plot_histogram(image):
    plt.figure()
    for i, col in zip(range(3), ['b', 'g', 'r']):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.title('Histogram')
    st.pyplot(plt)

# Mengatur brightness dan contrast
def adjust_brightness_contrast(image, brightness=0, contrast=0):
    alpha = (contrast + 127) / 127  # alpha range 1.0-2.0
    beta = brightness - 127         # beta range -127 to 127
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

# Mendeteksi kontur
def find_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)
    return image_with_contours

# Streamlit UI
def main():
    st.title('Image Manipulation App')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, caption='Original Image', use_column_width=True)
        
        

        if st.button('Convert RGB to HSV'):
            hsv_image = convert_rgb_to_hsv(image)
            st.image(hsv_image, caption='HSV Image', use_column_width=True)

        if st.button('Show Histogram'):
            plot_histogram(image)

        brightness = st.slider("Brightness", -127, 127, 0)
        contrast = st.slider("Contrast", -127, 127, 0)

        if st.button('Adjust Brightness and Contrast'):
            bc_image = adjust_brightness_contrast(image, brightness, contrast)
            st.image(bc_image, caption='Brightness and Contrast Adjusted Image', use_column_width=True)

        if st.button('Find Contours'):
            contours_image = find_contours(image)
            st.image(contours_image, caption='Image with Contours', use_column_width=True)

if __name__ == '__main__':
    main()
