import io
import streamlit as st
import numpy as np
import cv2


# Define the image enhancement function
def enhance_image(image, contrast, brightness, color_balance, blur_type, blur_kernel_size, enhance_grayscale,
                  cartoonize):
    # Apply blur filter
    if blur_type == "Averaging":
        image = cv2.blur(image, (blur_kernel_size, blur_kernel_size))
    elif blur_type == "Bilateral":
        image = cv2.bilateralFilter(image, blur_kernel_size, 75, 75)

    # Apply image enhancements
    if enhance_grayscale:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced_gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)
        enhanced_image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    else:
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        l_channel = cv2.multiply(l_channel, contrast)
        l_channel = cv2.add(l_channel, brightness)
        a_channel = cv2.add(a_channel, color_balance)
        b_channel = cv2.add(b_channel, color_balance)
        lab_image = cv2.merge((l_channel, a_channel, b_channel))
        enhanced_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    # Apply cartoonization if enabled
    if cartoonize:
        gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
        color = cv2.bilateralFilter(enhanced_image, 9, 250, 250)
        cartoonized_image = cv2.bitwise_and(color, color, mask=edges)
        return cartoonized_image

    return enhanced_image


bytes_data = None
orig_bytes_data = None
st.subheader("Image Enhancement App")
st.write("Made by Bletsas Apostolos & Georgiadis Panagiotis")
st.subheader("Input")
input_mode = st.radio("Input mode", ["Camera", "File upload"])

if input_mode == "Camera":
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
else:
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg'])
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()

if bytes_data is not None:
    orig_bytes_data = bytes_data
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_img = img.copy()

    if input_mode != "Camera":
        st.image(img_bgr)
    st.text(f"({img.shape[1]}x{img.shape[0]})")

    st.header("Image transform")

    with st.sidebar:
        st.subheader("Blur filter")
        blur_type = st.radio("Blur type", ["None", "Averaging", "Bilateral"])
        blur_kernel_size = 10
        if blur_type != "None":
            blur_kernel_size = st.slider("Kernel size", 1, 100, value=10)

        st.subheader("Image Enhancement")
        contrast = st.slider("Contrast", 0.1, 3.0, 1.0, step=0.1)
        brightness = st.slider("Brightness", -100, 100, 0)
        color_balance = st.slider("Color Balance", -100, 100, 0)

        enhance_grayscale = st.checkbox("Enhance Grayscale", value=False)

        cartoonize = st.checkbox("Cartoonize", value=False)

        img = enhance_image(img, contrast, brightness, color_balance, blur_type, blur_kernel_size, enhance_grayscale,
                            cartoonize)

        st.subheader("Encode")
        encode_type = st.radio("Encode type", ("PNG", "JPEG"))
        if encode_type == "PNG":
            encoded = cv2.imencode(".png", img)[1].tobytes()
        else:
            jpeg_quality = st.slider("JPEG quality", 0, 100, value=90)
            encoded = cv2.imencode(".jpg", img, (cv2.IMWRITE_JPEG_QUALITY, jpeg_quality))[1].tobytes()

    st.subheader("Output")
    encoded_bytes_data = io.BytesIO(encoded)
    st.image(encoded_bytes_data)

    st.download_button("Download Processed", encoded_bytes_data, file_name=f"processed.{encode_type.lower()}")

    if orig_bytes_data is not None:
        orig_img_bgr = cv2.imdecode(np.frombuffer(orig_bytes_data, np.uint8), cv2.IMREAD_COLOR)
        orig_encoded = cv2.imencode(".jpg", orig_img_bgr)[1].tobytes()
        orig_bytes_data = io.BytesIO(orig_encoded)
        st.download_button("Download Original", orig_bytes_data, file_name="original.jpg")
else:
    st.stop()
