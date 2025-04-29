import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image, method):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    if method == "Grayscale":
        return gray
    elif method == "Thresholding":
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        return thresh
    elif method == "Adaptive Thresholding":
        adapt_thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        return adapt_thresh
    elif method == "Noise Removal":
        blur = cv2.medianBlur(gray, 3)
        return blur
    return img

def ocr_core(image, custom_config=""):
    text = pytesseract.image_to_string(image, config=custom_config)
    return text

def main():
    st.set_page_config(page_title="OCR Sederhana", layout="wide")
    st.title("📄 OCR Sederhana dengan Tesseract")
    st.write("Unggah gambar untuk mengekstrak teks dengan hasil lebih baik.")
    
    uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "png", "jpeg", "bmp"])
    method = st.selectbox(
        "Pilih metode preprocessing", 
        ["None", "Grayscale", "Thresholding", "Adaptive Thresholding", "Noise Removal"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar Asli", use_container_width=True)
        
        processed_img = preprocess_image(image, method)
        st.image(processed_img, caption="Setelah Preprocessing", use_container_width=True, clamp=True)
        
        if st.button("🔎 Ekstrak Teks"):
            # Menggunakan config untuk tabel
            custom_config = r'--oem 3 --psm 6'  # OEM 3 = default, PSM 6 = assume block of text
            text = ocr_core(processed_img, custom_config=custom_config)
            
            st.subheader("📋 Hasil OCR:")
            st.text_area("Hasil OCR (Bisa di-copy)", text, height=400)

if __name__ == "__main__":
    main()
