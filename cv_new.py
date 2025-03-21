import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(layout="wide")  # Use full width
st.title("📷 Apply Filters to Your Image")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display

    filters = {
        "Original": img,
        "HSV": cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
        "Gray": cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        "LAB": cv2.cvtColor(img, cv2.COLOR_BGR2LAB),
        "LUV": cv2.cvtColor(img, cv2.COLOR_BGR2LUV),
        "Edges": cv2.Canny(img, 100, 200),
        "Blur": cv2.GaussianBlur(img, (15, 15), 0),
        "Median Blur": cv2.medianBlur(img, 5),
        "Bilateral Filter": cv2.bilateralFilter(img, 9, 75, 75),
        "Sharpen": cv2.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])),
        "Emboss": cv2.filter2D(img, -1, np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])),
        "Sepia": cv2.transform(img, np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])),
        "Invert": cv2.bitwise_not(img),
        "Pencil Sketch": cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)[0],
        "Cartoon": cv2.stylization(img, sigma_s=150, sigma_r=0.25),
        "Denoise": cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21),
        "Red Channel": cv2.merge([np.zeros_like(img[:,:,0]), np.zeros_like(img[:,:,1]), img[:,:,2]]),
        "Green Channel": cv2.merge([np.zeros_like(img[:,:,0]), img[:,:,1], np.zeros_like(img[:,:,2])]),
        "Blue Channel": cv2.merge([img[:,:,0], np.zeros_like(img[:,:,1]), np.zeros_like(img[:,:,2])]),
        "Threshold Binary": cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1],
        "Adaptive Threshold": cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
        "Morphology Gradient": cv2.morphologyEx(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.MORPH_GRADIENT, np.ones((5,5),np.uint8)),
        "Erosion": cv2.erode(img, np.ones((5,5),np.uint8), iterations=1),
        "Dilation": cv2.dilate(img, np.ones((5,5),np.uint8), iterations=1),
        "Sobel X": cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=5),
        "Sobel Y": cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1, ksize=5),
        "Laplacian": cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F),
        "Harris Corner": cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 2, 3, 0.04),
        "Equalized Histogram": cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    }

    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_column_width=True)
    
    selected_filter = st.sidebar.selectbox("Choose a filter", list(filters.keys()))
    
    with col2:
        st.image(filters[selected_filter], caption=f"{selected_filter} Image", use_column_width=True)
    
    # Convert to PIL Image and Prepare for Download
    if selected_filter != "Original":
        result = Image.fromarray(filters[selected_filter]) if selected_filter != "Grayscale" else Image.fromarray(filters[selected_filter]).convert("L")
        img_bytes = BytesIO()
        result.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()
        
        st.sidebar.download_button(
            label="⬇️ Download Filtered Image",
            data=img_bytes,
            file_name=f"filtered_{selected_filter}.png",
            mime="image/png"
        )
else:
    st.warning("⚠️ Please upload an image to apply filters.")
