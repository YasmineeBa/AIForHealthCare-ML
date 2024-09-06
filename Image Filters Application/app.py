#import libraries
import streamlit as st
import cv2
from PIL import Image
import numpy as np
from io import BytesIO

st.set_page_config(page_title="image filter", layout="wide")
st.title("Image Filter Application")
st.markdown("<h5>This web application allows you to upload a photo and apply various filters to it using image processing function from OpenCV.</h5>",unsafe_allow_html=True)

#function to apply filters
def apply_filters(img, filter_type, filterValue=None):
    if filter_type == "Gray":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    elif filter_type == "Median":
        img = cv2.medianBlur(img, ksize=filterValue)
        return img
    elif filter_type == "Gaussian Blur":
        img = cv2.GaussianBlur(img, (filter_value, filterValue), sigmaX=5)
        return img
    elif filter_type == "Canny":
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.Canny(img_gray, filterValue, filterValue * 2)
        return img
    
    else:
        return img

#function to download the filtered image result
def download_image(image, filter_name, original_name):

    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    st.download_button(
        label=f"Download {filter_name} Image",
        data=buffer.getvalue(),
        file_name=f"{filter_name}_{original_name}",
        mime="image/jpeg"
    )


# step 1 : Uploading the image
uploaded_file = st.file_uploader("Upload your image ...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    # In Sidebar => step2: select Filter type
    st.sidebar.title('Filter options')
    filter_type = st.sidebar.radio("Select a filter type : ", ["Original", "Gray", "Median", "Gaussian Blur", "Canny"]) #choose a filter type

    # Step 3: Adjust filter value if applicable
    filter_value = None
    if filter_type == "Gaussian Blur":
        filter_value = st.sidebar.slider("Adjusting Blur Intensity", 1, 21, 5, step=2)
    elif filter_type== "Canny":
        filter_value = st.sidebar.slider("Adjusting Canny Edge Threshold", 50, 150, 100)
    elif filter_type == "Median":
        filter_value = st.sidebar.slider("Adjusting Median Value", 3, 21, 5, step=2)

    
    image = np.array(Image.open(uploaded_file)) #convert the img into np array
    input_col, output_col = st.columns(2)
    with input_col:
        st.image(image, caption='Original Image', use_column_width=True)
    with output_col:

        filtred_image = apply_filters(image, filter_type, filter_value) # Apply the filter
        #display the filtred image with this name format : "FileName_OriginalName"
        st.image(filtred_image, caption=f"{filter_type} Image", use_column_width=True)
        #step4: Doawnloading Filtered Images
    download_image(filtred_image, filter_type, uploaded_file.name)

else:
    st.info("Please upload an image to get started.")

st.markdown("<center><h6>made by yasmine baroud </h6></center>", unsafe_allow_html=True)
