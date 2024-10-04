from sklearnex import patch_sklearn
import streamlit as st
import cv2
import numpy as np
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.applications.vgg16 import VGG16, preprocess_input
from keras._tf_keras.keras.preprocessing.image import img_to_array, load_img
import os
import face_recognition
import joblib
from PIL import Image
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import tempfile

from app_files.real_time_harrasment import predict_and_display_camera

# CSS for styling buttons and sidebar
st.markdown(
    """
        <style>
            /* General styles for buttons */
            .stButton button {
                display: inline-block; /* Allow margin and padding */
                padding: 12px 24px; /* Add padding for buttons */
                background-color: #f0f2f5; /* Green background */
                color: #112A46; /* White text */
                text-align: center; /* Center the text */
                text-decoration: none; /* Remove underline */
                border: none; /* Remove border */
                border-radius: 8px; /* Rounded corners */
                font-size: 16px; /* Font size */
                font-weight: bold; /* Bold text */
                transition: background-color 0.3s, transform 0.2s; /* Smooth transition */
                cursor: pointer; /* Pointer cursor on hover */
                margin: 5px 0; /* Space between buttons */
            }

            /* Hover effect */
            .stButton button:hover {
                background-color: #f0f2f5; /* Darker green on hover */
                color: #000000
                transform: translateY(-2px); /* Lift effect */
            }

            /* Active state */
            .stButton button:active {
                background-color: #f0f2f5; /* Even darker green when active */
                transform: translateY(0); /* Reset lift effect */
            }

            /* Sidebar styling */
            .sidebar .sidebar-content {
                background-color: #f0f2f5; /* Light gray background */
                padding: 20px; /* Padding around the sidebar */
                border-radius: 10px; /* Rounded corners for sidebar */
            }

            /* Title style in sidebar */
            .sidebar .sidebar-title {
                font-size: 24px; /* Title font size */
                font-weight: bold; /* Bold title */
                color: #333; /* Darker text for title */
                margin-bottom: 20px; /* Space below title */
            }
        </style>
    """, 
    unsafe_allow_html=True
)

def home():
    pass

def about():
    pass

def main():
    # Get the current query parameters to control page navigation
    query_params = st.experimental_get_query_params()
    page = query_params.get('page', ['home'])[0]  # Default to 'home'

    # Sidebar with navigation buttons
    st.sidebar.title("Navigation")

    # Use buttons to navigate
    if st.sidebar.button("Home"):
        st.experimental_set_query_params(page='Home')
        st.rerun()  # Refresh the page
    if st.sidebar.button("Capture Faces for Detection"):
        st.experimental_set_query_params(page='Capture faces for detection')
        st.rerun()  # Refresh the page
    if st.sidebar.button("Train Face Detection Model"):
        st.experimental_set_query_params(page='Train Face detection model')
        st.rerun()  # Refresh the page
    if st.sidebar.button("Real-Time Harassment Detection"):
        st.experimental_set_query_params(page='Real Time Harassment Detection')
        st.rerun()  # Refresh the page
    if st.sidebar.button("Real-Time Face Prediction"):
        st.experimental_set_query_params(page='Real Time Face Prediction')
        st.rerun()  # Refresh the page
    if st.sidebar.button("Detect Harassment for Image"):
        st.experimental_set_query_params(page='Detect harrasment for image')
        st.rerun()  # Refresh the page
    if st.sidebar.button("Detect Harassment for Video"):
        st.experimental_set_query_params(page='Detect harrasment for video')
        st.rerun()  # Refresh the page
    if st.sidebar.button("Detect Video and Alert"):
        st.experimental_set_query_params(page='Detect Video and alert')
        st.rerun()  # Refresh the page
    if st.sidebar.button("About"):
        st.experimental_set_query_params(page='About')
        st.rerun()  # Refresh the page
    
    # Execute the functionality based on the URL page parameter
    if page == "Home":
        home()
    elif page == "About":
        about()
    elif page == "Real Time Harassment Detection":
        st.title("Live Harassment Detection")
        model_path = "weight.keras"
        if os.path.exists(model_path):
            try:
                model = load_model(model_path)
                base_model = VGG16(weights='imagenet', include_top=False)
                if st.button("Start Prediction", key="start_harasment_detect"):
                    predict_and_display_camera(model, base_model)
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
        else:
            st.error("Model file not found.")
    elif page == "Real Time Face Prediction":
        pass
    elif page == "Capture faces for detection":
        pass
    elif page == "Train Face detection model":
        pass
    elif page == "Detect harrasment for image":
        pass
    elif page == "Detect harrasment for video":
        pass
    elif page == "Detect Video and alert":
        pass


def init():
    device = "/device:CPU:0"
    patch_sklearn()

    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    os.environ['DNNL_ENGINE_LIMIT_CPU_CAPABILITIES'] = '0'
    os.environ['ONEDNN_VERBOSE'] = '0' #'0' As i dont want to see the logs, set to '1' to see model epoch verbose logs

    os.environ['ONEAPI_DEVICE_SELECTOR'] = 'opencl:*'
    os.environ['SYCL_ENABLE_DEFAULT_CONTEXTS'] = '1'
    os.environ['SYCL_ENABLE_FUSION_CACHING'] = '1'

    os.environ['ITEX_XPU_BACKEND'] = 'CPU'
    os.environ['ITEX_AUTO_MIXED_PRECISION'] = '1'

if __name__ == "__main__":
    init()
    main()

# streamlit run app.py --client.showErrorDetails=false