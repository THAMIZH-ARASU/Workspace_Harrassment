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

#helpers
base_model = VGG16(weights='imagenet', include_top=False)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension as the model expects it
    return img

def make_image_prediction(image_path):
    image_model = load_model('harrasment_model.h5')
    new_image = preprocess_image(image_path)

    # Extract features using the VGG16 base model
    new_image_features = base_model.predict(new_image)

    # Reshape the features
    new_image_features = new_image_features.reshape(1, 7 * 7 * 512)

    # Make predictions
    predictions = image_model.predict(new_image_features)

    # Since your model has 2 output neurons (softmax), you can use argmax to get the predicted class index
    predicted_class_index = np.argmax(predictions[0])

    # If your classes are labeled as 0 and 1, you can map the index back to class labels
    class_labels = {0: 'Healthy Workspace Environment :)', 1: '!! Sexual Harassment Detected !!'}
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label


#Pages
def capture_faces():
    # Define a folder to store the images
    IMAGE_SAVE_FOLDER = 'DATA/Faces'
    CSV_FILE = 'DATA/Faces/image_data.csv'

    # Create the folder if it doesn't exist
    if not os.path.exists(IMAGE_SAVE_FOLDER):
        os.makedirs(IMAGE_SAVE_FOLDER)

    # Check if the CSV file exists, if not, create it with headers
    if not os.path.isfile(CSV_FILE):
        df = pd.DataFrame(columns=['Name', 'Image Path'])
        df.to_csv(CSV_FILE, index=False)

    # Streamlit app title
    st.title("Image Capture and Store")

    # Input for person's name
    name = st.text_input("Enter your name:")

    # Streamlit webcam capture feature
    img_file_buffer = st.camera_input("Take a picture")

    # Get the count of existing images for the given name
    existing_images = len([f for f in os.listdir(IMAGE_SAVE_FOLDER) if f.startswith(name)])

    if img_file_buffer and name:
        # Convert to PIL format
        img = Image.open(img_file_buffer)
        
        # Generate a unique filename using the person's name and the existing image count
        image_filename = f"{name}_{existing_images + 1}.jpg"
        
        # Save the image in the specified folder
        image_path = os.path.join(IMAGE_SAVE_FOLDER, image_filename)
        img.save(image_path)
        
        # Load existing CSV data
        df = pd.read_csv(CSV_FILE)
        
        # Create a DataFrame for the new entry
        new_data = pd.DataFrame({'Name': [name], 'Image Path': [image_path]})
        
        # Concatenate the new data with the existing DataFrame
        df = pd.concat([df, new_data], ignore_index=True)
        
        # Save the updated DataFrame back to the CSV
        df.to_csv(CSV_FILE, index=False)
        
        # Display success message
        st.success(f"Image saved for {name} at {image_path}")

        # Optionally show the saved image
        st.image(img, caption=f"Captured Image of {name}")

def predict_and_display_camera(model, base_model):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    stream = st.empty()
    temp = 0
    while True:
        temp += 1
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame.")
            break

        resized_frame = cv2.resize(frame, (224, 224))
        preprocessed_frame = img_to_array(resized_frame)
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
        preprocessed_frame = preprocess_input(preprocessed_frame)

        features = base_model.predict(preprocessed_frame)
        features_flatten = features.reshape(1, -1)

        prediction = model.predict(features_flatten)[0]
        class_label = np.argmax(prediction)
        class_prob = prediction[class_label]

        label = "Harassment" if class_label == 1 else "Non-Harassment"
        confidence = round(class_prob * 100, 2)
        prob_text = f"{label} ({confidence}%)"
        color = (0, 255, 0) if label == "Non-Harassment" else (0, 0, 255)
        cv2.putText(frame, prob_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=color, thickness=2)

        stream.image(frame, channels="BGR", caption="Live Prediction")

        if st.button("Stop Detection", key=f"stop_harrasment_detect{temp}"):
            break

    cap.release()
    stream.empty()

def detect_harrasment_image():
    st.title("Detect Harrasment for Image")
    uploaded_file = st.file_uploader("Input an Image to detect any incident of Sexual harassment", type=["jpg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Make prediction on the uploaded image
        predicted_class = make_image_prediction(uploaded_file)

        # Show the prediction result
        st.write("Prediction:", predicted_class)


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
        capture_faces()
    elif page == "Train Face detection model":
        pass
    elif page == "Detect harrasment for image":
        detect_harrasment_image()
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