import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
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

FROM_ADDRESS = ''
PASSWORD = ''
TO_ADDRESS = ''


def send_email(sender_email, sender_password, recipient_email, subject, message):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    body = message
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    text = msg.as_string()
    server.sendmail(sender_email, recipient_email, text)
    server.quit()
    
# CSS for styling buttons and sidebar
st.markdown("""
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
""", unsafe_allow_html=True)

# Define the different pages
# Load the KNeighborsClassifier model
model = joblib.load('face_recognition_model.pkl')

known_names = ["THAMIZHARASU S", "NATRAMIZH S"]

base_model = VGG16(weights='imagenet', include_top=False)

def predict_live_faces():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    stream = st.empty()  # Create a placeholder for the stream
    temp = 0
    while True:
        temp += 1
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            prediction = model.predict([face_encoding])[0]
            print(prediction)
            name = prediction if prediction in known_names else "Unknown"

            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        stream.image(frame, channels="BGR", caption="Face Detection")

        if st.button("Stop Face Detection", key=f"stop_face_detection{temp}"):
            break

    cap.release()
    stream.empty()

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

def train_face_model():
    # Paths
    CSV_FILE = 'DATA/Faces/image_data.csv'  # CSV file with image paths and names
    IMAGE_TEST_FOLDER = 'DATA/harrasment_detected'  # Folder with images to test

    # Step 1: Load known face encodings and names
    known_face_encodings = []
    known_face_names = []

    # Loop through each row in the CSV file to gather encodings
    df = pd.read_csv(CSV_FILE)
    for index, row in df.iterrows():
        image_path = row['Image Path']
        name = row['Name']
        
        # Load the image and extract the face encoding
        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            if face_encodings:
                known_face_encodings.append(face_encodings[0])  # Use the first face encoding
                known_face_names.append(name)
                if name not in known_names: 
                    known_names.append(name)
                st.write(f"Loaded encoding for {name}.")
            else:
                st.warning(f"No face found in the image: {image_path}.")
        except Exception as e:
            st.error(f"Error loading image {image_path}: {e}")

    if len(known_face_encodings) == 0 or len(known_face_names) == 0:
        st.error("No valid face encodings found. Please ensure you have images with faces.")
        return

    # Step 3: Train a KNeighborsClassifier model with probability estimates
    model = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree')  # Ensure to specify algorithm
    st.write("Training model...")
    model.fit(known_face_encodings, known_face_names)
    st.write("Model trained")

    # Step 4: Save the trained model
    joblib.dump(model, 'face_recognition_model.pkl')
    st.write("Model saved")


# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension as the model expects it
    return img

# Function to make predictions
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


def make_video_prediction(model, base_model, uploaded_file):
    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name  # Get the path of the temporary file

    cap = cv2.VideoCapture(temp_file_path)

    st.write("Live Stream:")
    stream = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
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

        # Display the processed frame with overlay using Streamlit's st.image
        stream.image(frame, channels="BGR", caption="Live Prediction")

    cap.release()

def predict_faces(frame, face_model_path, threshold=0.5):
    face_model = joblib.load(face_model_path)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Predict the name of the person based on the face encoding
        distances = face_model.kneighbors([face_encoding], n_neighbors=1, return_distance=True)
        distance = distances[0][0][0]  # Distance to the nearest neighbor
        
        # Define a distance threshold to determine known/unknown
        if distance < threshold:
            name = known_names[distances[1][0][0]]
        else:
            name = "Unknown"

        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        face_names.append(name)

    return face_names



def make_prediction_alert(image_model, base_model, face_model, uploaded_file):
    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name  # Get the path of the temporary file

    cap = cv2.VideoCapture(temp_file_path)

    st.write("Live Stream:")
    stream = st.empty()
    members = set()
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("End of video stream.")
            break

        resized_frame = cv2.resize(frame, (224, 224))
        preprocessed_frame = img_to_array(resized_frame)
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
        preprocessed_frame = preprocess_input(preprocessed_frame)

        features = base_model.predict(preprocessed_frame)
        features_flatten = features.reshape(1, -1)

        prediction = image_model.predict(features_flatten)
        class_label = np.argmax(prediction)
        class_prob = prediction[0][class_label]

        label = "Harassment" if class_label == 1 else "Non-Harassment"
        confidence = round(class_prob * 100, 2)
        prob_text = f"{label} ({confidence}%)"
        color = (0, 255, 0) if label == "Non-Harassment" else (0, 0, 255)
        cv2.putText(frame, prob_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=color, thickness=2)

        # Display the processed frame with overlay using Streamlit's st.image
        stream.image(frame, channels="BGR", caption="Live Prediction")

        if label == "Harassment":
            faces = predict_faces(frame, face_model)
            for face in faces:
                members.add(face)
                if face in known_names:
                    st.write(f"Alert: Known Person detected: {face}")
                else:
                    st.write("Alert: Unknown Person detected.")

    if faces:
        send_email(FROM_ADDRESS, PASSWORD, TO_ADDRESS, "Harassment Detected", f"Harassment detected (In frame : {list(members)}) at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")    
        st.write("Email sent to authorities.")
    cap.release()


# Streamlit App
def home():
    st.title("Home Page")
    st.write("Welcome to the Harassment Detection System.")
    st.write("This application is designed to detect harassment in real-time using advanced deep learning techniques like VGG16 and face recognition models.")
    st.write("You can explore the following features in this application:")
    st.write("- **Real-Time Harassment Detection**: Detect harassment incidents in real-time using live camera feed or uploaded videos.")
    st.write("- **Real-Time Face Detection**: Detect and recognize faces in real-time.")
    st.write("- **Image/Video Processing**: Upload images or videos for offline harassment detection.")
    st.write("- **Model Training**: Capture faces, train models, and improve the system over time.")
    st.markdown("### Get Started by navigating through the sidebar and selecting a feature to explore!")

def about():
    st.title("About Page")
    st.write("### About this Application")
    st.write("""
    This application leverages the power of deep learning and computer vision to detect incidents of sexual harassment in real-time.
    It integrates VGG16 for feature extraction and face recognition models to identify individuals involved.
    The system aims to assist in enhancing safety and providing an early alert system in sensitive environments.
    """)
    st.write("### Technologies Used:")
    st.markdown("""
    - **VGG16**: A Convolutional Neural Network for image classification and feature extraction.
    - **Face Recognition**: To recognize and track individuals.
    - **KNeighborsClassifier**: For training the face recognition model.
    - **Streamlit**: The web framework used to build this interactive interface.
    """)
    st.write("### Developer Info:")
    st.markdown("""
    Developed by a passionate team, aiming to provide innovative safety solutions through artificial intelligence and machine learning.
    For any inquiries or contributions, feel free to contact the development team.
    """)

# Main function for navigation
def main():
    # Get the current query parameters to control page navigation
    query_params = st.experimental_get_query_params()
    page = query_params.get('page', ['Home'])[0]  # Default to 'home' 


    # Sidebar with navigation buttons
    st.sidebar.title("Navigation")
    # Use buttons to navigate
    if st.sidebar.button("Home", use_container_width=True):
        st.experimental_set_query_params(page='Home')
        st.rerun()  # Refresh the page
    if st.sidebar.button("Capture Faces for Detection", use_container_width=True):
        st.experimental_set_query_params(page='Capture faces for detection')
        st.rerun()  # Refresh the page
    if st.sidebar.button("Train Face Detection Model", use_container_width=True):
        st.experimental_set_query_params(page='Train Face detection model')
        st.rerun()  # Refresh the page
    if st.sidebar.button("Real-Time Harassment Detection", use_container_width=True):
        st.experimental_set_query_params(page='Real Time Harassment Detection')
        st.rerun()  # Refresh the page
    if st.sidebar.button("Real-Time Face Prediction", use_container_width=True):
        st.experimental_set_query_params(page='Real Time Face Prediction')
        st.rerun()  # Refresh the page
    if st.sidebar.button("Detect Harassment for Image", use_container_width=True):
        st.experimental_set_query_params(page='Detect harrasment for image')
        st.rerun()  # Refresh the page
    if st.sidebar.button("Detect Harassment for Video", use_container_width=True):
        st.experimental_set_query_params(page='Detect harrasment for video')
        st.rerun()  # Refresh the page
    if st.sidebar.button("Detect Video and Alert", use_container_width=True):
        st.experimental_set_query_params(page='Detect Video and alert')
        st.rerun()  # Refresh the page
    if st.sidebar.button("About", use_container_width=True):
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
        st.title("Real-Time Face Detection")
        if st.button("Start Prediction", key="start_prediction"):
            predict_live_faces()
    elif page == "Capture faces for detection":
        st.title("Capture Faces for Detection")
        capture_faces()
    elif page == "Train Face detection model":
        st.title("Train Face Detection Model")
        if st.button("Train Face Detection model", key= "TrainFace"):
            train_face_model()
    elif page == "Detect harrasment for image":
        st.title("Detect Harrasment for Image")
        uploaded_file = st.file_uploader("Input an Image to detect any incident of Sexual harassment", type=["jpg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

            # Make prediction on the uploaded image
            predicted_class = make_image_prediction(uploaded_file)

            # Show the prediction result
            st.write("Prediction:", predicted_class)

    elif page == "Detect harrasment for video":
        st.title("Detect Harrasment for Video")
        
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
        os.environ['DNNL_ENGINE_LIMIT_CPU_CAPABILITIES'] = '0'
        os.environ['ITEX_XPU_BACKEND'] = 'GPU'
        os.environ['ITEX_AUTO_MIXED_PRECISION'] = '1'

        st.title("Live Harassment Detection")

        uploaded_file = st.file_uploader("Input Video to detect any incident of Sexual harassment", type=["mp4"])
        model_path = "harrasment_model.h5"

        model = load_model(model_path)

        base_model = VGG16(weights='imagenet', include_top=False)

        st.write("Press the button to start prediction:")
        if st.button("Start Prediction"):
            if (uploaded_file == None):
                st.write("Please upload a video file")
            else:
                make_video_prediction(model, base_model, uploaded_file)

    elif page == "Detect Video and alert":
        st.title("Detect Video and Alert")

        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
        os.environ['DNNL_ENGINE_LIMIT_CPU_CAPABILITIES'] = '0'
        os.environ['ITEX_XPU_BACKEND'] = 'GPU'
        os.environ['ITEX_AUTO_MIXED_PRECISION'] = '1'

        uploaded_file = st.file_uploader("Input Video to detect any incident of Sexual harassment", type=["mp4"])
        model_path = "harrasment_model.h5"
        image_model = load_model(model_path)        
        face_model = 'face_recognition_model.pkl'
        base_model = VGG16(weights='imagenet', include_top=False)

        st.write("Press the button to start prediction:")
        if st.button("Start Prediction"):
            if uploaded_file is None:
                st.write("Please upload a video file")
            else:
                make_prediction_alert(image_model, base_model, face_model, uploaded_file)



if __name__ == "__main__":
    main()
