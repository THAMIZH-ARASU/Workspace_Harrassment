import cv2
import streamlit as st
import numpy as np
from keras._tf_keras.keras.preprocessing.image import img_to_array
from keras._tf_keras.keras.applications.vgg16 import preprocess_input

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