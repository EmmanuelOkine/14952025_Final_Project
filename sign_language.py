import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Load the trained model and label encoder
model = load_model('sign.h5')  # Update with your model file path
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder.npy')  # Update with your label encoder file path

# Function to extract keypoints
def extract_keypoints(results):
    def flatten_landmarks(landmarks, num_landmarks, num_coords):
        if landmarks:
            return np.array([[res.x, res.y, res.z] for res in landmarks.landmark]).flatten()[:num_landmarks * num_coords]
        else:
            return np.zeros(num_landmarks * num_coords)

    # Extract keypoints for each type of landmark
    pose = flatten_landmarks(results.pose_landmarks, 33, 4)
    face = flatten_landmarks(results.face_landmarks, 468, 3)
    lh = flatten_landmarks(results.left_hand_landmarks, 21, 3)
    rh = flatten_landmarks(results.right_hand_landmarks, 21, 3)

    # Concatenate and pad the keypoints to match the required shape (30 sequences, 1629 features)
    keypoints = np.concatenate([pose, face, lh, rh])
    keypoints = np.pad(keypoints, (0, 30 * 1629 - len(keypoints)), mode='constant')
    keypoints = keypoints.reshape((30, 1629))  # Reshape to match the required input shape

    return keypoints
# Function to make predictions
def make_prediction(keypoints):
    prediction = model.predict(np.expand_dims(keypoints, axis=0))
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return predicted_label

# Streamlit app
def main():
    st.title('Sign Language Detection')

    # Initialize the video capture
    cap = cv2.VideoCapture(0)

    # Set Mediapipe Holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read video feed
            ret, frame = cap.read()
            if not ret or frame is None:
                continue  # Skip this iteration if the frame is empty or invalid

            # Convert frame to RGB for Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detections
            results = holistic.process(image)
            
            # Extract keypoints
            keypoints = extract_keypoints(results)
            
            # Make prediction
            predicted_label = make_prediction(keypoints)
            
            # Draw landmarks and predicted label on the frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                cv2.putText(frame, f'Prediction: {predicted_label}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the processed frame in Streamlit
            st.image(frame, channels="BGR", use_column_width=True)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the capture and destroy OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()