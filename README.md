# Project Overview

This project involves sign language recognition using MediaPipe's Holistic model and Keras, enabling gesture recognition through webcam input. It includes extracting keypoints using MediaPipe Holistic and training a model to recognize various sign language actions.

## Requirements
- Python 3.x
- OpenCV
- NumPy
- Mediapipe
- Keras
- Streamlit

## Code Structure

### Key Components
- **main.py:** Contains core code for capturing webcam input, extracting keypoints, and making predictions with the trained model.
- **train.py:** Code for training the sign language recognition model.
- **deployment.py:** Streamlit app using the trained model for real-time sign language recognition.

### Files
- **ASL.h5:** Trained sign language recognition model.
- **label_encoder.npy:** File for encoding and decoding categorical labels.

## Usage

### Training the Model
To train the model:
```bash
python train.py

bash
python train.py
This script uses keypoints extracted by MediaPipe Holistic from webcam data to train the model.

Real-time Sign Language Recognition
To run the real-time sign language recognition app:

## bash
python deployment.py
This Streamlit app captures webcam input, processes it using MediaPipe Holistic, and recognizes sign language gestures using the trained model.

## Libraries Used
The project primarily uses OpenCV, NumPy, Mediapipe, Keras, and Streamlit. Ensure these libraries are installed before running the scripts.

## Acknowledgments
MediaPipe by Google Research: https://mediapipe.dev/
Keras: https://keras.io/
Streamlit: https://streamlit.io/

Authors: Emmanuel Okine
Mariam Wahab

link to how our model works is seen below
https://youtu.be/gxnq_PBmP54  
