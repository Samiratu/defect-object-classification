import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from skimage.metrics import structural_similarity as ssim
from sklearn import preprocessing
import xgboost as xgb
import joblib
import cv2
import glob
import os


# ... Your functions and model loading code ...
def sift_image_similarity(img1, img2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors for each image
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Initialize the brute-force matcher
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Calculate similarity score based on the number of good matches
    similarity_score = len(good_matches) / max(len(kp1), len(kp2))
    return similarity_score

# Load the saved XGBoost model
model_filename = "xgboost_model.joblib"
model = joblib.load(model_filename)

# Load model without classifier/fully connected layers
resnet_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Load label encoder
label_encoder = joblib.load("label_encoder.joblib")

def main():
    st.header("Casting Defect Detection for submersible pump impeller")
    st.subheader("Casting defect is an undesired irregularity in a metal casting process.")
    st.write("Casting is a manufacturing process involving pouring a liquid material into a mold with a desired shape, and then letting it solidify. Defects from the process include pinholes, burr, shrinkage defects, mould material defects, etc")
    # Load the reference image
    reference_image_path = "cast_ok_0_14.jpeg"
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    uploaded_image = st.file_uploader("Upload an image of top view of a submersible pump impeller:", type=["jpg", "png", "jpeg"])

     # Create two columns for displaying images side by side
    col1, col2 = st.columns(2)
    
    # Display the reference image in the first column
    with col1:
        st.write("Reference Image:")
        st.image(reference_image, use_column_width=True)

    

    if uploaded_image is not None:
        # Read the uploaded image as 3 channels (RGB)
        img1 = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        img2 = reference_image

        similarity_score = sift_image_similarity(img1, img2)

        if similarity_score >= 0.085:
            # Display the uploaded image in the second column
            with col2:
                st.write("Uploaded Image:")
                st.image(img1, use_column_width=True)

            # ... The rest of your code for image classification ...
            # Preprocess the user input image
            img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Preprocess the image for ResNet50 (resize and normalization)
            img_resized = cv2.resize(img, (128, 128))  # Resize to 128x128
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)  # Convert to BGR (OpenCV format)
            img_resized = preprocess_input(img_resized)  # Preprocess the image for ResNet50

            # Extract features using ResNet50
            input_img_feature = resnet_model.predict(np.expand_dims(img_resized, axis=0))
            input_img_features = input_img_feature.reshape(input_img_feature.shape[0], -1)


            # Make predictions with probabilities
            prediction_probs = model.predict_proba(input_img_features)[0]
            prediction_class = np.argmax(prediction_probs)
            prediction_label = label_encoder.inverse_transform([prediction_class])[0]

            st.success(f"The uploaded image is similar to the reference image! Similarity Score: {similarity_score:.2f}")

            # Display the prediction
            # st.write(f"Predicted Class: {prediction_class} - {prediction_label}")
            st.write("Probability Estimates:")
            for i, prob in enumerate(prediction_probs):
                class_name = label_encoder.inverse_transform([i])[0]
                st.write(f"Class {i} ({class_name}): {prob:.4f}")

            # Classification based on probability thresholds
            defective_threshold = 0.17
            ok_threshold = 0.8

            if prediction_probs[0] >= defective_threshold and prediction_probs[0] <= 0.6:
                st.write("The image is DEFECTIVE.")
            elif np.max(prediction_probs) >= ok_threshold:
                st.write("The image is OK.")

        else:
            with col2:
                st.write("Uploaded Image:")
            st.warning(f"The uploaded image has nothing to do with the task! Similarity Score: {similarity_score:.2f}")

if __name__ == "__main__":
    main()










