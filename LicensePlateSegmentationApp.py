import streamlit as st
import cv2
import numpy as np
from tensorflow import keras


def load_and_predict(model_path, threshold, uploaded_file, image_size, model_name):
    model = keras.models.load_model(model_path)

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        image_pred = cv2.resize(img, (image_size, image_size))
        image_pred = image_pred / 255.0

        image_pred = np.expand_dims(image_pred, axis=0)

        prediction = model.predict(image_pred)
        # st.image(prediction, caption="prediction", use_column_width=True)
        # st.write(f"max value of prediction: {np.max(prediction)}")
        # st.write(f"min value of prediction: {np.min(prediction)}")

        binary_prediction = (prediction > threshold).astype(np.uint8) * 255
        binary_prediction_squeezed = np.squeeze(binary_prediction, axis=0)

        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        resized_mask = cv2.resize(
            binary_prediction_squeezed,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        resized_mask_rgb = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2RGB)
        overlaid_image = cv2.addWeighted(image_rgb, 0.5, resized_mask_rgb, 0.5, 0)
        st.image(
            overlaid_image,
            caption=f"Segmentation Mask - Model: {model_name}",
            use_column_width=True,
        )


# Define your models, thresholds, and corresponding image sizes
models_info = [
    {
        "path": "license_plate_segmentation_model_unet.h5",
        "threshold": 0.748,
        "image_size": 128,
        "name": "UNetMobileV2",
    },
    {
        "path": "license_plate_segmentation_model_deeplab.h5",
        "threshold": 0.357,
        "image_size": 224,
        "name": "MobileNetV2Seg",
    },
]

# Create a list of model names for the selection dropdown
model_names = [model_info["name"] for model_info in models_info]

# Create a selection dropdown for models
selected_model_index = st.selectbox("Select Model", model_names)

# Find the selected model information based on the selected name
selected_model_info = next(
    (
        model_info
        for model_info in models_info
        if model_info["name"] == selected_model_index
    ),
    None,
)

# File uploader for user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Load and predict using the selected model and threshold
if selected_model_info:
    load_and_predict(
        selected_model_info["path"],
        selected_model_info["threshold"],
        uploaded_file,
        selected_model_info["image_size"],
        selected_model_info["name"],
    )
else:
    st.warning("Selected model information not found.")
