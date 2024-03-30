# Copyright 2021 Alvaro Bartolome, alvarobartt @ GitHub
# See LICENSE for details.

import tensorflow as tf

from constants import MAPPING

def image2tensor(image_as_bytes):
    """
    Receives a image as bytes as input, that will be loaded,
    preprocessed and turned into a Tensor so as to include it
    in the TF-Serving request data.
    """

    # Apply the same preprocessing as during training (resize and rescale)
    image = tf.io.decode_image(image_as_bytes, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image/255.

    # Convert the Tensor to a batch of Tensors and then to a list
    image = tf.expand_dims(image, 0)
    image = image.numpy().tolist()
    return image


def prediction2label(prediction):
    """
    Receives the prediction Tensor and retrieves the index with the highest
    probability, so as to then map the index value in order to get the
    predicted label. 
    """

    # Retrieve the highest probablity index of the Tensor (actual prediction)
    prediction = tf.argmax(prediction)
    return MAPPING[prediction.numpy()]

# TF-Serving URLs for both gRPC and REST APIs
GRPC_URL = "tfserving:8500"
REST_URL = "http://tfserving:8501/v1/models/simpsonsnet:predict"

# Mapping of ids to labels (The Simpsons characters)
MAPPING = {
    0: "abraham_grampa_simpson", 1: "apu_nahasapeemapetilon", 2: "barney_gumble", 3: "bart_simpson",
    4: "carl_carlson", 5: "charles_montgomery_burns", 6: "chief_wiggum", 7: "comic_book_guy",
    8: "disco_stu", 9: "edna_krabappel", 10: "groundskeeper_willie", 11: "homer_simpson",
    12: "kent_brockman", 13: "krusty_the_clown", 14: "lenny_leonard", 15: "lisa_simpson",
    16: "maggie_simpson", 17: "marge_simpson", 18: "martin_prince", 19: "mayor_quimby",
    20: "milhouse_van_houten", 21: "moe_szyslak", 22: "ned_flanders", 23: "nelson_muntz",
    24: "patty_bouvier", 25: "principal_skinner", 26: "professor_john_frink", 27: "ralph_wiggum",
    28: "selma_bouvier", 29: "sideshow_bob", 30: "snake_jailbird", 31: "waylon_smithers"
}

import streamlit as st
import requests

# General information about the UI
st.title("TensorFlow Serving + Streamlit! ✨🖼️")
st.header("UI to use a TensorFlow image classification model of The Simpsons characters (named SimpsonsNet) served with TensorFlow Serving.")

# Show which are the classes that the SimpsonsNet model can predict
if st.checkbox("Show classes"):
    st.write("The SimpsonsNet can predict the following characters:")
    st.write(MAPPING)

# Create a FileUploader so that the user can upload an image to the UI
uploaded_file = st.file_uploader(label="Upload an image of any of the available The Simpsons characters (please see Classes).",
                                 type=["png", "jpeg", "jpg"])

# Display the predict button just when an image is being uploaded
if not uploaded_file:
    st.warning("Please upload an image before proceeding!")
    st.stop()
else:
    image_as_bytes = uploaded_file.read()
    st.image(image_as_bytes, use_column_width=True)
    pred_button = st.button("Predict")

if pred_button:
    # Converts the input image into a Tensor
    image_tensor = image2tensor(image_as_bytes=image_as_bytes)

    # Prepare the data that is going to be sent in the POST request
    json_data = {
        "instances": image_tensor
    }

    # Send the request to the Prediction API
    response = requests.post(REST_URL, json=json_data)

    # Retrieve the highest probablity index of the Tensor (actual prediction)
    prediction = response.json()['predictions'][0]
    label = prediction2label(prediction=prediction)

    # Write the predicted label for the input image
    st.write(f"Predicted The Simpsons character is: {label}")
