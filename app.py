import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
import tensorflow_hub as hub


def load_model():
    model = tf.keras.models.load_model('EfficientNetB0.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
    return model


def prep_image(filename, img_shape=224):
    """
  Reads an image from filename, turns it into a tensor and reshape it
  to (img_shape, img_shape, colour_channels)
  """

    # # Read in the image
    # img = tf.io.read_file(filename)
    # Decode the read file into a tensor
    img = tf.convert_to_tensor(filename, dtype=tf.float32)
    # img = tf.image.decode_image(filename)
    # Resize the image
    img = tf.image.resize(img, size=[img_shape, img_shape])
    # Rescale the image (get all values between o and 1)
    img = img / 255.
    return img


def prediction(model, img, class_names):
    """
  Imports an image located at filename, makes a prediction with model.
  """

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Add in logic for multi-class
    if len(pred[0]) > 1:
        pred_class = class_names[tf.argmax(pred[0])]
    else:
        pred_class = class_names[int(tf.round(pred[0]))]

    return pred_class


st.title('GEO-VISION')

uploaded_file = st.sidebar.file_uploader('Upload Image')

st.sidebar.divider()

pred_button = st.sidebar.button("Predict", type='primary')

class_names = ['Amphibolite', 'Andesite', 'Basalt', 'Breccia', 'Coal', 'Conglomerate', 'Gabbro', 'Gneiss', 'Limestone', 'Quartz_diorite', 'Quartzite', 'Sandstone', 'Shale', 'schist']

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(bytes_data))
    img_array = np.array(image)

    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image')

    # Preprocess image
    img_preprocessed = prep_image(img_array)

    # Preprocess and make predictions
    if pred_button:
        predicted_class = prediction(model=load_model(), img=img_preprocessed, class_names=class_names)

        # Display the prediction
        st.markdown(f"<p style='font-size:24px'><strong>Prediction:</strong> {predicted_class}</p>", unsafe_allow_html=True)
else:
    st.warning("Please upload an image using the sidebar.")
