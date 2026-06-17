import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model # Safer import for tensorflow-cpu
from PIL import Image, ImageOps
import numpy as np

# Set up page configurations
st.set_page_config(page_title="Raghul Classifier", page_icon="📸", layout="centered")
st.title("📸 Image Classification App")
st.write("Upload an image to see if it's **Raghul** or a **Hand**.")

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Cache the model so it doesn't reload on every user interaction
@st.cache_resource

def init_model():
    # Use tf.keras to prevent internal backend errors
    model = tf.keras.models.load_model("keras_Model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

try:
    model, class_names = init_model()
except Exception as e:
    st.error(f"Error loading model or labels: {e}")
    st.stop()

# Streamlit Image Uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Analyzing image..."):
        # 2. Prepare the data array (Your exact logic)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # 3. Resize and crop from center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # 4. Turn into numpy array & Normalize
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # 5. Load image into array & Predict
        data[0] = normalized_image_array
        prediction = model.predict(data)
        
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

    # 6. Display formatting (Slicing 'class_name[2:]' to remove the prefix index numbers)
    st.success("### Analysis Complete!")
    st.metric(label="Predicted Class", value=f"{class_name[2:]}")
    st.metric(label="Confidence Score", value=f"{confidence_score * 100:.2f}%")
