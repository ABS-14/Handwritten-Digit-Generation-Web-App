import streamlit as st
import tensorflow as tf
# type: ignore[import]
from tensorflow.keras.models import load_model
import numpy as np

# Set the page configuration
st.set_page_config(layout="wide")

# Use Streamlit's cache to load the model only once
@st.cache_resource
def load_generator_model():
    # Load the Keras model. The 'compile=False' is important for inference-only models.
    model = load_model('cgan_generator.h5', compile=False)
    return model

# Load the generator
generator = load_generator_model()
LATENT_DIM = 100

# --- User Interface ---
st.title("Handwritten Digit Image Generator (TensorFlow)")
st.write("Generate synthetic MNIST-like images using a trained Conditional GAN model.")

st.sidebar.header("Controls")
selected_digit = st.sidebar.selectbox(
    "Choose a digit to generate (0-9):",
    options=list(range(10))
)

generate_button = st.sidebar.button("Generate Images", type="primary")

st.header(f"Generated images of digit {selected_digit}")

if generate_button:
    num_images = 5

    # Prepare input for the model
    noise = tf.random.normal([num_images, LATENT_DIM])
    labels = np.array([selected_digit] * num_images)

    # Generate images using the loaded TensorFlow model
    generated_images = generator.predict([noise, labels])
    
    # De-normalize images from [-1, 1] to [0, 1] for display
    generated_images = 0.5 * generated_images + 0.5

    # Display images in columns
    cols = st.columns(num_images)
    for i, col in enumerate(cols):
        with col:
            st.image(
                generated_images[i],
                caption=f"Sample {i+1}",
                use_container_width=True
            )
else:
    st.info("Choose a digit from the sidebar and click 'Generate Images' to start.")