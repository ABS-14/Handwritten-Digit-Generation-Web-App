# Streamlit Web Application for Handwritten Digit Generation

import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# ---------------------------------
# 1. Model Architecture (Must match the training script)
# ---------------------------------
# Define the same Generator architecture used during training.

latent_dim = 100
n_classes = 10
img_size = 28
channels = 1
img_shape = (channels, img_size, img_size)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_emb = self.label_embedding(labels)
        gen_input = torch.cat((noise, label_emb), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img

# ---------------------------------
# 2. Load Trained Model
# ---------------------------------

# Use a cache to load the model only once
@st.cache_resource
def load_model():
    # Instantiate the generator
    generator = Generator()
    # Load the trained weights. Ensure the .pth file is in the same directory.
    # Use map_location to ensure it runs on CPU if no GPU is available.
    generator.load_state_dict(torch.load('cgan_generator.pth', map_location=torch.device('cpu')))
    # Set the model to evaluation mode
    generator.eval()
    return generator

generator = load_model()

# ---------------------------------
# 3. Web App User Interface
# ---------------------------------

st.set_page_config(layout="wide")

st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using a trained Conditional GAN model.")

# --- Sidebar for user input ---
st.sidebar.header("Controls")
selected_digit = st.sidebar.selectbox(
    "Choose a digit to generate (0-9):",
    options=list(range(10))
)

generate_button = st.sidebar.button("Generate Images", type="primary")

# --- Main panel for displaying images ---
st.header(f"Generated images of digit {selected_digit}")

if generate_button:
    # Generate 5 images
    num_images = 5

    # Prepare input for the model
    with torch.no_grad():
        # Create 5 random noise vectors
        z = torch.randn(num_images, latent_dim)
        # Create labels for the selected digit
        labels = torch.LongTensor([selected_digit] * num_images)

        # Generate images
        generated_imgs = generator(z, labels)

    # De-normalize images from [-1, 1] to [0, 1] for display
    generated_imgs = 0.5 * generated_imgs + 0.5

    # Display images in columns
    cols = st.columns(num_images)
    for i, col in enumerate(cols):
        with col:
            st.image(
                generated_imgs[i].squeeze().numpy(),
                caption=f"Sample {i+1}",
                width=150, # Set image width
                use_column_width='auto'
            )
else:
    st.info("Choose a digit from the sidebar and click 'Generate Images' to start.")