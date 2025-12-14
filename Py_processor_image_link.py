import streamlit as st
from PIL import Image
import torch
from transformers import BlipForConditionalGeneration, AutoProcessor
import requests
from io import BytesIO

# -----------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="BLIP Image Captioning", page_icon="📷", layout="centered")
st.title("📷 BLIP-1 Image Captioning")

# -----------------------------
# TAB SETUP
# -----------------------------
tabs = st.tabs(["Generate Caption", "Processed Images", "Helper"])
generate_tab, processed_tab, helper_tab = tabs

# -----------------------------
# SESSION STATE STORAGE
# -----------------------------
if "processed_images" not in st.session_state:
    st.session_state.processed_images = []  # List of (image, caption)

# -----------------------------
# LOAD BLIP-1 MODEL (CACHE)
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_blip():
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model

processor, model = load_blip()

# -----------------------------
# GENERATE CAPTION TAB
# -----------------------------
with generate_tab:
    st.write("Upload an image, take a photo, or provide an image URL to generate a caption.")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    camera_image = st.camera_input("Or take a photo")
    image_url = st.text_input("Or enter an image URL")

    image = None

    # Load image from input
    if uploaded_file:
        image = Image.open(uploaded_file)
    elif camera_image:
        image = Image.open(camera_image)
    elif image_url:
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            st.warning(f"Could not load image from URL: {e}")

    # Generate caption button
    if image:
        st.image(image, caption="Selected Image", width="stretch")
        if st.button("Generate Caption"):
            try:
                inputs = processor(image, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(**inputs)
                    caption = processor.decode(out[0], skip_special_tokens=True)

                st.markdown(f"**Caption:** {caption}")

                # Save to session_state
                st.session_state.processed_images.append((image.copy(), caption))

            except Exception as e:
                st.warning("BLIP-1 captioning unavailable.")
                st.code(str(e))
    else:
        st.info("Please upload an image, take a photo, or enter an image URL.")

# -----------------------------
# PROCESSED IMAGES TAB
# -----------------------------
with processed_tab:
    st.write("Previously processed images and their captions:")

    if st.session_state.processed_images:
        for idx, (img, cap) in enumerate(st.session_state.processed_images):
            st.image(img, caption=f"Caption: {cap}", use_column_width=True)
    else:
        st.info("No images have been processed yet.")

# -----------------------------
# HELPER TAB
# -----------------------------
with helper_tab:
    st.write("💡 App Helper")
    if st.button("Explain App"):
        st.info("""
        **How to use this app:**
        1. Go to the 'Generate Caption' tab.
        2. You can either:
           - Upload an image,
           - Take a photo with your camera,
           - Or provide a direct image URL.
        3. Click 'Generate Caption' to create a description of your image using BLIP-1.
        4. Go to the 'Processed Images' tab to view all images you've captioned along with their captions.
        
        The app automatically handles device selection (GPU if available, CPU otherwise).
        """)
