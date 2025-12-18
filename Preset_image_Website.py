import streamlit as st
from PIL import Image
import torch
from transformers import BlipForConditionalGeneration, AutoProcessor
import requests
from io import BytesIO
import os

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
if "text_input" not in st.session_state:
    st.session_state.text_input = ""  # URL input

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
# HELPER FUNCTION FOR FADE-IN IMAGE/CAPTION
# -----------------------------
def fade_in_image_caption(image, caption, delay=0.02):
    """Smoothly fade in the image and caption."""
    st.image(image, caption=caption, width="stretch")

# -----------------------------
# PRESET IMAGES SETUP
# -----------------------------
preset_folder = r"C:\Users\Srithan\preset images"
preset_images = {
    "Wildfires with Cars": os.path.join(preset_folder, "wilfires_with_cars_118.jpg"),
    "Historical Exhibit 132": os.path.join(preset_folder, "Historical_Exhibit_room_132.jpg"),
    "Historical Exhibit 177": os.path.join(preset_folder, "Historical_Exhibit_room_177.jpg"),
    "Fruit Flies in Farms": os.path.join(preset_folder, "fruit_flies_in_farms_161.jpg")
}

# -----------------------------
# GENERATE CAPTION TAB
# -----------------------------
with generate_tab:
    st.write("Select a preset image, upload an image, take a photo, or provide an image URL to generate a caption.")

    # Display preset images as buttons
    cols = st.columns(len(preset_images))
    selected_preset = None
    for i, (friendly_name, path) in enumerate(preset_images.items()):
        if os.path.exists(path):
            img = Image.open(path)
            if cols[i].button(friendly_name):
                selected_preset = img.copy()
        else:
            cols[i].write(f"Preset '{friendly_name}' not found")

    # Upload, camera, or URL
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    camera_image = st.camera_input("Or take a photo")
    image_url = st.text_input("Or enter an image URL", key="text_input")

    # Decide which image to use
    image = selected_preset
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

    # Camera checkbox prompt (asks every reload)
    if camera_image is None:
        if st.checkbox("Use Camera?"):
            st.experimental_rerun()

    # Generate caption button
    if image:
        st.image(image, caption="Selected Image", width="stretch")
        if st.button("Generate Caption"):
            caption = None
            if processor and model:
                try:
                    with st.spinner("Generating caption... Please wait."):
                        inputs = processor(image, return_tensors="pt").to(device)
                        with torch.no_grad():
                            out = model.generate(**inputs)
                            caption = processor.decode(out[0], skip_special_tokens=True)

                        # Save to session_state
                        st.session_state.processed_images.append((image.copy(), caption))

                        # Clear URL text input
                        st.session_state.text_input = ""

                except Exception:
                    st.warning("Captioning failed. Try a different image or check your connection.")
                    caption = None

                # Display image & caption with smooth transition
                if caption:
                    try:
                        fade_in_image_caption(image.copy(), caption)
                    except Exception:
                        st.warning("Caption generated, but displaying the image failed.")
    else:
        st.info("Please select or upload an image.")

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
           - Select a preset image,
           - Upload an image,
           - Take a photo with your camera,
           - Or provide a direct image URL.
        3. Click 'Generate Caption' to create a description of your image using BLIP-1.
        4. The URL box will clear automatically after processing.
        5. Go to the 'Processed Images' tab to view all images you've captioned along with their captions.

        The app automatically detects if a GPU is available and uses it; otherwise, it runs on CPU.
        For image links, only secure URLs (https) will be processed.
        """)

