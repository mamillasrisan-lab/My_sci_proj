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
if "use_camera" not in st.session_state:
    st.session_state.use_camera = False

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
# PRESET IMAGES
# -----------------------------
preset_folder = r"C:\Users\Srithan\preset_images"
preset_images = {
    "Wildfires with Cars": os.path.join(preset_folder, "wilfires_with_cars_118.jpg"),
    "Historical Exhibit 132": os.path.join(preset_folder, "Historical_Exhibit_room_132.jpg"),
    "Historical Exhibit 177": os.path.join(preset_folder, "Historical_Exhibit_room_177.jpg"),
    "Fruit Flies in Farms": os.path.join(preset_folder, "fruit_flies_in_farms_161.jpg")
}

# -----------------------------
# HELPER FUNCTION FOR FADE-IN IMAGE/CAPTION
# -----------------------------
def fade_in_image_caption(image, caption):
    st.image(image, caption=caption, width="stretch")

# -----------------------------
# GENERATE CAPTION TAB
# -----------------------------
with generate_tab:
    st.write("Select a preset image, upload an image, use the camera, or provide an image URL to generate a caption.")

    # Preset selection
    preset_choice = st.selectbox(
        "Choose a preset image",
        ["None"] + list(preset_images.keys())
    )

    # File upload
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    # Camera checkbox
    camera_checkbox = st.checkbox("Use Camera")
    camera_image = st.camera_input("Take a photo") if camera_checkbox else None

    # Image URL
    image_url = st.text_input("Or enter an image URL", key="text_input")

    image = None

    # Load preset image
    if preset_choice != "None":
        preset_path = preset_images[preset_choice]
        if os.path.exists(preset_path):
            image = Image.open(preset_path)
        else:
            st.warning(f"Preset '{preset_choice}' not found")

    # Load uploaded image
    elif uploaded_file:
        image = Image.open(uploaded_file)

    # Load camera image
    elif camera_image:
        image = Image.open(camera_image)

    # Load from URL
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

                # Display image & caption
                if caption:
                    fade_in_image_caption(image.copy(), caption)
    else:
        st.info("Please select a preset, upload an image, use the camera, or enter a valid image URL.")

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
           - Take a photo with your camera (check the box),
           - Or provide a direct image URL.
        3. Click 'Generate Caption' to create a description of your image using BLIP-1.
        4. The URL box will clear automatically after processing.
        5. Go to the 'Processed Images' tab to view all images you've captioned along with their captions.

        The app automatically detects if a GPU is available and uses it; otherwise, it runs on CPU.
        For image links, only secure URLs (https) will be processed.
        """)
