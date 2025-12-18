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
# SESSION STATE
# -----------------------------
if "processed_images" not in st.session_state:
    st.session_state.processed_images = []
if "text_input" not in st.session_state:
    st.session_state.text_input = ""
if "use_camera" not in st.session_state:
    st.session_state.use_camera = False

# -----------------------------
# PRESET IMAGES
# -----------------------------
preset_folder = r"C:\Users\Srithan\preset_images"
friendly_names = {
    "wilfires_with_cars_118.jpg": "Wildfires with Cars",
    "Historical_Exhibit_room_132.jpg": "Historical Exhibit 132",
    "Historical_Exhibit_room_177.jpg": "Historical Exhibit 177",
    "fruit_flies_in_farms_161.jpg": "Fruit Flies in Farms"
}

preset_images = {}
for fname, friendly in friendly_names.items():
    full_path = os.path.join(preset_folder, fname)
    if os.path.exists(full_path):
        preset_images[friendly] = full_path

if not preset_images:
    st.warning("No preset images found! Check your folder path and file names.")

# -----------------------------
# LOAD BLIP MODEL
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_blip():
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model

processor, model = load_blip()

# -----------------------------
# HELPER FUNCTION
# -----------------------------
def fade_in_image_caption(image, caption):
    st.image(image, caption=caption, width="stretch")

# -----------------------------
# GENERATE CAPTION TAB
# -----------------------------
with generate_tab:
    st.write("Select a preset, upload an image, use the camera, or enter an image URL to generate a caption.")

    # Camera checkbox
    use_camera = st.checkbox("Use Camera", value=st.session_state.use_camera)
    st.session_state.use_camera = use_camera

    # Preset dropdown
    preset_choice = st.selectbox("Preset Images", [""] + list(preset_images.keys()))
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    # Camera input
    camera_image = None
    if use_camera:
        camera_image = st.camera_input("Take a photo")

    # URL input
    image_url = st.text_input("Or enter an image URL", key="text_input")

    image = None

    # Priority: Preset > Upload > Camera > URL
    if preset_choice and preset_choice in preset_images:
        try:
            image = Image.open(preset_images[preset_choice])
        except Exception as e:
            st.warning(f"Could not load preset image: {e}")
    elif uploaded_file:
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

    # Display selected image and generate caption
    if image:
        st.image(image, caption="Selected Image", width="stretch")
        if st.button("Generate Caption"):
            try:
                with st.spinner("Generating caption..."):
                    inputs = processor(image, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = model.generate(**inputs)
                        caption = processor.decode(out[0], skip_special_tokens=True)

                    st.session_state.processed_images.append((image.copy(), caption))
                    st.session_state.text_input = ""
                    fade_in_image_caption(image.copy(), caption)

            except Exception as e:
                st.warning(f"Captioning failed: {e}")
    else:
        st.info("Please select a preset, upload an image, use the camera, or enter a valid image URL.")

# -----------------------------
# PROCESSED IMAGES TAB
# -----------------------------
with processed_tab:
    st.write("Previously processed images and captions:")
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
        2. Select a preset, upload an image, use the camera, or enter a URL.
        3. Click 'Generate Caption' to generate a description using BLIP-1.
        4. The URL box will clear automatically after processing.
        5. View processed images in the 'Processed Images' tab.
        """)
