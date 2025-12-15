import streamlit as st
from PIL import Image
import torch
from transformers import BlipForConditionalGeneration, AutoProcessor
import requests
from io import BytesIO
import base64
import warnings

# -----------------------------
# SUPPRESS WARNINGS
# -----------------------------
warnings.filterwarnings("ignore")

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
    st.session_state.use_camera = False  # Ask if user wants to use camera

# -----------------------------
# LOAD BLIP-1 MODEL (CACHE)
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_blip():
    try:
        processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        return processor, model
    except Exception:
        st.warning("Could not load BLIP model. Please check your internet connection or model availability.")
        return None, None

processor, model = load_blip()

# -----------------------------
# HELPER FUNCTION FOR FADE-IN
# -----------------------------
def fade_in_image_caption(img: Image.Image, caption: str):
    try:
        # Convert image to base64 for inline HTML
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        html_code = f"""
        <style>
        .fade-in {{
            animation: fadeIn 1s ease-in-out;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        </style>
        <div class="fade-in">
            <img src="data:image/png;base64,{img_str}" style="max-width:100%;"/>
            <p><b>Caption:</b> {caption}</p>
        </div>
        """
        st.markdown(html_code, unsafe_allow_html=True)
    except Exception:
        st.warning("Could not display image.")

# -----------------------------
# GENERATE CAPTION TAB
# -----------------------------
with generate_tab:
    st.write("Upload an image, take a photo, or provide an image URL to generate a caption.")

    # Ask user if they want to use the camera
    st.session_state.use_camera = st.checkbox("Use Camera?", value=False)

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    camera_image = None
    if st.session_state.use_camera:
        camera_image = st.camera_input("Or take a photo")

    image_url = st.text_input("Or enter an image URL", key="text_input")

    image = None

    # Load image from input
    try:
        if uploaded_file:
            image = Image.open(uploaded_file)
        elif camera_image:
            image = Image.open(camera_image)
        elif image_url:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
    except Exception:
        st.warning("Could not load the image. Please check the file or URL.")

    # Generate caption button
    if image:
        st.image(image, caption="Selected Image", use_column_width=True)
        if st.button("Generate Caption"):
            if processor and model:
                try:
                    with st.spinner("Generating caption... Please wait."):
                        inputs = processor(image, return_tensors="pt").to(device)
                        with torch.no_grad():
                            out = model.generate(**inputs)
                            caption = processor.decode(out[0], skip_special_tokens=True)

                        # Display with fade-in
                        fade_in_image_caption(image.copy(), caption)

                        # Save to session_state
                        st.session_state.processed_images.append((image.copy(), caption))

                        # Clear URL text input
                        st.session_state.text_input = ""

                except Exception:
                    st.warning("Captioning failed. Try a different image or check your connection.")
            else:
                st.warning("Model is not loaded. Cannot generate captions.")
    else:
        st.info("Please upload an image, take a photo, or enter an image URL.")

# -----------------------------
# PROCESSED IMAGES TAB
# -----------------------------
with processed_tab:
    st.write("Previously processed images and their captions:")

    if st.session_state.processed_images:
        for idx, (img, cap) in enumerate(st.session_state.processed_images):
            fade_in_image_caption(img, cap)
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
           - Take a photo with your camera (enable the checkbox first),
           - Or provide a direct image URL.
        3. Click 'Generate Caption' to create a description of your image using BLIP-1.
        4. The URL box will clear automatically after processing.
        5. Go to the 'Processed Images' tab to view all images you've captioned along with their captions.

        The app automatically detects if a GPU is available and uses it; otherwise, it runs on CPU.
        If you are using an image link, make sure it is secure (https), otherwise the image will not load.
        """)
