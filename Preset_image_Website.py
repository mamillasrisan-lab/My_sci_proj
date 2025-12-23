import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import BlipForConditionalGeneration, AutoProcessor

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="BLIP Image Captioning",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# SILENT ERROR HANDLING
# -------------------------------
def safe(fn):
    try:
        return fn()
    except Exception:
        return None

# -------------------------------
# LOAD MODEL (CACHED)
# -------------------------------
@st.cache_resource
def load_blip():
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model

processor, model = load_blip()

# -------------------------------
# PRESET IMAGES (GITHUB RAW)
# -------------------------------
PRESETS = {
    "Fruit Flies in Farms": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/FF/fruit_flies_in_farms_135.jpg",
    "Audi Car": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/CAR/cars_1.jpg",
    "Historical Exhibit Room": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/Exhibit/Historical_Exhibit_room_177.jpg",
    "Household Objects": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/HO/House_hold_objects_156.jpg",
    "Wildfires With Cars": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/WF/wilfires_with_cars_184.jpg",
}

# -------------------------------
# SESSION STATE
# -------------------------------
if "processed" not in st.session_state:
    st.session_state.processed = []

if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

if "url_input" not in st.session_state:
    st.session_state.url_input = ""

# -------------------------------
# IMAGE LOADER
# -------------------------------
def load_image_from_url(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

# -------------------------------
# CAPTION FUNCTION
# -------------------------------
def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40)
    return processor.decode(out[0], skip_special_tokens=True)

# -------------------------------
# UI TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["Generate Caption", "Processed Images", "Helper"])

# =========================================================
# TAB 1 — GENERATE
# =========================================================
with tab1:
    st.markdown(
        "Select an image or picture from one of the following sources and click generate caption. Options: 1. Select a sample image. 2. Upload a file. 3. Paste a secure image URL into the text box. 4. Take a picture with your camera. "
    )

    # ---------- PRESETS ----------
    st.subheader("Preset Images")
    cols = st.columns(len(PRESETS))
    for col, (name, url) in zip(cols, PRESETS.items()):
        with col:
            if st.button(name):
                img = safe(lambda: load_image_from_url(url))
                if img:
                    st.session_state.selected_image = img

    st.divider()

    # ---------- UPLOAD ----------
    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded:
        st.session_state.selected_image = Image.open(uploaded).convert("RGB")

    

    # ---------- URL ----------
    st.session_state.url_input = st.text_input(
        "Image URL",
        value=st.session_state.url_input,
        placeholder="https://raw.githubusercontent.com/..."
    )
# ---------- CAMERA ----------
    use_camera = st.checkbox("Use Camera")
    if use_camera:
        camera_img = st.camera_input("Take a photo")
        if camera_img:
            st.session_state.selected_image = Image.open(camera_img).convert("RGB")
    if st.button("Load Image from URL"):
        img = safe(lambda: load_image_from_url(st.session_state.url_input))
        if img:
            st.session_state.selected_image = img
            st.session_state.url_input = ""

    st.divider()

    # ---------- DISPLAY + CAPTION ----------
    if st.session_state.selected_image:
        st.image(st.session_state.selected_image, width=400)

        with st.spinner("Generating caption..."):
            caption = safe(lambda: generate_caption(st.session_state.selected_image))

        if caption:
            st.success("Caption generated")
            st.markdown(f"**Caption:** {caption}")

            st.session_state.processed.append({
                "image": st.session_state.selected_image,
                "caption": caption
            })

# =========================================================
# TAB 2 — PROCESSED IMAGES
# =========================================================
with tab2:
    if not st.session_state.processed:
        st.info("No processed images yet.")
    else:
        for item in st.session_state.processed:
            st.image(item["image"], width=300)
            st.markdown(f"**Caption:** {item['caption']}")
            st.divider()

# =========================================================
# TAB 3 — HELPER
# =========================================================
with tab3:
    st.markdown("""
### How this app works

• Choose a **preset image**  
• Or **upload**, **use the camera**, or **paste an image URL**  
• The app uses **BLIP-2 image captioning**  
• Captions are saved in the **Processed Images** tab  

This app is designed to:
- Never crash visually
- Hide all internal errors
- Always load valid images
- Work on Streamlit Cloud


""")
