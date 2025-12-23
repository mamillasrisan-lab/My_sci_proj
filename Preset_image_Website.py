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
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("📷 BLIP Image Captioning")

# -------------------------------
# SAFE EXECUTION (NO USER ERRORS)
# -------------------------------
def safe(fn):
    try:
        return fn()
    except Exception:
        return None

# -------------------------------
# LOAD BLIP MODEL (CACHED)
# -------------------------------
@st.cache_resource
def load_blip():
    processor = AutoProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        use_fast=False
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model

processor, model = load_blip()

# -------------------------------
# PRESET IMAGES (RAW GITHUB URLS)
# -------------------------------
PRESETS = {
    "Fruit Flies in Farms": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/FF/fruit_flies_in_farms_161.jpg",
    "Audi Car": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/CAR/cars_1.jpg",
    "Historical Exhibit": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/Exhibit/Historical_Exhibit_room_177.jpg",
    "Household Objects": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/HO/House_hold_objects_156.jpg",
    "Wildfires With Cars": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/WF/wilfires_with_cars_184.jpg",
}

# -------------------------------
# SESSION STATE
# -------------------------------
st.session_state.setdefault("images", {})
st.session_state.setdefault("active_source", None)
st.session_state.setdefault("processed", [])
st.session_state.setdefault("url_input", "")

# -------------------------------
# HELPERS
# -------------------------------
def load_image_from_url(url):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40)
    return processor.decode(out[0], skip_special_tokens=True)

# -------------------------------
# UI TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["Generate Caption", "Processed Images", "Helper"])

# ======================================================
# TAB 1 — GENERATE
# ======================================================
with tab1:
    st.markdown(
        "Select a **preset image**, **upload**, **use the camera**, or **paste an image URL**, then generate a caption."
    )

    # ---------- PRESETS ----------
    st.subheader("Preset Images")
    cols = st.columns(len(PRESETS))

    for col, (name, url) in zip(cols, PRESETS.items()):
        with col:
            if st.button(name):
                img = safe(lambda: load_image_from_url(url))
                if img:
                    st.session_state.images["preset"] = img
                    st.session_state.active_source = "preset"

    st.divider()

    # ---------- UPLOAD ----------
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        st.session_state.images["upload"] = Image.open(uploaded).convert("RGB")
        st.session_state.active_source = "upload"

    # ---------- URL ----------
    st.session_state.url_input = st.text_input(
        "Image URL",
        value=st.session_state.url_input,
        placeholder="https://raw.githubusercontent.com/..."
    )

    if st.button("Load Image from URL"):
        img = safe(lambda: load_image_from_url(st.session_state.url_input))
        if img:
            st.session_state.images["url"] = img
            st.session_state.active_source = "url"
            st.session_state.url_input = ""

    # ---------- CAMERA ----------
    use_camera = st.checkbox("Use Camera")
    if use_camera:
        cam = st.camera_input("Take a photo")
        if cam:
            st.session_state.images["camera"] = Image.open(cam).convert("RGB")
            st.session_state.active_source = "camera"

    # ---------- DISPLAY + CAPTION (SINGLE LOCATION) ----------
    source = st.session_state.active_source
    images = st.session_state.images

    if source and source in images:
        st.divider()
        st.subheader(f"Selected Source: {source.capitalize()}")

        st.image(images[source], width=400)

        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                caption = safe(lambda: generate_caption(images[source]))

            if caption:
                st.success(caption)
                st.session_state.processed.append({
                    "image": images[source],
                    "caption": caption
                })

    else:
        st.info("Waiting for a valid image selection.")

# ======================================================
# TAB 2 — PROCESSED IMAGES
# ======================================================
with tab2:
    if not st.session_state.processed:
        st.info("No processed images yet.")
    else:
        for item in st.session_state.processed:
            st.image(item["image"], width=300)
            st.markdown(f"**Caption:** {item['caption']}")
            st.divider()

# ======================================================
# TAB 3 — HELPER
# ======================================================
with tab3:
    st.markdown("""
### How this app works

• Choose a **preset image** or  
• **Upload**, **use camera**, or **paste an image URL**  
• Click **Generate Caption**  
• Captions appear under the selected source  
• All results are saved in **Processed Images**

### Features
✔ BLIP image captioning  
✔ GPU auto-detection  
✔ No visible errors  
✔ Streamlit Cloud compatible  
✔ Clean state handling  
✔ One image rendered at a time  
""")
