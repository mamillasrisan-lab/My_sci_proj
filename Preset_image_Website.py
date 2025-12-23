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
# SAFE WRAPPER
# -------------------------------
st.title("Image Captioner")
def safe(fn):
    try:
        return fn()
    except Exception:
        return None

# -------------------------------
# LOAD MODEL
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
# PRESET IMAGES
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
st.session_state.setdefault("image", None)
st.session_state.setdefault("caption", None)
st.session_state.setdefault("source", None)
st.session_state.setdefault("processed", [])

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

def set_image(img, source):
    st.session_state.image = img
    st.session_state.source = source
    st.session_state.caption = None

# -------------------------------
# UI
# -------------------------------
tab1, tab2, tab3 = st.tabs(["Generate Caption", "Processed Images", "Helper"])

# =====================================================
# TAB 1
# =====================================================
with tab1:
    st.markdown("""Choose a source, and the image and the generate caption button will appear below the source.""")
    st.markdown("**Options**")
    st.markdown("1. Sample images")
    st.markdown(" 2. Upload an image from your device")
    st.markdown("3. Paste a secure Image URL into the text box")
    st.markdown("4. Allow Access to your camera and take a picture.")
    

    
    
    # ---------- PRESETS ----------
    st.subheader("Sample Images")
    cols = st.columns(len(PRESETS))

    for col, (name, url) in zip(cols, PRESETS.items()):
        with col:
            if st.button(name, key=f"preset_{name}"):
                img = safe(lambda: load_image_from_url(url))
                if img:
                    set_image(img, "preset")

    if st.session_state.source == "preset" and st.session_state.image:
        st.image(st.session_state.image, width=350)
        if st.button("Generate Caption", key="preset_generate"):
            st.session_state.caption = generate_caption(st.session_state.image)

        if st.session_state.caption:
            st.success(st.session_state.caption)

    st.divider()

    # ---------- UPLOAD ----------
    st.subheader("Upload Image")
    uploaded = st.file_uploader("Upload", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        set_image(img, "upload")

    if st.session_state.source == "upload" and st.session_state.image:
        st.image(st.session_state.image, width=350)
        if st.button("Generate Caption", key="upload_generate"):
            st.session_state.caption = generate_caption(st.session_state.image)

        if st.session_state.caption:
            st.success(st.session_state.caption)

    st.divider()

    # ---------- URL ----------
    st.subheader("Image URL")
    url = st.text_input("Paste raw image URL")

    if st.button("Load Image from URL"):
        img = safe(lambda: load_image_from_url(url))
        if img:
            set_image(img, "url")

    if st.session_state.source == "url" and st.session_state.image:
        st.image(st.session_state.image, width=350)
        if st.button("Generate Caption", key="url_generate"):
            st.session_state.caption = generate_caption(st.session_state.image)

        if st.session_state.caption:
            st.success(st.session_state.caption)

    st.divider()

    # ---------- CAMERA ----------
    st.subheader("Camera")
    camera_img = st.camera_input("Take a picture")

    if camera_img:
        img = Image.open(camera_img).convert("RGB")
        set_image(img, "camera")

    if st.session_state.source == "camera" and st.session_state.image:
        st.image(st.session_state.image, width=350)
        if st.button("Generate Caption", key="camera_generate"):
            st.session_state.caption = generate_caption(st.session_state.image)

        if st.session_state.caption:
            st.success(st.session_state.caption)

    # ---------- SAVE ----------
    if st.session_state.caption:
        st.session_state.processed.append({
            "image": st.session_state.image,
            "caption": st.session_state.caption
        })

# =====================================================
# TAB 2
# =====================================================
with tab2:
    if not st.session_state.processed:
        st.info("No processed images yet.")
    else:
        for item in st.session_state.processed:
            st.image(item["image"], width=300)
            st.markdown(f"**Caption:** {item['caption']}")
            st.divider()

# =====================================================
# TAB 3
# =====================================================
with tab3:
    st.markdown("""
### Fixed Issues

✔ No duplicate keys  
✔ Caption appears under the correct source  
✔ No UI duplication  
✔ Stable Streamlit behavior  
""")
