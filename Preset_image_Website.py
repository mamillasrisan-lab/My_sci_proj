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
# SAFE EXECUTION
# -------------------------------
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
    processor = AutoProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
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
# SESSION STATE INIT
# -------------------------------
st.session_state.setdefault("processed", [])
st.session_state.setdefault("selected_image", None)
st.session_state.setdefault("url_input", "")
st.session_state.setdefault("last_captioned", None)

# -------------------------------
# IMAGE LOADER
# -------------------------------
def load_image_from_url(url):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

# -------------------------------
# CAPTION FUNCTION
# -------------------------------
def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=40)
    return processor.decode(output[0], skip_special_tokens=True)

# -------------------------------
# UI TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["Generate Caption", "Processed Images", "Helper"])

# =========================================================
# TAB 1 — GENERATE
# =========================================================
with tab1:
    st.markdown(
        "Select an image using **one** of the options below, then generate a caption."
    )

    # ---------- PRESETS ----------
    st.subheader("Preset Images")
    cols = st.columns(len(PRESETS))

    for col, (name, url) in zip(cols, PRESETS.items()):
        with col:
            if st.button(name, key=f"preset_{name}"):
                img = safe(lambda: load_image_from_url(url))
                if img:
                    st.session_state.selected_image = img

    st.divider()

    # ---------- UPLOAD ----------
    uploaded = st.file_uploader(
        "Upload Image", type=["jpg", "jpeg", "png"]
    )
    if uploaded:
        st.session_state.selected_image = Image.open(uploaded).convert("RGB")

    # ---------- URL ----------
    st.text_input(
        "Image URL",
        key="url_input",
        placeholder="https://raw.githubusercontent.com/..."
    )

    if st.button("Load Image from URL"):
        img = safe(lambda: load_image_from_url(st.session_state.url_input))
        if img:
            st.session_state.selected_image = img

    # ---------- CAMERA ----------
    use_camera = st.checkbox("Use Camera")
    if use_camera:
        camera_img = st.camera_input("Take a photo")
        if camera_img:
            st.session_state.selected_image = Image.open(camera_img).convert("RGB")

    st.divider()

    # ---------- DISPLAY + CAPTION ----------
    if st.session_state.selected_image:
        st.image(st.session_state.selected_image, width=400)

        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                caption = safe(
                    lambda: generate_caption(st.session_state.selected_image)
                )

            if caption:
                st.success("Caption generated")
                st.markdown(f"**Caption:** {caption}")

                if caption != st.session_state.last_captioned:
                    st.session_state.processed.append({
                        "image": st.session_state.selected_image,
                        "caption": caption
                    })
                    st.session_state.last_captioned = caption

# =========================================================
# TAB 2 — PROCESSED
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
• Click **Generate Caption**  
• Captions are saved in **Processed Images**

✔ Stable on Streamlit Cloud  
✔ No session state crashes  
✔ Handles GitHub RAW URLs correctly  
✔ Camera is optional and controlled
""")
