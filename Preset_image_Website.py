import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import BlipForConditionalGeneration, AutoProcessor

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Image Identification and Captioning",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# SAFE ERROR HANDLER
# ===============================
def safe(fn):
    try:
        return fn()
    except Exception:
        return None

st.title("Image Identification and Captioning")

# ===============================
# LOAD MODEL (CACHED)
# ===============================
@st.cache_resource
def load_blip():
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model

processor, model = load_blip()

# ===============================
# PRESET IMAGES (RAW GITHUB)
# ===============================
PRESETS = {
    "Flies":
        "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/FF/fruit_flies_in_farms_161.jpg",
    "Vehicle":
        "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/CAR/cars_1.jpg",
    "Exhibit":
        "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/Exhibit/Historical_Exhibit_room_177.jpg",
    "Multiple Objects":
        "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/HO/House_hold_objects_156.jpg",
    "Fire":
        "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/WF/wilfires_with_cars_184.jpg",
    "Plane 1":
        "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/JP%2BRP/planes_23.jpg",
    "Plane 2":
        "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/JP%2BRP/planes_94.jpg",
}

# ===============================
# SESSION STATE
# ===============================
for key, default in {
    "image": None,
    "caption": None,
    "source": None,
    "processed": [],
    "url_input": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ===============================
# IMAGE + CAPTION FUNCTIONS
# ===============================
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
    st.session_state.caption = None
    st.session_state.source = source

# ===============================
# UI TABS
# ===============================
tab1, tab2, tab3 = st.tabs(["Generate Caption", "Processed Images", "Instructions"])

# ======================================================
# TAB 1 — GENERATE
# ======================================================
with tab1:
    # -------------------- INSTRUCTIONS --------------------
    st.markdown("""
Once you choose a source and choose an image, the generate caption button will appear below the source, and when you click the button, the image will be identified and captioned
""")
    st.markdown("**Options**")
    st.markdown("1. Sample images")
    st.markdown("2. Upload an image from your device")
    st.markdown("3. Paste a secure Image URL into the text box")
    st.markdown("4. Allow Access to your camera and take a picture.")
    
    # ---------- PRESETS ----------
    st.subheader("Sample Images")
    cols = st.columns(len(PRESETS))
    for col, (name, url) in zip(cols, PRESETS.items()):
        with col:
            img = safe(lambda: load_image_from_url(url))
            if img:
                st.image(img, width=150)
                if st.button(f"Select {name}", key=f"preset_{name}"):
                    set_image(img, "preset")

    st.divider()

    # ---------- UPLOAD ----------
    st.subheader("Upload Image")
    uploaded = st.file_uploader("Upload", type=["jpg", "png", "jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        set_image(img, "upload")

    st.divider()

    # ---------- URL ----------
    st.subheader("Image URL")
    st.session_state.url_input = st.text_input(
        "Paste an image URL, then click the button labeled Load Image from URL ",
        value=st.session_state.url_input,
        placeholder="https://raw.githubusercontent.com/..."
    )

    if st.button("Load Image from URL", key="url_load"):
        img = safe(lambda: load_image_from_url(st.session_state.url_input))
        if img:
            set_image(img, "url")
            st.session_state.url_input = ""

    st.divider()

    # ---------- CAMERA ----------
    st.subheader("Camera")
    use_camera = st.checkbox("Use Camera", key="camera_toggle")
    if use_camera:
        camera_img = st.camera_input("Take a picture", key="camera_input")
        if camera_img:
            img = Image.open(camera_img).convert("RGB")
            set_image(img, "camera")

    st.divider()

    # ---------- DISPLAY + CAPTION ----------
    if st.session_state.image:
        st.image(st.session_state.image, width=400)

        if st.button("Generate Caption", key=f"gen_{st.session_state.source}"):
            with st.spinner("Generating caption..."):
                st.session_state.caption = safe(
                    lambda: generate_caption(st.session_state.image)
                )

            if st.session_state.caption:
                st.session_state.processed.append({
                    "image": st.session_state.image,
                    "caption": st.session_state.caption
                })

        if st.session_state.caption:
            st.success(st.session_state.caption)

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
# TAB 3 — INSTRUCTIONS
# ======================================================
with tab3:
    st.markdown("""
### How this app works

• Choose a **preset image**, **upload**, **camera**, or **URL**  
• Click **Generate Caption**  
• Captions are created using **BLIP Image Captioning**  
• Results are saved in **Processed Images**  

This app is optimized for education and research use.
""")
