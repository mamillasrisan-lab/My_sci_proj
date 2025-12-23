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

st.title("Image Identification and Captioning")

# ===============================
# SAFE ERROR HANDLER
# ===============================
def safe(fn):
    try:
        return fn()
    except Exception:
        return None

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
    "Flies": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/FF/fruit_flies_in_farms_161.jpg",
    "Vehicle": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/CAR/cars_1.jpg",
    "Exhibit": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/Exhibit/Historical_Exhibit_room_177.jpg",
    "Multiple Objects": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/HO/House_hold_objects_156.jpg",
    "Fire": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/WF/wilfires_with_cars_184.jpg",
    "Plane 1": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/JP%2BRP/planes_23.jpg",
    "Plane 2": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/JP%2BRP/planes_94.jpg",
}

# ===============================
# SESSION STATE
# ===============================
for key, default in {
    "preset_selected": {},  # stores selected image per preset
    "preset_caption": {},   # stores caption per preset
    "uploaded_image": None,
    "uploaded_caption": None,
    "url_input": "",
    "url_image": None,
    "url_caption": None,
    "camera_image": None,
    "camera_caption": None
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

# ===============================
# UI TABS
# ===============================
tab1, tab2, tab3 = st.tabs(["Generate Caption", "Processed Images", "Instructions"])

# ======================================================
# TAB 1 — GENERATE
# ======================================================
with tab1:
    # -------------------- INSTRUCTIONS --------------------
    st.markdown("""Once you choose a source and choose an image, the generate caption button will appear below the source, and when you click the button, the image will be identified and captioned""")
    st.markdown("**Options**")
    st.markdown("1. Sample images")
    st.markdown("2. Upload an image from your device")
    st.markdown("3. Paste a secure Image URL into the text box")
    st.markdown("4. Allow Access to your camera and take a picture.")

    st.subheader("Sample Images")
    cols = st.columns(len(PRESETS))
    for col, (name, url) in zip(cols, PRESETS.items()):
        with col:
            if st.button(name, key=f"preset_{name}"):
                img = safe(lambda: load_image_from_url(url))
                if img:
                    st.session_state.preset_selected[name] = img
                    st.session_state.preset_caption[name] = None
            # Display image + generate button only if selected
            if name in st.session_state.preset_selected:
                st.image(st.session_state.preset_selected[name], width=200)
                if st.button(f"Generate Caption", key=f"gen_{name}"):
                    with st.spinner("Generating caption..."):
                        cap = safe(lambda: generate_caption(st.session_state.preset_selected[name]))
                    st.session_state.preset_caption[name] = cap
                if st.session_state.preset_caption.get(name):
                    st.success(st.session_state.preset_caption[name])

    st.divider()

    # ---------- UPLOAD ----------
    st.subheader("Upload Image")
    uploaded = st.file_uploader("Upload", type=["jpg", "png", "jpeg"])
    if uploaded:
        st.session_state.uploaded_image = Image.open(uploaded).convert("RGB")
        st.session_state.uploaded_caption = None
    if st.session_state.uploaded_image:
        st.image(st.session_state.uploaded_image, width=200)
        if st.button("Generate Caption for Uploaded", key="gen_upload"):
            with st.spinner("Generating caption..."):
                st.session_state.uploaded_caption = safe(lambda: generate_caption(st.session_state.uploaded_image))
        if st.session_state.uploaded_caption:
            st.success(st.session_state.uploaded_caption)

    st.divider()

    # ---------- URL ----------
    st.subheader("Image URL")
    st.session_state.url_input = st.text_input(
        "Paste image URL",
        value=st.session_state.url_input,
        placeholder="https://raw.githubusercontent.com/..."
    )
    if st.button("Load Image from URL", key="url_load"):
        img = safe(lambda: load_image_from_url(st.session_state.url_input))
        if img:
            st.session_state.url_image = img
            st.session_state.url_caption = None
            st.session_state.url_input = ""
    if st.session_state.url_image:
        st.image(st.session_state.url_image, width=200)
        if st.button("Generate Caption for URL", key="gen_url"):
            with st.spinner("Generating caption..."):
                st.session_state.url_caption = safe(lambda: generate_caption(st.session_state.url_image))
        if st.session_state.url_caption:
            st.success(st.session_state.url_caption)

    st.divider()

    # ---------- CAMERA ----------
    st.subheader("Camera")
    use_camera = st.checkbox("Use Camera", key="camera_toggle")
    if use_camera:
        camera_img = st.camera_input("Take a picture", key="camera_input")
        if camera_img:
            st.session_state.camera_image = Image.open(camera_img).convert("RGB")
            st.session_state.camera_caption = None
    if st.session_state.camera_image:
        st.image(st.session_state.camera_image, width=200)
        if st.button("Generate Caption for Camera", key="gen_camera"):
            with st.spinner("Generating caption..."):
                st.session_state.camera_caption = safe(lambda: generate_caption(st.session_state.camera_image))
        if st.session_state.camera_caption:
            st.success(st.session_state.camera_caption)

# ======================================================
# TAB 2 — PROCESSED IMAGES
# ======================================================
with tab2:
    st.subheader("Processed Images")
    any_processed = False
    # Presets
    for name, cap in st.session_state.preset_caption.items():
        if cap:
            st.image(st.session_state.preset_selected[name], width=200)
            st.markdown(f"**Caption:** {cap}")
            st.divider()
            any_processed = True
    # Uploaded
    if st.session_state.uploaded_caption:
        st.image(st.session_state.uploaded_image, width=200)
        st.markdown(f"**Caption:** {st.session_state.uploaded_caption}")
        st.divider()
        any_processed = True
    # URL
    if st.session_state.url_caption:
        st.image(st.session_state.url_image, width=200)
        st.markdown(f"**Caption:** {st.session_state.url_caption}")
        st.divider()
        any_processed = True
    # Camera
    if st.session_state.camera_caption:
        st.image(st.session_state.camera_image, width=200)
        st.markdown(f"**Caption:** {st.session_state.camera_caption}")
        st.divider()
        any_processed = True
    if not any_processed:
        st.info("No processed images yet.")

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
