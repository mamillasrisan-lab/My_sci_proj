import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import BlipForConditionalGeneration, AutoProcessor
import streamlit.components.v1 as components

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
# MODEL
# ===============================
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

# ===============================
# BROWSER TTS (STREAMLIT SAFE)
# ===============================
def speak(text):
    components.html(
        f"""
        <script>
        const msg = new SpeechSynthesisUtterance({text!r});
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(msg);
        </script>
        """,
        height=0,
    )

# ===============================
# PRESETS
# ===============================
PRESETS = {
    "Flies": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/FF/fruit_flies_in_farms_135.jpg",
    "Vehicle": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/CAR/cars_1.jpg",
    "Exhibit": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/Exhibit/Historical_Exhibit_room_177.jpg",
    "Multiple Objects": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/HO/House_hold_objects_156.jpg",
    "Fire": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/WF/wilfires_with_cars_184.jpg",
}

# ===============================
# SESSION STATE (MINIMAL & CORRECT)
# ===============================
if "active_source" not in st.session_state:
    st.session_state.active_source = None

if "image" not in st.session_state:
    st.session_state.image = None

if "caption" not in st.session_state:
    st.session_state.caption = None

# ===============================
# HELPERS
# ===============================
def clear_and_set(source):
    st.session_state.active_source = source
    st.session_state.image = None
    st.session_state.caption = None

def load_image(url):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

def generate_caption(img):
    inputs = processor(img, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40)
    return processor.decode(out[0], skip_special_tokens=True)

# ===============================
# TABS
# ===============================
tab1, tab2 = st.tabs(["Generate Caption", "Instructions"])

# ======================================================
# TAB 1
# ======================================================
with tab1:
    st.markdown("""
Once you choose a source and image, the **Generate Caption** button will appear
below that source.
""")

    # ---------- SAMPLE IMAGES ----------
    st.subheader("Sample Images")
    cols = st.columns(len(PRESETS))

    for col, (name, url) in zip(cols, PRESETS.items()):
        with col:
            if st.button(name):
                clear_and_set("preset")
                st.session_state.image = load_image(url)

    if st.session_state.active_source == "preset" and st.session_state.image:
        st.image(st.session_state.image, width=300)
        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                st.session_state.caption = generate_caption(
                    st.session_state.image
                )

    st.divider()

    # ---------- UPLOAD ----------
    st.subheader("Upload Image")
    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "png", "jpeg"]
    )
    if uploaded:
        clear_and_set("upload")
        st.session_state.image = Image.open(uploaded).convert("RGB")

    if st.session_state.active_source == "upload" and st.session_state.image:
        st.image(st.session_state.image, width=300)
        if st.button("Generate Caption", key="gen_upload"):
            with st.spinner("Generating caption..."):
                st.session_state.caption = generate_caption(
                    st.session_state.image
                )

    st.divider()

    # ---------- URL ----------
    st.subheader("Image URL")
    url = st.text_input("Paste image URL")
    if url:
        clear_and_set("url")
        st.session_state.image = load_image(url)

    if st.session_state.active_source == "url" and st.session_state.image:
        st.image(st.session_state.image, width=300)
        if st.button("Generate Caption", key="gen_url"):
            with st.spinner("Generating caption..."):
                st.session_state.caption = generate_caption(
                    st.session_state.image
                )

    st.divider()

    # ---------- CAMERA ----------
    st.subheader("Camera")
    use_camera = st.checkbox("Use Camera")

    if use_camera:
        clear_and_set("camera")
        cam = st.camera_input("Take a picture")
        if cam:
            st.session_state.image = Image.open(cam).convert("RGB")

    if st.session_state.active_source == "camera" and st.session_state.image:
        st.image(st.session_state.image, width=300)
        if st.button("Generate Caption", key="gen_cam"):
            with st.spinner("Generating caption..."):
                st.session_state.caption = generate_caption(
                    st.session_state.image
                )

    # ---------- RESULT ----------
    if st.session_state.caption:
        st.success(st.session_state.caption)
        if st.button("🔊 Read Caption Aloud"):
            speak(st.session_state.caption)

# ======================================================
# TAB 2
# ======================================================
with tab2:
    st.markdown("""
### Instructions

• Select **one source**  
• Choose or capture an image  
• Generate a caption  
• Optional text-to-speech  

This version is intentionally stable.
""")
