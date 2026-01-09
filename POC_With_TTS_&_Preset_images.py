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
# LOAD MODEL
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
# BROWSER TTS (WORKING)
# ===============================
def speak(text):
    components.html(
        f"""
        <script>
        const u = new SpeechSynthesisUtterance({text!r});
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
        </script>
        """,
        height=0,
    )

# ===============================
# PRESET IMAGES
# ===============================
PRESETS = {
    "Flies": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/FF/fruit_flies_in_farms_135.jpg",
    "Vehicle": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/CAR/cars_1.jpg",
    "Exhibit": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/Exhibit/Historical_Exhibit_room_177.jpg",
    "Multiple Objects": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/HO/House_hold_objects_156.jpg",
    "Fire": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/WF/wilfires_with_cars_184.jpg",
}

# ===============================
# SESSION STATE (CRITICAL)
# ===============================
if "image" not in st.session_state:
    st.session_state.image = None

if "caption" not in st.session_state:
    st.session_state.caption = None

if "source" not in st.session_state:
    st.session_state.source = None

if "preset_name" not in st.session_state:
    st.session_state.preset_name = None

if "processed" not in st.session_state:
    st.session_state.processed = []

# ===============================
# HELPERS
# ===============================
def load_image_from_url(url):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

def generate_caption(img):
    inputs = processor(img, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40)
    return processor.decode(out[0], skip_special_tokens=True)

def reset_except(source):
    if st.session_state.source != source:
        st.session_state.image = None
        st.session_state.caption = None
        st.session_state.source = source
        st.session_state.preset_name = None

# ===============================
# TABS
# ===============================
tab1, tab2, tab3 = st.tabs(
    ["Generate Caption", "Processed Images", "Instructions"]
)

# ======================================================
# TAB 1 — GENERATE
# ======================================================
with tab1:
    st.markdown("""
Once you choose a source and image, the **Generate Caption** button will appear
below that source.
""")

    # ---------- PRESETS ----------
    st.subheader("Sample Images")
    reset_except("preset")

    cols = st.columns(len(PRESETS))
    for col, (name, url) in zip(cols, PRESETS.items()):
        with col:
            if st.button(name, key=f"preset_{name}"):
                st.session_state.image = load_image_from_url(url)
                st.session_state.preset_name = name
                st.session_state.caption = None

    if st.session_state.source == "preset" and st.session_state.image:
        st.image(st.session_state.image, width=300)
        if st.button("Generate Caption", key="gen_preset"):
            with st.spinner("Generating caption..."):
                st.session_state.caption = generate_caption(
                    st.session_state.image
                )
                st.session_state.processed.append({
                    "image": st.session_state.image,
                    "caption": st.session_state.caption
                })

    # ---------- UPLOAD ----------
    st.divider()
    st.subheader("Upload Image")
    reset_except("upload")

    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "png", "jpeg"]
    )
    if uploaded:
        st.session_state.image = Image.open(uploaded).convert("RGB")
        st.session_state.caption = None

    if st.session_state.source == "upload" and st.session_state.image:
        st.image(st.session_state.image, width=300)
        if st.button("Generate Caption", key="gen_upload"):
            with st.spinner("Generating caption..."):
                st.session_state.caption = generate_caption(
                    st.session_state.image
                )
                st.session_state.processed.append({
                    "image": st.session_state.image,
                    "caption": st.session_state.caption
                })

    # ---------- URL ----------
    st.divider()
    st.subheader("Image URL")
    reset_except("url")

    url = st.text_input("Paste image URL")
    if url:
        st.session_state.image = load_image_from_url(url)
        st.session_state.caption = None

    if st.session_state.source == "url" and st.session_state.image:
        st.image(st.session_state.image, width=300)
        if st.button("Generate Caption", key="gen_url"):
            with st.spinner("Generating caption..."):
                st.session_state.caption = generate_caption(
                    st.session_state.image
                )
                st.session_state.processed.append({
                    "image": st.session_state.image,
                    "caption": st.session_state.caption
                })

    # ---------- CAMERA ----------
    st.divider()
    st.subheader("Camera")

    use_camera = st.checkbox("Use Camera", key="camera_checkbox")

    if use_camera:
        reset_except("camera")
        camera_img = st.camera_input("Take a picture")
        if camera_img:
            st.session_state.image = Image.open(camera_img).convert("RGB")
            st.session_state.caption = None

    if st.session_state.source == "camera" and st.session_state.image:
        st.image(st.session_state.image, width=300)
        if st.button("Generate Caption", key="gen_camera"):
            with st.spinner("Generating caption..."):
                st.session_state.caption = generate_caption(
                    st.session_state.image
                )
                st.session_state.processed.append({
                    "image": st.session_state.image,
                    "caption": st.session_state.caption
                })

    # ---------- RESULT ----------
    if st.session_state.caption:
        st.success(st.session_state.caption)
        if st.button("🔊 Read Caption Aloud"):
            speak(st.session_state.caption)

# ======================================================
# TAB 2 — PROCESSED
# ======================================================
with tab2:
    st.subheader("Processed Images")
    if not st.session_state.processed:
        st.info("No processed images yet.")
    else:
        for item in st.session_state.processed[::-1]:
            st.image(item["image"], width=200)
            st.markdown(f"**Caption:** {item['caption']}")
            st.divider()

# ======================================================
# TAB 3 — INSTRUCTIONS
# ======================================================
with tab3:
    st.markdown("""
### How it works

• Choose a source  
• Select or capture an image  
• Click **Generate Caption**  
• Optional: read caption aloud  

Captions are saved automatically.
""")
