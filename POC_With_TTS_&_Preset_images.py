import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import BlipForConditionalGeneration, AutoProcessor
import edge_tts
import asyncio
import os
import hashlib
import time

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
# PRESET IMAGES
# ===============================
PRESETS = {
    "Flies": "https://raw.githubusercontent.com/mamillasrisan-lab/Images/refs/heads/main/FF/fruit_flies_in_farms_135.jpg",
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
if "processed" not in st.session_state:
    st.session_state.processed = []

if "url_input" not in st.session_state:
    st.session_state.url_input = ""

if "current" not in st.session_state:
    st.session_state.current = {
        "image": None,
        "caption": None,
        "source": None,
        "audio": None
    }

# ===============================
# IMAGE + CAPTION
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

def set_current(img, source):
    st.session_state.current = {
        "image": img,
        "caption": None,
        "source": source,
        "audio": None
    }

# ===============================
# TTS (EDGE-TTS)
# ===============================
AUDIO_DIR = "tts_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

def cleanup_audio(max_age=600):
    now = time.time()
    for f in os.listdir(AUDIO_DIR):
        path = os.path.join(AUDIO_DIR, f)
        if os.path.isfile(path) and now - os.path.getmtime(path) > max_age:
            os.remove(path)

cleanup_audio()

def caption_to_audio(text, voice, rate, volume):
    h = hashlib.md5(f"{text}{voice}{rate}{volume}".encode()).hexdigest()
    filename = f"{AUDIO_DIR}/{h}.mp3"

    if os.path.exists(filename):
        return filename

    async def _tts():
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            volume=volume
        )
        await communicate.save(filename)

    asyncio.run(_tts())
    return filename

# ===============================
# UI TABS
# ===============================
tab1, tab2, tab3 = st.tabs(["Generate Caption", "Processed Images", "Instructions"])

# ======================================================
# TAB 1 — GENERATE
# ======================================================
with tab1:
    st.markdown("""
Once you choose a source and choose an image, the generate caption button will appear below the source, and when you click the button, the image will be identified and captioned
""")
    st.markdown("**Options**")
    st.markdown("1. Sample images")
    st.markdown("2. Upload an image from your device")
    st.markdown("3. Paste a secure Image URL into the text box")
    st.markdown("4. Allow Access to your camera and take a picture.")

    # ---------- SAMPLE IMAGES ----------
    st.subheader("Sample Images")
    cols = st.columns(len(PRESETS))
    for col, (name, url) in zip(cols, PRESETS.items()):
        with col:
            if st.button(name):
                img = safe(lambda: load_image_from_url(url))
                if img:
                    set_current(img, f"preset_{name}")

            if st.session_state.current["source"] == f"preset_{name}":
                st.image(st.session_state.current["image"], width=250)
                if st.button("Generate Caption", key=f"gen_{name}"):
                    with st.spinner("Generating caption..."):
                        st.session_state.current["caption"] = safe(
                            lambda: generate_caption(st.session_state.current["image"])
                        )

    # ---------- UPLOAD ----------
    st.divider()
    st.subheader("Upload Image")
    uploaded = st.file_uploader("Upload", type=["jpg", "png", "jpeg"])
    if uploaded:
        set_current(Image.open(uploaded).convert("RGB"), "upload")

    if st.session_state.current["source"] == "upload":
        st.image(st.session_state.current["image"], width=250)
        if st.button("Generate Caption", key="gen_upload"):
            with st.spinner("Generating caption..."):
                st.session_state.current["caption"] = safe(
                    lambda: generate_caption(st.session_state.current["image"])
                )

    # ---------- URL ----------
    st.divider()
    st.subheader("Image URL")
    st.session_state.url_input = st.text_input(
        "Paste image URL",
        value=st.session_state.url_input
    )
    if st.button("Load Image from URL"):
        img = safe(lambda: load_image_from_url(st.session_state.url_input))
        if img:
            set_current(img, "url")
            st.session_state.url_input = ""

    if st.session_state.current["source"] == "url":
        st.image(st.session_state.current["image"], width=250)
        if st.button("Generate Caption", key="gen_url"):
            with st.spinner("Generating caption..."):
                st.session_state.current["caption"] = safe(
                    lambda: generate_caption(st.session_state.current["image"])
                )

    # ---------- CAMERA ----------
    st.divider()
    st.subheader("Camera")
    if st.checkbox("Use Camera"):
        cam = st.camera_input("Take a picture")
        if cam:
            set_current(Image.open(cam).convert("RGB"), "camera")

    if st.session_state.current["source"] == "camera":
        st.image(st.session_state.current["image"], width=250)
        if st.button("Generate Caption", key="gen_camera"):
            with st.spinner("Generating caption..."):
                st.session_state.current["caption"] = safe(
                    lambda: generate_caption(st.session_state.current["image"])
                )

    # ---------- CAPTION + TTS ----------
    if st.session_state.current["caption"]:
        st.success(st.session_state.current["caption"])

        st.subheader("🔊 Text to Speech")

        voice = st.selectbox(
            "Voice",
            ["en-US-JennyNeural", "en-US-GuyNeural", "en-GB-SoniaNeural"]
        )
        rate = st.slider("Speech Rate", -50, 50, 0, step=5)
        volume = st.slider("Volume", -50, 50, 0, step=5)

        if st.button("🔊 Play Caption"):
            audio = caption_to_audio(
                st.session_state.current["caption"],
                voice,
                f"{rate:+}%",
                f"{volume:+}%"
            )
            st.audio(audio)

        st.session_state.processed.append({
            "image": st.session_state.current["image"],
            "caption": st.session_state.current["caption"]
        })

# ======================================================
# TAB 2 — PROCESSED
# ======================================================
with tab2:
    if not st.session_state.processed:
        st.info("No processed images yet.")
    else:
        for item in st.session_state.processed:
            st.image(item["image"], width=200)
            st.markdown(item["caption"])
            st.divider()

# ======================================================
# TAB 3 — INSTRUCTIONS
# ======================================================
with tab3:
    st.markdown("""
• Choose an image source  
• Generate a caption  
• Listen to the caption with neural TTS  
• Results persist across the session  

Designed for Streamlit Cloud and classroom use.
""")
