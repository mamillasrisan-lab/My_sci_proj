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
# SESSION STATE
# ===============================
if "current" not in st.session_state:
    st.session_state.current = {
        "image": None,
        "caption": None,
        "source": None
    }

if "processed" not in st.session_state:
    st.session_state.processed = []

def reset_on_source_change(source):
    if st.session_state.current["source"] != source:
        st.session_state.current = {
            "image": None,
            "caption": None,
            "source": source
        }

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
        output = model.generate(**inputs, max_new_tokens=40)
    return processor.decode(output[0], skip_special_tokens=True)

# ===============================
# TABS
# ===============================
tab1, tab2, tab3 = st.tabs(["Generate Caption", "Processed Images", "Instructions"])

# ======================================================
# TAB 1 — GENERATE
# ======================================================
with tab1:
    st.markdown("""
Choose a source, select an image, then click **Generate Caption**.
Only the active source will display content.
""")

    # ---------- PRESETS ----------
    st.subheader("Sample Images")
    reset_on_source_change("preset")

    cols = st.columns(len(PRESETS))
    for col, (name, url) in zip(cols, PRESETS.items()):
        with col:
            if st.button(name, key=f"preset_{name}"):
                st.session_state.current["image"] = load_image_from_url(url)
                st.session_state.current["caption"] = None

    if st.session_state.current["source"] == "preset" and st.session_state.current["image"]:
        st.image(st.session_state.current["image"], width=300)
        if st.button("Generate Caption", key="gen_preset"):
            with st.spinner("Generating caption..."):
                caption = generate_caption(st.session_state.current["image"])
                st.session_state.current["caption"] = caption
                st.session_state.processed.append({
                    "image": st.session_state.current["image"],
                    "caption": caption
                })

    # ---------- UPLOAD ----------
    st.divider()
    st.subheader("Upload Image")
    reset_on_source_change("upload")

    uploaded = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
    if uploaded:
        st.session_state.current["image"] = Image.open(uploaded).convert("RGB")
        st.session_state.current["caption"] = None

    if st.session_state.current["source"] == "upload" and st.session_state.current["image"]:
        st.image(st.session_state.current["image"], width=300)
        if st.button("Generate Caption", key="gen_upload"):
            with st.spinner("Generating caption..."):
                caption = generate_caption(st.session_state.current["image"])
                st.session_state.current["caption"] = caption
                st.session_state.processed.append({
                    "image": st.session_state.current["image"],
                    "caption": caption
                })

    # ---------- URL ----------
    st.divider()
    st.subheader("Image URL")
    reset_on_source_change("url")

    url = st.text_input("Paste image URL")
    if url:
        st.session_state.current["image"] = load_image_from_url(url)
        st.session_state.current["caption"] = None

    if st.session_state.current["source"] == "url" and st.session_state.current["image"]:
        st.image(st.session_state.current["image"], width=300)
        if st.button("Generate Caption", key="gen_url"):
            with st.spinner("Generating caption..."):
                caption = generate_caption(st.session_state.current["image"])
                st.session_state.current["caption"] = caption
                st.session_state.processed.append({
                    "image": st.session_state.current["image"],
                    "caption": caption
                })

    # ---------- CAMERA ----------
    st.divider()
    st.subheader("Camera")
    reset_on_source_change("camera")

    camera_img = st.camera_input("Take a picture")
    if camera_img:
        st.session_state.current["image"] = Image.open(camera_img).convert("RGB")
        st.session_state.current["caption"] = None

    if st.session_state.current["source"] == "camera" and st.session_state.current["image"]:
        st.image(st.session_state.current["image"], width=300)
        if st.button("Generate Caption", key="gen_camera"):
            with st.spinner("Generating caption..."):
                caption = generate_caption(st.session_state.current["image"])
                st.session_state.current["caption"] = caption
                st.session_state.processed.append({
                    "image": st.session_state.current["image"],
                    "caption": caption
                })

    # ---------- RESULT ----------
    if st.session_state.current["caption"]:
        st.success(st.session_state.current["caption"])
        if st.button("🔊 Read Caption Aloud"):
            speak(st.session_state.current["caption"])

# ======================================================
# TAB 2 — PROCESSED IMAGES
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
### How this app works

• Choose an image source  
• Select or upload an image  
• Click **Generate Caption**  
• Optional: read caption aloud  

Captions and images are saved in **Processed Images**.

This app is optimized for education and research.
""")
