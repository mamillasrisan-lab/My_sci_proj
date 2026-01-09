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
if "active_image" not in st.session_state:
    st.session_state.active_image = None
if "active_caption" not in st.session_state:
    st.session_state.active_caption = None
if "active_source" not in st.session_state:
    st.session_state.active_source = None
if "processed" not in st.session_state:
    st.session_state.processed = []
if "url_input" not in st.session_state:
    st.session_state.url_input = ""

# ===============================
# FUNCTIONS
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

def set_active(img, source):
    st.session_state.active_image = img
    st.session_state.active_caption = None
    st.session_state.active_source = source

def tts_button(text):
    components.html(
        f"""
        <script>
        function speak() {{
            const msg = new SpeechSynthesisUtterance({text!r});
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(msg);
        }}
        </script>
        <button onclick="speak()" style="
            padding:8px 12px;
            font-size:14px;
            border-radius:6px;
            border:1px solid #ccc;
            cursor:pointer;">
            🔊 Read Caption Aloud
        </button>
        """,
        height=60
    )

# ===============================
# TABS
# ===============================
tab1, tab2, tab3 = st.tabs(["Generate Caption", "Processed Images", "Instructions"])

# ======================================================
# TAB 1 — GENERATE
# ======================================================
with tab1:
    st.markdown("""
Once you choose a source and choose an image, the **Generate Caption** button will appear
below that source.
""")

    st.markdown("**Options**")
    st.markdown("1. Sample images")
    st.markdown("2. Upload an image")
    st.markdown("3. Paste an image URL")
    st.markdown("4. Use your camera")

    # ---------- SAMPLE IMAGES ----------
    st.subheader("Sample Images")
    cols = st.columns(len(PRESETS))

    for col, (name, url) in zip(cols, PRESETS.items()):
        with col:
            if st.button(name, key=f"preset_{name}"):
                img = safe(lambda: load_image_from_url(url))
                if img:
                    set_active(img, f"preset_{name}")

            if st.session_state.active_source == f"preset_{name}":
                st.image(st.session_state.active_image, width=250)
                if st.button("Generate Caption", key=f"gen_{name}"):
                    with st.spinner("Generating caption..."):
                        st.session_state.active_caption = generate_caption(
                            st.session_state.active_image
                        )
                        st.session_state.processed.append({
                            "image": st.session_state.active_image,
                            "caption": st.session_state.active_caption
                        })

                if st.session_state.active_caption:
                    st.success(st.session_state.active_caption)
                    tts_button(st.session_state.active_caption)

    st.divider()

    # ---------- UPLOAD ----------
    st.subheader("Upload Image")
    uploaded = st.file_uploader("Upload", type=["jpg", "png", "jpeg"])
    if uploaded:
        set_active(Image.open(uploaded).convert("RGB"), "upload")

    if st.session_state.active_source == "upload":
        st.image(st.session_state.active_image, width=300)
        if st.button("Generate Caption", key="gen_upload"):
            with st.spinner("Generating caption..."):
                st.session_state.active_caption = generate_caption(
                    st.session_state.active_image
                )
                st.session_state.processed.append({
                    "image": st.session_state.active_image,
                    "caption": st.session_state.active_caption
                })
        if st.session_state.active_caption:
            st.success(st.session_state.active_caption)
            tts_button(st.session_state.active_caption)

    st.divider()

    # ---------- URL ----------
    st.subheader("Image URL")
    st.session_state.url_input = st.text_input(
        "Paste image URL",
        value=st.session_state.url_input
    )

    if st.button("Load Image from URL"):
        img = safe(lambda: load_image_from_url(st.session_state.url_input))
        if img:
            set_active(img, "url")
            st.session_state.url_input = ""

    if st.session_state.active_source == "url":
        st.image(st.session_state.active_image, width=300)
        if st.button("Generate Caption", key="gen_url"):
            with st.spinner("Generating caption..."):
                st.session_state.active_caption = generate_caption(
                    st.session_state.active_image
                )
                st.session_state.processed.append({
                    "image": st.session_state.active_image,
                    "caption": st.session_state.active_caption
                })
        if st.session_state.active_caption:
            st.success(st.session_state.active_caption)
            tts_button(st.session_state.active_caption)

    st.divider()

    # ---------- CAMERA ----------
    st.subheader("Camera")
    use_camera = st.checkbox("Use Camera", value=False)

    if use_camera:
        camera_img = st.camera_input("Take a picture")
        if camera_img:
            set_active(Image.open(camera_img).convert("RGB"), "camera")

    if st.session_state.active_source == "camera":
        st.image(st.session_state.active_image, width=300)
        if st.button("Generate Caption", key="gen_camera"):
            with st.spinner("Generating caption..."):
                st.session_state.active_caption = generate_caption(
                    st.session_state.active_image
                )
                st.session_state.processed.append({
                    "image": st.session_state.active_image,
                    "caption": st.session_state.active_caption
                })
        if st.session_state.active_caption:
            st.success(st.session_state.active_caption)
            tts_button(st.session_state.active_caption)

# ======================================================
# TAB 2 — PROCESSED
# ======================================================
with tab2:
    if not st.session_state.processed:
        st.info("No processed images yet.")
    else:
        for item in st.session_state.processed:
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
• Click **Generate Caption**  
• Use **🔊 Read Caption Aloud** to hear it  
• View history in **Processed Images**  
""")
