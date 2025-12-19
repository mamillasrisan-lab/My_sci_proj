import streamlit as st
from PIL import Image, UnidentifiedImageError
import torch
from transformers import BlipForConditionalGeneration, AutoProcessor
import requests
from io import BytesIO

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="BLIP Image Captioning",
    page_icon="📷",
    layout="centered"
)
st.title("📷 BLIP Image Captioning")

# -----------------------------
# PRESET IMAGE URLS
# -----------------------------
PRESET_IMAGES = {
    "Wildfires with Cars": "https://lh3.googleusercontent.com/rd-d/ALs6j_GdEknkM1jYBD0aCGKw4CpGhSGxF2297IySBYujOPxV6qtNM3IbnglaQY2Y9wvp0jg7P-MpNTcUgx0T_NMAhTXtmmdZBfG5mizTvwxuIME1ywEqOOobdr596M4FrKwOK0LOX9RYqiB4-Rr4LwLExLOMocNu1ZVsJEFH3jzva2yQom1O-B7bDamExLpqi7_p-qGZwDz-Vi8MbTZLWhwHwfE0pusS7bs2sRAe6Cp-n6Ml-Gf4le5Gmfa9Zingp6I2JlBJUn8qtJ9MCK_zsdl5GOH5JQzewXHRz5sZOuUsC8WK-8pmuGiXlk5vHKWYvuIGl8tqTOMCiC4ndcs-m6CLZYfinY6jF23q14RVekaEtRhrRzEWrqd4J4mAKQxuK03MNAcT1Rys0dW9AlqZ8lAKmvakZOUf9XuySeJ8gtn5EU2AOq9W55cQFAiWhfvZ3GDbAKVGLzlXK0B637b137VD12RiLCu5gzPt0bmwgX_zbp6RR4vSWaLtrlP8e1G1iq9UNG0hjREFqymuq2IicD0qHarVcy9AOj7qFSqCmTT38NlMOlUdvFC60hixY4bbZj4C4hFykt1-G_433RGJKgfF5XAZA1KGxhQPLK4DRjLFQZPklllF1Zkk4WeK6FQ4fIpI7JZ1Q5wqaLAiB9HjIv8nSty3jI6q1wOB2xNvUs_U4HVNxgi3c53wM7_MlvXnLrhCWdijjdiFYewWDgRlmnXeooG31d8FYd2fgda-cWUqlERfRfcT5eh-NYTAJ2dfa4MjcP-8arrUjz40-08HHeH3cevQ24PmT0-3Lj7Ytnh-6yWLJCBz-O5K4cm2wT1ygDgRfAfDslHX_3cdm2o0JS6jciN9nZizkDVws9mSdcV-IBtf0YNHQDrGPDg0iSL_5WYPdd9o6lIykh8qLHaW7Sd3KSGelqPCQBvDD1aBjM18EVWFP4aWMOUBHXvjc7OH1N9wKtK2HUumyI7M6CV2MgXSmPJUit1yQrVGH7b0i4IIGR2IpPcW5JqsZwX7dmL1B2A84f35DA=s1600",
}

# -----------------------------
# SESSION STATE
# -----------------------------
st.session_state.setdefault("processed_images", [])
st.session_state.setdefault("selected_image", None)
st.session_state.setdefault("url_input", "")

# -----------------------------
# LOAD MODEL
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_blip():
    processor = AutoProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        use_fast=False
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    return processor, model

processor, model = load_blip()

# -----------------------------
# IMAGE LOADER (SAFE)
# -----------------------------
def load_image_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()

        img = Image.open(BytesIO(r.content))
        img.verify()

        return Image.open(BytesIO(r.content)).convert("RGB")
    except (UnidentifiedImageError, requests.RequestException):
        return None

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Generate Caption", "Processed Images", "Helper"])

# -----------------------------
# GENERATE TAB
# -----------------------------
with tab1:
    st.write("Select a preset image, upload one, use the camera, or enter a URL.")

    cols = st.columns(len(PRESET_IMAGES))
    for col, (name, url) in zip(cols, PRESET_IMAGES.items()):
        with col:
            if st.button(name):
                img = load_image_from_url(url)
                if img:
                    st.session_state.selected_image = img

    st.divider()

    uploaded = st.file_uploader("Upload an image", ["jpg", "jpeg", "png"])

    use_camera = st.checkbox("Use camera")
    camera_img = st.camera_input("Camera") if use_camera else None

    url = st.text_input("Image URL", key="url_input")

    image = (
        st.session_state.selected_image
        or (Image.open(uploaded).convert("RGB") if uploaded else None)
        or (Image.open(camera_img).convert("RGB") if camera_img else None)
        or load_image_from_url(url)
    )

    if image:
        st.image(image, width="stretch")
        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                inputs = processor(image, return_tensors="pt").to(device)
                with torch.no_grad():
                    output = model.generate(**inputs)
                caption = processor.decode(output[0], skip_special_tokens=True)

                st.success(caption)
                st.session_state.processed_images.append((image.copy(), caption))
                st.session_state.url_input = ""
    else:
        st.info("Waiting for a valid image...")

# -----------------------------
# PROCESSED TAB
# -----------------------------
with tab2:
    if st.session_state.processed_images:
        for img, cap in st.session_state.processed_images:
            st.image(img, caption=cap, use_container_width=True)
    else:
        st.info("No images processed yet.")

# -----------------------------
# HELPER TAB
# -----------------------------
with tab3:
    st.markdown("""
    ### How this app works
    - Choose a preset image, upload one, use the camera, or paste a URL
    - Click **Generate Caption**
    - BLIP automatically describes the image
    - All processed images appear in the second tab

    This app hides technical errors and only proceeds when an image is valid.
    """)
