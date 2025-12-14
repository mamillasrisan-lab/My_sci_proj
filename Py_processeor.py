import os
import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# --------------------------------------------------
# Detect environment
# --------------------------------------------------
RUNNING_IN_BROWSER = os.getenv("STREAMLIT_SERVER_RUNNING") == "1"

# --------------------------------------------------
# Optional imports (ONLY for local use)
# --------------------------------------------------
if not RUNNING_IN_BROWSER:
    try:
        import pytesseract
        from gtts import gTTS
        from playsound3 import playsound
        OCR_AVAILABLE = True
    except Exception:
        OCR_AVAILABLE = False
else:
    OCR_AVAILABLE = False

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Image Understanding App", layout="centered")
st.title("🖼️ Image Understanding App")

st.markdown(
    """
This app automatically adapts to where it runs:

- 🌐 **Web version** → Image captioning only  
- 💻 **Desktop version** → Captioning + OCR + Speech
"""
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

# --------------------------------------------------
# Load BLIP model (cached)
# --------------------------------------------------
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model

processor, model = load_model()

# --------------------------------------------------
# Process image
# --------------------------------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

    st.subheader("📝 Image Caption")
    st.success(caption)

    # --------------------------------------------------
    # OCR (LOCAL ONLY)
    # --------------------------------------------------
    if OCR_AVAILABLE:
        st.subheader("🔎 Extracted Text (OCR)")
        try:
            text = pytesseract.image_to_string(image)
            if text.strip():
                st.text(text)
            else:
                st.info("No readable text detected.")
        except Exception as e:
            st.warning("OCR failed.")
    else:
        st.subheader("🔎 Extracted Text (OCR)")
        st.info("OCR is disabled in the web version.")

    # --------------------------------------------------
    # Text-to-Speech (LOCAL ONLY)
    # --------------------------------------------------
    if OCR_AVAILABLE:
        if st.button("🔊 Read Caption Aloud"):
            tts = gTTS(text=caption, lang="en")
            tts.save("caption.mp3")
            playsound("caption.mp3")
            os.remove("caption.mp3")

else:
    st.info("Please upload an image to begin.")
