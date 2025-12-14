import streamlit as st
from PIL import Image
import pytesseract
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Image Caption + TTS",
    layout="centered"
)

st.title("📸 Image Captioning with Browser Text-to-Speech")

# -----------------------------
# LOAD MODEL (CACHED)
# -----------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return model, processor, device

model, processor, device = load_model()

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -----------------------------
    # CAPTION GENERATION
    # -----------------------------
    with st.spinner("Generating caption..."):
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

    st.subheader("📝 Generated Caption")
    st.write(caption)

    # -----------------------------
    # OCR (OPTIONAL)
    # -----------------------------
    extracted_text = pytesseract.image_to_string(image).strip()

    if extracted_text:
        st.subheader("🔍 Extracted Text")
        st.write(extracted_text)

    # -----------------------------
    # 🔊 BROWSER TEXT-TO-SPEECH
    # -----------------------------
    st.subheader("🔊 Text-to-Speech")

    st.markdown(
        f"""
        <script>
        function speakText() {{
            const text = `{caption}`;
            const msg = new SpeechSynthesisUtterance(text);
            msg.lang = "en-US";
            msg.rate = 1;
            msg.pitch = 1;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(msg);
        }}
        </script>

        <button onclick="speakText()" style="
            font-size:18px;
            padding:10px 20px;
            border-radius:8px;
            border:none;
            background-color:#4CAF50;
            color:white;
            cursor:pointer;
        ">
            🔊 Speak Caption
        </button>
        """,
        unsafe_allow_html=True
    )
