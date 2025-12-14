import streamlit as st
from PIL import Image
import os
import platform
import torch

# -----------------------------
# BLIP-2 IMPORTS
# -----------------------------
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# -----------------------------
# ADAPTIVE OCR SETUP
# -----------------------------
ocr_available = False
ocr_engine = "None"

try:
    import pytesseract

    # Auto-detect Tesseract path
    if platform.system() == "Windows":
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                ocr_available = True
                ocr_engine = "Tesseract (Windows)"
                break
    elif platform.system() in ["Linux", "Darwin"]:
        ocr_available = True
        ocr_engine = "Tesseract (System)"

except Exception:
    ocr_available = False

# -----------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Smart Image Processor", page_icon="📷", layout="centered")
st.title("📷 Smart Image Processor (BLIP-2)")
st.write("Upload or take a photo → OCR + BLIP-2 caption → Button-activated TTS")

# -----------------------------
# IMAGE INPUT
# -----------------------------
st.subheader("1️⃣ Choose Image Source")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
camera_image = st.camera_input("Or take a photo")
image = None

if uploaded_file:
    image = Image.open(uploaded_file)
elif camera_image:
    image = Image.open(camera_image)

# -----------------------------
# IMAGE DISPLAY + OCR + CAPTION
# -----------------------------
if image:
    st.image(image, caption="Selected Image", use_container_width=True)

    # OCR
    st.subheader("2️⃣ OCR Text Extraction")
    ocr_text = ""
    if ocr_available:
        try:
            ocr_text = pytesseract.image_to_string(image).strip()
            if ocr_text:
                st.text_area("Extracted Text", ocr_text, height=200)
            else:
                st.info("No readable text detected.")
        except Exception as e:
            st.error("OCR failed at runtime.")
            st.code(str(e))
    else:
        st.warning("OCR engine not available on this system.")
    st.caption(f"OCR Engine: {ocr_engine}")

    # CAPTION (BLIP-2)
    st.subheader("3️⃣ Image Caption (BLIP-2)")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load BLIP-2 processor and model once
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

        st.markdown(f"**Caption:** {caption}")

    except Exception as e:
        st.warning("BLIP-2 captioning unavailable.")
        st.code(str(e))

    # -----------------------------
    # BUTTON-ACTIVATED TTS
    # -----------------------------
    st.subheader("4️⃣ Text-to-Speech")
    combined_text = ""
    if ocr_text:
        combined_text += ocr_text + " "
    if 'caption' in locals():
        combined_text += caption

    if combined_text:
        st.markdown(f"""
        <script>
        function speakText() {{
            const text = `{combined_text.replace("`","")}`;
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
            border-radius:10px;
            border:none;
            background-color:#1f77b4;
            color:white;
            cursor:pointer;
        ">
            🔊 Speak OCR + Caption
        </button>
        """, unsafe_allow_html=True)
    else:
        st.info("No text or caption available for speech.")

else:
    st.info("Please upload an image or take a photo to begin.")
