import streamlit as st
from PIL import Image
import os
import platform

# -----------------------------
# 🔍 ADAPTIVE OCR SETUP
# -----------------------------
ocr_available = False
ocr_engine = "None"

try:
    import pytesseract

    # Try auto-detecting Tesseract path
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
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Smart Image Processor",
    page_icon="📷",
    layout="centered"
)

st.title("📷 Smart Image Processor Hi daddy :)")
st.caption("Upload or take a photo → adaptive OCR → button-activated TTS")

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
# IMAGE DISPLAY + OCR
# -----------------------------
if image:
    st.image(image, caption="Selected Image", use_container_width=True)

    st.subheader("2️⃣ OCR Text Extraction")

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

    # -----------------------------
    # 🔊 BUTTON-ACTIVATED TTS
    # -----------------------------
    st.subheader("3️⃣ Text-to-Speech")

    if ocr_available and ocr_text:
        st.markdown(
            f"""
            <script>
            function speakText() {{
                const text = `{ocr_text.replace("`", "")}`;
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
                🔊 Speak Extracted Text
            </button>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("No text available for speech.")

else:
    st.info("Please upload an image or take a photo to begin.")
