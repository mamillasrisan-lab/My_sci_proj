import os
import sys
import tempfile
import time
import shutil
import re

from PIL import Image
import pytesseract
from gtts import gTTS
from playsound3 import playsound
import torch
from transformers import BlipForConditionalGeneration, AutoProcessor

import pandas as pd
import streamlit as st

# ------------------------------
# Automatic environment detection
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Detected device: {device}")

# Use a temporary directory for audio files
tmp_dir = tempfile.gettempdir()

# Base app folder
base_dir = os.path.dirname(sys.argv[0])  # relative to where app runs

# Input/output folders
watch_folder = os.path.join(base_dir, "Image_Input")
processed_folder = os.path.join(base_dir, "Processed")
os.makedirs(watch_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)

# CSV output
csv_path = os.path.join(base_dir, "parsed_images.csv")

# ------------------------------
# Load BLIP model
# ------------------------------
@st.cache_resource
def load_model():
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base", use_fast=False
    )
    return model, processor

model, processor = load_model()

# ------------------------------
# Helper functions
# ------------------------------
def clean_text(text):
    return text.replace("\n", " ").replace("\r", " ").strip()

def process_image(file_path):
    start_time = time.perf_counter()
    img = Image.open(file_path).convert("RGB")

    # Caption
    inputs = processor(img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        caption = clean_text(caption)

    # Text extraction
    extracted_text = clean_text(pytesseract.image_to_string(img))

    # TTS
    audio_file = os.path.join(tmp_dir, f"temp_{os.path.basename(file_path)}.mp3")
    speech = gTTS(text=f"Caption: {caption}. Extracted text: {extracted_text}", lang='en')
    speech.save(audio_file)
    playsound(audio_file)
    os.remove(audio_file)

    # Move file
    shutil.move(file_path, os.path.join(processed_folder, os.path.basename(file_path)))

    total_time = round(time.perf_counter() - start_time, 2)
    return caption, extracted_text, total_time

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🖼️ Automatic Image Caption & Text Extractor")

uploaded_files = st.file_uploader("Upload images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    results = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(watch_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        caption, text, proc_time = process_image(file_path)
        results.append({
            "Image Name": uploaded_file.name,
            "Caption": caption,
            "Extracted Text": text,
            "Processing Time": proc_time
        })

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    st.subheader("Processed Results")
    st.dataframe(df)

    st.success(f"✅ All images processed! CSV saved at: {csv_path}")
