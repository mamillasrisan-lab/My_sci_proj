import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import BlipForConditionalGeneration, AutoProcessor

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Image Identification and Captioning",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# SAFE ERROR HANDLER
# ===============================
def safe(fn):
    try:
        return fn()
    except Exception:
        return None

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

processor, model
