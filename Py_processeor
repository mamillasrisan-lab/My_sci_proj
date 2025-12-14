import os
import shutil
import sys
import time
import re
from pathlib import Path

from PIL import Image
import pytesseract
from gtts import gTTS
from playsound3 import playsound
import torch
from transformers import BlipForConditionalGeneration, AutoProcessor
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from plyer import notification

# ---------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------
# Folder to watch for new images
watch_folder = r"C:\Users\Srithan\Python_Projects\My_Sci_Fair_Project\Image_Input"
# Folder to move processed images
processed_folder = r"C:\Users\Srithan\Python_Projects\My_Sci_Fair_Project\Processed"
# CSV output
csv_path = r"C:\Users\Srithan\Downloads\parsed_images.csv"
# Google Sheets config
SERVICE_ACCOUNT_FILE = r"C:\Users\Srithan\service_account.json"
sheet_url = "https://docs.google.com/spreadsheets/d/1RL5Du_GcQQZkBVcKlkHqnsy-KJB_jsoCjjNdKUBfzMg/edit#gid=0"
sheet_tab_name = "Sheet1"

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def clean_text(text):
    return text.replace("\n", " ").replace("\r", " ").strip()

def notify(title, message):
    notification.notify(title=title, message=message, timeout=5)

# ---------------------------------------------------------
# SETUP MODEL
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)
processor = AutoProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base", use_fast=False
)

# ---------------------------------------------------------
# INIT DATA
# ---------------------------------------------------------
image_names = []
captions = []
extracted_texts = []
capt_times = []
proc_times = []

os.makedirs(watch_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)

# ---------------------------------------------------------
# GOOGLE SHEETS SETUP
# ---------------------------------------------------------
creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE,
                                              scopes=["https://www.googleapis.com/auth/spreadsheets",
                                                      "https://www.googleapis.com/auth/drive"])
gc = gspread.authorize(creds)
sh = gc.open_by_url(sheet_url)
worksheet = sh.worksheet(sheet_tab_name)

# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
print("Watching folder for new images...")

processed_files = set()

while True:
    try:
        files = [f for f in os.listdir(watch_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for file in files:
            if file in processed_files:
                continue  # Skip already processed files

            file_path = os.path.join(watch_folder, file)
            start_time = time.perf_counter()

            try:
                # Open image
                img = Image.open(file_path).convert("RGB")

                # Generate caption
                inputs = processor(img, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(**inputs)
                    caption = processor.decode(out[0], skip_special_tokens=True)
                    caption = clean_text(caption)

                # Extract text
                extracted_text = clean_text(pytesseract.image_to_string(img))

                # TTS
                full_text = f"Caption: {caption}. Extracted text: {extracted_text}"
                audio_file = f"temp_{file}.mp3"
                speech = gTTS(text=full_text, lang='en')
                speech.save(audio_file)
                playsound(audio_file)
                os.remove(audio_file)

                end_time = time.perf_counter()
                total_time = round(end_time - start_time, 2)

                # Move processed file
                shutil.move(file_path, os.path.join(processed_folder, file))

                # Append to data
                image_names.append(file)
                captions.append(caption)
                extracted_texts.append(extracted_text)
                capt_times.append(total_time)
                proc_times.append(total_time)

                # Notify
                notify("Image Processed", f"{file} processed!\nCaption: {caption}")

                # Update CSV
                df = pd.DataFrame({
                    "Image Name": image_names,
                    "Caption": captions,
                    "Extracted Text": extracted_texts,
                    "Processing Time": proc_times
                })
                df.to_csv(csv_path, index=False)

                # Update Google Sheets
                worksheet.clear()
                worksheet.update([df.columns.tolist()] + df.astype(str).values.tolist())

                processed_files.add(file)

            except Exception as e:
                notify("Processing Error", f"Failed to process {file}\n{e}")
                continue

        time.sleep(2)  # Check every 2 seconds

    except KeyboardInterrupt:
        print("Stopping image watcher...")
        break
    except Exception as e:
        notify("Watcher Error", str(e))
        time.sleep(5)
