import streamlit as st
from PIL import Image
import torch
from transformers import BlipForConditionalGeneration, AutoProcessor
import requests
from io import BytesIO
import time

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="BLIP Image Captioning",
    page_icon="📷",
    layout="centered"
)

st.title("📷 BLIP-1 Image Captioning")

# -----------------------------
# PRESET IMAGE URLS
# -----------------------------
PRESET_IMAGES = {
    "Wildfires with Cars": "https://lh3.googleusercontent.com/rd-d/ALs6j_GdEknkM1jYBD0aCGKw4CpGhSGxF2297IySBYujOPxV6qtNM3IbnglaQY2Y9wvp0jg7P-MpNTcUgx0T_NMAhTXtmmdZBfG5mizTvwxuIME1ywEqOOobdr596M4FrKwOK0LOX9RYqiB4-Rr4LwLExLOMocNu1ZVsJEFH3jzva2yQom1O-B7bDamExLpqi7_p-qGZwDz-Vi8MbTZLWhwHwfE0pusS7bs2sRAe6Cp-n6Ml-Gf4le5Gmfa9Zingp6I2JlBJUn8qtJ9MCK_zsdl5GOH5JQzewXHRz5sZOuUsC8WK-8pmuGiXlk5vHKWYvuIGl8tqTOMCiC4ndcs-m6CLZYfinY6jF23q14RVekaEtRhrRzEWrqd4J4mAKQxuK03MNAcT1Rys0dW9AlqZ8lAKmvakZOUf9XuySeJ8gtn5EU2AOq9W55cQFAiWhfvZ3GDbAKVGLzlXK0B637b137VD12RiLCu5gzPt0bmwgX_zbp6RR4vSWaLtrlP8e1G1iq9UNG0hjREFqymuq2IicD0qHarVcy9AOj7qFSqCmTT38NlMOlUdvFC60hixY4bbZj4C4hFykt1-G_433RGJKgfF5XAZA1KGxhQPLK4DRjLFQZPklllF1Zkk4WeK6FQ4fIpI7JZ1Q5wqaLAiB9HjIv8nSty3jI6q1wOB2xNvUs_U4HVNxgi3c53wM7_MlvXnLrhCWdijjdiFYewWDgRlmnXeooG31d8FYd2fgda-cWUqlERfRfcT5eh-NYTAJ2dfa4MjcP-8arrUjz40-08HHeH3cevQ24PmT0-3Lj7Ytnh-6yWLJCBz-O5K4cm2wT1ygDgRfAfDslHX_3cdm2o0JS6jciN9nZizkDVws9mSdcV-IBtf0YNHQDrGPDg0iSL_5WYPdd9o6lIykh8qLHaW7Sd3KSGelqPCQBvDD1aBjM18EVWFP4aWMOUBHXvjc7OH1N9wKtK2HUumyI7M6CV2MgXSmPJUit1yQrVGH7b0i4IIGR2IpPcW5JqsZwX7dmL1B2A84f35DA=s1600",
    "Historical Exhibit Room": "https://lh3.googleusercontent.com/rd-d/ALs6j_HQyzMuFhQZJhffKXr2zjW-WuUvLhnkp-eyH187JPfHL9YPFE31OIjxlHSi0IlEXSaVVyWN6EZMT0TOHo4PG7CiYclK43_yx6cMS0RyVgfZ7tDWxhKJ5VgQXsjxSFBFD7TJwSxB_XR7TjPK4CNyfzkvLVUK_tBzzMAZ0FRncN3kWmJ-NMDEZLDGooKU-4t0LgJKoRSdhwSmyAJ36iinTLTaeT-2FEFH8ResmgsitJQGaMa469vkBE_-FQJhvHt3pWznZ78mpz60dlNuxGzWdpGBvqctAssjnURoq-qkSKMXiDTjSUSWr7ZhBp0n7NfGCr5CpXf_UMoiWKPR--n-ZKdWezsQOPCUbwXW4J25MK1EUs96oGVCul56NfYKySAp8IfpfPA4bPPtWeGGhNRinB9OITr0bC7rDkpkHamcmHUxC8tUMI6zOEWb8vXobydBMNx-ilxlJBo2DXoqOudC9iaP-ddL-aQL1ejo27oYOw_yAjK_yqHy7QrYS_0zFyuHm9osPIBtI0qQnm2JMZbX9WrHmv1A865goQx7gE4ejwCJ5vULBHlFVaFnSoFHXcRWwq7tiyulIyHeUekXYV8TDB6rg0PVvnnoVKU1K5Qa-5awtOp3TBvI_yBb5-A7yk-75Oe-XqCb00ypOchRmDttawpROPq_EkwJd6KdaH5Yif0LwoEHNMuV8C5KAL6UodgRlLMqFGAIPPOqqKf_KjJwptjMxFDcqfBq85thnAHicH-h-wdUEg1AmJOUB-UDqpvAdfo172Tt3A4V6j7agi_w6QUAiDMSwWkt4xBlOYGbFebaLD4TWD6hSX8T1oQq2GZ5uzurH6MpsKX99nDe0FKxwDeZATH-q8h5NwsXtT91tDF_ngtmKXrjKtd-zGUqqJGc8gKI5fNvgHi2xFsTIMRJVrZ2bfvxAbRwFDJR9bBBQaneXdNM6oTPCAxHbsJnBs7ycXKoONzk0MwgOhJEBaoa6z4JfRGXI17EueSXuTMTPj5OVoZ7JtEVh3H89scMI2bKUdCSDA=s1600",
    "Fruit Flies in Farms": "https://lh3.googleusercontent.com/rd-d/ALs6j_EBW0qhEE6KnfKKNKYBqIQGSAUr7QnMwA8qCfbtvM2Pc5TZPl9DCSmn3m8LAL23ivDZC_Atud7gS1X9hSuHnaU3v_FYjgrFoBG4FPcfcm5tyjQNVJwHn5F_lWxR2b1IDepl3la5fdlozt1hGoSd92CZRIOzhRqeYOp_ON5s95qqYRGpaZw5auIVH5x9BRJt70o81wRF4xaASIsDa7M0ecIZzMNTd8Q8gDMioUSY2xjpWJ3lqS5xUagb9l6kb2ZoG7LRJIU3fqYw3AIfaHc28sr0IrtXzCUPnnOvwlXmgwOWzt2Q8Wn__UP1uUqe6bry55DdpsYYN0dnrVU-TOmdaWW4wTMZHx23wjyTZgRywK4AeblLz-ynxbr8U-jDDRbL2Soqaz_jWqei0kE-AlB4mrzO-6uHt3fpXNIelUNWf_obMxctjGrcAm5BwY80IvAEyZVi-5p1zucUKtxbtPcqyGT8xWYcOPgWOLCJJ1ZVmw9YN-jGwvaV394yZSoXDXhcpWhto4xC-_e-3iZ7H-WMnazpyZ8cV60yqy3SruzhmJ82fICvBKVr10g-qkjSniNjwBFKoDqNFs1--cus98SgjFR1gYo9FepCJQ5CnckkflzkwWjDleJ3A3NHMZC029gHWYe7hf7-4XLUFfXk7RGx_Krkk9V_ccUKWujoITzW3mhiSwml-6CeJrajV2Ub_WT5GPS0n8mNXGLAOmzpKpT2Xc4W3_3Xew_3smJKZoMXZQyODNzPiCy_80T7Z1P1CTqqduSPqWjVT2alA4wISB4eGIC3OrafrHtnRFIV3StiDcsRS71CsF1kQ1Bnos_8iDq3PrNBzk16n4AUuFX-Yx-utXMBnqGaba3VZKeHHGjzC7kB0g9YJwkpE3_V-Vo2wT6MsiBZ1ZJXiYgYhjstLiFY2GDaR-uOjWwN9V6zxdu4zf1IrSZrgw1ko-qx1h7fY2iRsozOZz0mAid6sya5YUs0DAn7FSHvCZIN_7L5Dhn36joH146TpPsfATB0txAYGn5aU7iigQ=s1600",
    "Museum Sculpture": "https://lh3.googleusercontent.com/rd-d/ALs6j_GLZOcvj-TI3xzwqLb_y5LtZqgr_xPYQRStlM007LzbpHruQCBq2PikFDPIv_pKPK40gZ48EVfE3HtduiQAJf1Go79V-21T5HUHXO_kBkpRUw7N1h_fa-bn2bir46XzaCG99Sc8WBaeqUmO6JQqh2NwMA-eyIrRUXi9KkhM4LcJnce9Yaw_iCYVrQmBCw1w2JGwnELtsHaQC738gvwqR9MFFFSv75RLs5q1gTj0W0MGD1-PZVqPchlUohvtE1OH0EyXGoiT-aGagpjwTMeHrE7VSVxD47fVHsL9xaLpZ69aM7OZPQwyShCvMbz1q6YQL2wDx01CU47mSJ9FMa5iDN8x7LMQBUvFdiZT41Y1iZZP97HtgtQVWbKx4c3bL1c6Y0i2A4y84iDWrtj1flPnlpgJEKIlKNbghWPGJLf0QSyUWF3W5HFRLNvRmn9XWrKKxR4vlfYKe0zIrtlHK0yzlWNm9NXYQcurism0Du8kWHrXjzNSH3x65SZibGUoPqpcMh7J2njc9y2QIOk0G9cGkXVIiPjlTdg_ftLBrrus-WnSNLYeBRaikFBXxzRvh-SRuqvo-lkzc4PZVmpHPwDjjT_DZM2iDAG4A-E63sLK5VudVlfRWAxwMNK9KH4GkM1XMxEax1VaoCzrVoVJylZZzGUqtRaE0xSF08T0rpsvMyghLJtKUHATkuQfC74kh36ydVIjwzrQswvETt0P79fk_4D5bF07drYQq9lWOdUdq_1RcwoaEc-IW7SdYme1P9TXh5AlSCIxZ6XdT-BjuQAi74VUV1hp2oWv-ZPcuTFDWFI2P_RpA1QSP3o6WYhMUNR6-X0CEokYfcI04oTWOQTV5oRofZ7c2fPW8TPSvc0_IMT09p8fHdAiwL7TluVilCCWhSByYvoEmf3AT1z9YWOKCKBbdz7DhNFre-FshFTMNY5d4Z1_X1sKgfVV8Ht85HicU5tQ5hPpWIweUEe8BTW4WVK-zajYCo8HKdQvXmLHZmFwbbksk5MJz3FrGzs-nbHVvbo0Zw=s1600"
}

# -----------------------------
# SESSION STATE
# -----------------------------
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

if "processed" not in st.session_state:
    st.session_state.processed = []

# -----------------------------
# LOAD BLIP
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_blip():
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    return processor, model

processor, model = load_blip()

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Generate Caption", "Processed Images", "Helper"])

# =============================
# GENERATE TAB
# =============================
with tab1:
    st.subheader("Preset Images")
    cols = st.columns(4)

    for col, (name, url) in zip(cols, PRESET_IMAGES.items()):
        with col:
            if st.button(name):
                response = requests.get(url)
                st.session_state.selected_image = Image.open(BytesIO(response.content))

    st.divider()

    uploaded = st.file_uploader("Upload Image", ["jpg", "png", "jpeg"])
    use_camera = st.checkbox("Use Camera")
    camera_image = st.camera_input("Camera", disabled=not use_camera)
    url_input = st.text_input("Or Image URL")

    if uploaded:
        st.session_state.selected_image = Image.open(uploaded)
    elif camera_image:
        st.session_state.selected_image = Image.open(camera_image)
    elif url_input:
        try:
            r = requests.get(url_input)
            st.session_state.selected_image = Image.open(BytesIO(r.content))
        except:
            pass

    if st.session_state.selected_image:
        st.image(st.session_state.selected_image, caption="Selected Image")

        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                inputs = processor(
                    st.session_state.selected_image, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    out = model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)

                st.success(caption)
                st.session_state.processed.append(
                    (st.session_state.selected_image.copy(), caption)
                )

# =============================
# PROCESSED TAB
# =============================
with tab2:
    if st.session_state.processed:
        for img, cap in st.session_state.processed:
            st.image(img, caption=cap)
    else:
        st.info("No images processed yet.")

# =============================
# HELPER TAB
# =============================
with tab3:
    st.info(
        """
**How to use this app**
• Pick a preset image  
• OR upload an image  
• OR enable the camera checkbox  
• OR paste an image URL  

Click **Generate Caption** to describe the image using BLIP-1.
"""
    )
