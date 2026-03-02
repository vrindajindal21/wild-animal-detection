
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import tempfile

# ================= CONFIG =================
MODEL_PATH = "yolov8n.pt"
ALERT_SOUND = "siren-alert-96052.mp3"
WILD_CLASSES = ["bear", "elephant", "tiger", "lion", "leopard", "wolf", "giraffe", "zebra"]

# ================= HELPER FUNCTIONS =================
@st.cache_resource
def load_model():
    # Use the model in the current project directory
    return YOLO(MODEL_PATH)

def safe_label(model, cls_id):
    return model.names.get(cls_id, "").lower()

def process_frame(frame, model):
    results = model.predict(source=frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()
    alert_triggered = False
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = safe_label(model, cls_id)
        if label in WILD_CLASSES:
            alert_triggered = True
            break
    
    return annotated_frame, alert_triggered

# ================= MAIN APP =================
# ================= MAIN APP =================
st.set_page_config(
    page_title="Wild Animal Detection System", 
    page_icon="🐯", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load AI Model
model = load_model()

# ================= CSS: ORIGINAL GUI MIRROR =================
st.markdown("""
<style>
    /* Exact Background */
    .stApp {
        background-color: #E3F2FD !important;
    }

    /* Remove top padding of the first block */
    [data-testid="stVerticalBlock"] > div:first-child {
        padding-top: 50px !important;
    }

    /* Exact Title Styling from Screen */
    .original-title {
        color: #0D47A1;
        font-family: 'Arial Black', sans-serif;
        font-weight: 900;
        font-size: 32px;
        text-align: center;
        margin-bottom: 20px;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 10px;
    }

    /* Exact Tkinter-style Buttons */
    div.stButton > button {
        width: 300px !important;
        height: 2.5em !important;
        color: white !important;
        font-weight: bold !important;
        font-family: 'Arial', sans-serif !important;
        font-size: 16px !important;
        border: 2px solid rgba(0,0,0,0.1) !important;
        border-radius: 4px !important;
        margin: 0 auto !important;
        display: block !important;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2) !important;
    }

    /* Ensure buttons are centered */
    div.stButton {
        display: flex;
        justify-content: center;
    }

    /* Styling File Uploaders to look like Buttons */
    div[data-testid="stFileUploader"] {
        width: 300px !important;
        margin: 0 auto !important;
    }
    div[data-testid="stFileUploader"] section {
        padding: 0 !important;
        border: none !important;
        background-color: transparent !important;
    }
    div[data-testid="stFileUploader"] label {
        display: none !important;
    }
    /* Hide the 'Browse files' text and drag-drop area */
    div[data-testid="stFileUploader"] section > div:nth-child(2) {
        display: none !important;
    }
    /* Make the actual button look like our GUI Button */
    div[data-testid="stFileUploader"] button {
        width: 300px !important;
        height: 2.5em !important;
        color: white !important;
        font-weight: bold !important;
        font-family: 'Arial', sans-serif !important;
        font-size: 16px !important;
        border: 2px solid rgba(0,0,0,0.1) !important;
        border-radius: 4px !important;
        margin: 0 auto !important;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2) !important;
        transition: transform 0.1s !important;
    }
    
    /* Hide the default 'Browse files' text and replace it */
    div[data-testid="stFileUploader"] button {
        font-size: 0 !important; /* Hide original text */
    }
    
    /* Force 'Upload Images' text */
    [data-testid="stVerticalBlock"] > div:nth-child(3) div[data-testid="stFileUploader"] button::after {
        content: "Upload Images" !important;
        font-size: 16px !important;
        color: white !important;
        font-weight: bold !important;
    }
    
    /* Force 'Upload Videos' text */
    [data-testid="stVerticalBlock"] > div:nth-child(4) div[data-testid="stFileUploader"] button::after {
        content: "Upload Videos" !important;
        font-size: 16px !important;
        color: white !important;
        font-weight: bold !important;
    }

    /* Assign Colors */
    [data-testid="stVerticalBlock"] > div:nth-child(3) div[data-testid="stFileUploader"] button { background-color: #1976D2 !important; }
    [data-testid="stVerticalBlock"] > div:nth-child(4) div[data-testid="stFileUploader"] button { background-color: #4CAF50 !important; }
    /* 3. Live Detection - Orange */
    [data-testid="stVerticalBlock"] > div:nth-child(5) button { background-color: #FF9800 !important; }
    /* 4. Stop - Red */
    [data-testid="stVerticalBlock"] > div:nth-child(6) button { background-color: #F44336 !important; }
    /* 5. Exit - Purple */
    [data-testid="stVerticalBlock"] > div:nth-child(7) button { background-color: #8E24AA !important; }

    header, footer, #MainMenu { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="original-title">🐾 Wild Animal Detection System</div>', unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = "main"

# ================= MAIN MENU =================
if st.session_state.page == "main":
    # 1. Image Uploader (Styled as Blue Button)
    uploaded_images = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="img_up")
    
    # 2. Video Uploader (Styled as Green Button)
    uploaded_video = st.file_uploader("Upload Videos", type=["mp4", "avi", "mov"], key="vid_up")
    
    # 3. Live Detection
    if st.button("Start Live Detection"):
        st.session_state.page = "live"
        st.rerun()
        
    # 4. Stop
    if st.button("Stop Live Detection"):
        st.session_state.page = "main"
        # Clear uploads on stop
        st.rerun()

    # 5. Exit
    if st.button("Exit"):
        st.stop()

    # --- Result Display Section (Directly on Main Page) ---
    if uploaded_images:
        st.markdown("---")
        for uploaded_file in uploaded_images:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            annotated_img, alert = process_frame(image, model)
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
            if alert: st.audio(ALERT_SOUND, format="audio/mp3", autoplay=True)

    if uploaded_video:
        st.markdown("---")
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or st.button("Stop Playback", key="stop_vid"): break
            frame = cv2.resize(frame, (640, 480))
            annotated_frame, alert = process_frame(frame, model)
            st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        cap.release()
        os.remove(tfile.name)

# ================= LIVE MODE =================
elif st.session_state.page == "live":
    st.markdown('<div style="text-align:center; color:#0D47A1; font-weight:bold; font-size:20px;">📡 Live Mode Access</div>', unsafe_allow_html=True)
    if st.button("⬅️ Back to Menu"):
        st.session_state.page = "main"
        st.rerun()

    img_file_buffer = st.camera_input("Scanner Active")
    if img_file_buffer:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        annotated_img, alert = process_frame(cv2_img, model)
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        if alert: st.audio(ALERT_SOUND, format="audio/mp3", autoplay=True)

st.markdown("<br><br>", unsafe_allow_html=True)
