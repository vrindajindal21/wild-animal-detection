
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

    /* Specific Hex Colors from original GUI - Indexing fixed */
    /* 1. Upload Images - Blue */
    [data-testid="stVerticalBlock"] > div:nth-child(3) button { background-color: #1976D2 !important; }
    /* 2. Upload Videos - Green */
    [data-testid="stVerticalBlock"] > div:nth-child(4) button { background-color: #4CAF50 !important; }
    /* 3. Live Detection - Orange */
    [data-testid="stVerticalBlock"] > div:nth-child(5) button { background-color: #FF9800 !important; }
    /* 4. Stop - Red */
    [data-testid="stVerticalBlock"] > div:nth-child(6) button { background-color: #F44336 !important; }
    /* 5. Exit - Purple */
    [data-testid="stVerticalBlock"] > div:nth-child(7) button { background-color: #8E24AA !important; }

    /* Hide Streamlit components */
    header, footer, #MainMenu { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)

# Centered Title (with paw icons)
st.markdown('<div class="original-title">🐾 Wild Animal Detection System</div>', unsafe_allow_html=True)

# Session State for navigation
if 'page' not in st.session_state:
    st.session_state.page = "main"

# Menu Container (Center aligned)
if st.session_state.page == "main":
    if st.button("Upload Images"):
        st.session_state.page = "images"
    if st.button("Upload Videos"):
        st.session_state.page = "videos"
    if st.button("Start Live Detection"):
        st.session_state.page = "live"
    if st.button("Stop Live Detection"):
        st.session_state.page = "main"
        st.rerun()
    if st.button("Exit"):
        st.stop()

# Execution Modules - Simplified for "Direct" feel
if st.session_state.page == "images":
    col1, col2 = st.columns([0.8, 0.2])
    with col2:
        if st.button("✖", help="Back to Menu"):
            st.session_state.page = "main"
            st.rerun()
    
    uploaded_files = st.file_uploader("📂 Drop images here to start...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            annotated_img, alert = process_frame(image, model)
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
            if alert: st.audio(ALERT_SOUND, format="audio/mp3", autoplay=True)

elif st.session_state.page == "videos":
    col1, col2 = st.columns([0.8, 0.2])
    with col2:
        if st.button("✖", help="Back to Menu"):
            st.session_state.page = "main"
            st.rerun()

    uploaded_video = st.file_uploader("📂 Drop video here to analyze...", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (640, 480))
            annotated_frame, alert = process_frame(frame, model)
            st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        cap.release()
        os.remove(tfile.name)

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
