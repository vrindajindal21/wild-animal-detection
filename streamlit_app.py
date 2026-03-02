
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
st.set_page_config(
    page_title="Wild Animal Detection System", 
    page_icon="🐯", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ================= MAIN APP =================
st.set_page_config(
    page_title="Wild Animal Detection System", 
    page_icon="🐯", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load AI Model
model = load_model()

# Premium Ultra-Modern Nature CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        font-family: 'Outfit', sans-serif !important;
        color: white;
    }

    /* Container Styling */
    [data-testid="stVerticalBlock"] > div:first-child {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 24px;
        padding: 40px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-top: 20px;
    }

    /* Header */
    .premium-header {
        background: linear-gradient(to right, #ffd452 0%, #f76b1c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }
    .sub-caption {
        color: rgba(255, 255, 255, 0.7);
        text-align: center;
        font-size: 16px;
        margin-bottom: 40px;
        font-weight: 300;
    }

    /* Button Overrides */
    div.stButton > button {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        height: 3.5em !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 15px !important;
        width: 100% !important;
    }

    div.stButton > button:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 0 20px rgba(255, 212, 82, 0.2) !important;
        transform: translateY(-2px) !important;
    }

    /* Button Specific Gradients for "Better" UX */
    div[data-testid="stVerticalBlock"] > div:nth-child(1) [data-testid="stVerticalBlock"] > div:nth-child(1) button {
        border-left: 5px solid #1565C0 !important;
    }
    div[data-testid="stVerticalBlock"] > div:nth-child(1) [data-testid="stVerticalBlock"] > div:nth-child(2) button {
        border-left: 5px solid #43A047 !important;
    }
    div[data-testid="stVerticalBlock"] > div:nth-child(1) [data-testid="stVerticalBlock"] > div:nth-child(3) button {
        border-left: 5px solid #ffa500 !important;
    }
    div[data-testid="stVerticalBlock"] > div:nth-child(1) [data-testid="stVerticalBlock"] > div:nth-child(4) button {
        border-left: 5px solid #f44336 !important;
    }
    div[data-testid="stVerticalBlock"] > div:nth-child(1) [data-testid="stVerticalBlock"] > div:nth-child(5) button {
        border-left: 5px solid #9c27b0 !important;
    }

    /* Inputs and Uploaders */
    section[data-testid="stFileUploader"] {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 12px;
        padding: 20px;
        border: 1px dashed rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Centered Premium Header
st.markdown('<div class="premium-header">� WILD ANIMAL SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-caption">AI-Powered Wildlife Surveillance & Safety Hub</div>', unsafe_allow_html=True)

# Session State for navigation
if 'page' not in st.session_state:
    st.session_state.page = "main"

# Menu Section
c1, main_card, c3 = st.columns([0.1, 0.8, 0.1])

with main_card:
    if st.button("📸 UPLOAD IMAGES"):
        st.session_state.page = "images"
    if st.button("📹 UPLOAD VIDEOS"):
        st.session_state.page = "videos"
    if st.button("📡 START LIVE SCAN"):
        st.session_state.page = "live"
    if st.button("🛑 STOP ANALYSIS"):
        st.session_state.page = "main"
        st.rerun()
    if st.button("❌ EXIT SYSTEM"):
        st.stop()

st.markdown("<br>", unsafe_allow_html=True)

# Execution Modules (Displayed below the menu)
if st.session_state.page == "images":
    st.subheader("�️ Image Detection Mode")
    uploaded_files = st.file_uploader("Select Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                annotated_img, alert = process_frame(image, model)
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption=uploaded_file.name)
                
                if alert:
                    st.error(f"🚨 WILD ANIMAL ALERT in {uploaded_file.name}!")
                    st.audio(ALERT_SOUND, format="audio/mp3", autoplay=True)

elif st.session_state.page == "videos":
    st.subheader("📹 Video Detection Mode")
    uploaded_video = st.file_uploader("Select Videos", type=["mp4", "avi", "mov"])
    
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
            if alert:
                st.warning("🚨 WILD ANIMAL DETECTED!")
        
        cap.release()
        os.remove(tfile.name)

elif st.session_state.page == "live":
    st.subheader("📺 Live Detection Mode")
    img_file_buffer = st.camera_input("Camera Feed")

    if img_file_buffer:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        annotated_img, alert = process_frame(cv2_img, model)
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        if alert:
            st.error("🚨 DANGER! Wild animal detected.")
            st.audio(ALERT_SOUND, format="audio/mp3", autoplay=True)

st.markdown("---")
st.caption("E:\\wild_animal_detection (Mirrored from Tkinter GUI)")
