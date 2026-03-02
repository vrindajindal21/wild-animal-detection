
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

# Load AI Model
model = load_model()

# Refined CSS for exact Tkinter Match
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #E3F2FD;
    }
    
    /* Header Style */
    .header-text {
        color: #0D47A1;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        font-size: 28px;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Center the menu buttons */
    .stButton > button {
        width: 100% !important;
        height: 2.5em;
        font-weight: bold;
        color: white !important;
        border-radius: 4px;
        border: 1px solid rgba(0,0,0,0.1);
        margin-bottom: 10px;
        transition: transform 0.1s;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
    }

    /* Button Specific Colors - Targeted by index */
    /* Images - Blue */
    div[data-testid="stVerticalBlock"] > div:nth-child(1) [data-testid="stVerticalBlock"] > div:nth-child(1) button { background-color: #1565C0 !important; }
    /* Videos - Green */
    div[data-testid="stVerticalBlock"] > div:nth-child(1) [data-testid="stVerticalBlock"] > div:nth-child(2) button { background-color: #43A047 !important; }
    /* Live - Orange */
    div[data-testid="stVerticalBlock"] > div:nth-child(1) [data-testid="stVerticalBlock"] > div:nth-child(3) button { background-color: orange !important; }
    /* Stop - Red */
    div[data-testid="stVerticalBlock"] > div:nth-child(1) [data-testid="stVerticalBlock"] > div:nth-child(4) button { background-color: red !important; }
    /* Exit - Purple */
    div[data-testid="stVerticalBlock"] > div:nth-child(1) [data-testid="stVerticalBlock"] > div:nth-child(5) button { background-color: purple !important; }

    /* Hover effects */
    .stButton > button:hover {
        opacity: 0.9 !important;
        border: 1px solid white !important;
    }
</style>
""", unsafe_allow_html=True)

# Centered Title
st.markdown('<div class="header-text">🐾 Wild Animal Detection System</div>', unsafe_allow_html=True)

# Session State for navigation
if 'page' not in st.session_state:
    st.session_state.page = "main"

# Center Column for the Menu
c1, menu_col, c3 = st.columns([1, 2, 1])

with menu_col:
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

st.markdown("---")

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
