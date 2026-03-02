
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

# ================= CSS: IMPROVED TKINTER MIRROR =================
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #E3F2FD !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }

    /* Fix for White Buttons in Screenshot */
    div.stButton > button {
        width: 100% !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15) !important;
        filter: brightness(1.1);
    }

    /* Assigning Colors using specialized selectors */
    /* Images - Blue */
    div[data-testid="stVerticalBlock"] > div:nth-child(1) [data-testid="stVerticalBlock"] > div:nth-child(1) button,
    div.stButton button:contains("Upload Images") { background-color: #1565C0 !important; }
    
    /* Videos - Green */
    div[data-testid="stVerticalBlock"] > div:nth-child(1) [data-testid="stVerticalBlock"] > div:nth-child(2) button,
    div.stButton button:contains("Upload Videos") { background-color: #43A047 !important; }
    
    /* Live - Orange */
    div[data-testid="stVerticalBlock"] > div:nth-child(1) [data-testid="stVerticalBlock"] > div:nth-child(3) button,
    div.stButton button:contains("Start Live") { background-color: #FF9800 !important; }
    
    /* Stop - Red */
    div[data-testid="stVerticalBlock"] > div:nth-child(1) [data-testid="stVerticalBlock"] > div:nth-child(4) button,
    div.stButton button:contains("Stop Live") { background-color: #D32F2F !important; }
    
    /* Exit - Purple */
    div[data-testid="stVerticalBlock"] > div:nth-child(1) [data-testid="stVerticalBlock"] > div:nth-child(5) button,
    div.stButton button:contains("Exit") { background-color: #7B1FA2 !important; }

    /* Title Styling */
    .tk-label {
        color: #0D47A1;
        font-weight: 800;
        font-size: 32px;
        text-align: center;
        margin-bottom: 40px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }

    /* Clean up headers */
    header { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Centered Title (Tkinter Label)
st.markdown('<div class="tk-label">🐾 Wild Animal Detection System</div>', unsafe_allow_html=True)

# Session State for navigation
if 'page' not in st.session_state:
    st.session_state.page = "main"

# Menu Section (Centered via columns to limit width like Tkinter's width=25)
mc1, menu_col, mc3 = st.columns([1, 1.2, 1])

with menu_col:
    # We use st.button but the CSS above will force the colors
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

# Execution Modules
if st.session_state.page == "images":
    st.markdown('<div style="color:#0D47A1; font-weight:bold;">Select Images</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True, label_visibility="collapsed")
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            annotated_img, alert = process_frame(image, model)
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption=uploaded_file.name)
            if alert:
                st.audio(ALERT_SOUND, format="audio/mp3", autoplay=True)

elif st.session_state.page == "videos":
    st.markdown('<div style="color:#0D47A1; font-weight:bold;">Select Videos</div>', unsafe_allow_html=True)
    uploaded_video = st.file_uploader("", type=["mp4", "avi", "mov"], label_visibility="collapsed")
    
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
    st.markdown('<div style="color:#0D47A1; font-weight:bold;">Live Detection</div>', unsafe_allow_html=True)
    img_file_buffer = st.camera_input("")
    if img_file_buffer:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        annotated_img, alert = process_frame(cv2_img, model)
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        if alert:
            st.audio(ALERT_SOUND, format="audio/mp3", autoplay=True)

st.markdown("<br><br>", unsafe_allow_html=True)
