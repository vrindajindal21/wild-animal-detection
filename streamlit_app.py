
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

# Custom Color Palette from original UI
# Bg: #E3F2FD, Title Fg: #0D47A1, Buttons: #1565C0, #43A047, orange, red, purple

st.markdown(f"""
<style>
    /* Main Background */
    .stApp {{
        background-color: #E3F2FD;
    }}
    
    /* Header Style */
    .main-title {{
        color: #0D47A1;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        font-size: 32px;
        text-align: center;
        padding: 20px;
    }}
    
    /* Button Styles to match Tkinter */
    div.stButton > button {{
        width: 100%;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 10px;
    }}
    
    /* Button Specific Colors */
    /* Images - Blue */
    [data-testid="stBaseButton-secondary"]:nth-child(1) {{
        background-color: #1565C0;
    }}
    /* Videos - Green */
    [data-testid="stBaseButton-secondary"]:nth-child(2) {{
        background-color: #43A047;
    }}
    /* Live - Orange */
    [data-testid="stBaseButton-secondary"]:nth-child(3) {{
        background-color: orange;
    }}
    /* Stop - Red */
    [data-testid="stBaseButton-secondary"]:nth-child(4) {{
        background-color: red;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🐾 Wild Animal Detection System</div>', unsafe_allow_html=True)
st.markdown("---")

# Load AI Model
model = load_model()

# Session State for navigation
if 'page' not in st.session_state:
    st.session_state.page = "🏠 Home"

# Layout: Column buttons (like the Tkinter menu)
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("Upload Images"):
        st.session_state.page = "📸 Image Detection"
    if st.button("Upload Videos"):
        st.session_state.page = "🎥 Video Detection"
    if st.button("Start Live Detection"):
        st.session_state.page = "🤳 Snapshot Detection"
    if st.button("Stop Analysis"):
        st.session_state.page = "🏠 Home"
    if st.button("Exit"):
        st.stop()

st.markdown("---")

# Conditional Display based on page
if st.session_state.page == "🏠 Home":
    st.image("https://images.unsplash.com/photo-1546182990-dffeafbe841d?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", caption="Wild Forest", use_container_width=True)
    st.write("""
    ### Detected Animals:
    - 🐯 Tiger, 🦁 Lion, 🐆 Leopard
    - 🐘 Elephant, 🦒 Giraffe, 🦓 Zebra
    - 🐻 Bear, 🐺 Wolf
    """)

elif st.session_state.page == "📸 Image Detection":
    st.header("Upload Image for Analysis")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
            
        with st.spinner("🔍 AI is analyzing..."):
            annotated_img, alert = process_frame(image, model)
            
        with col2:
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Detection Results", use_container_width=True)
        
        if alert:
            st.error("🚨 **WILD ANIMAL ALERT!** 🚨")
            st.audio(ALERT_SOUND, format="audio/mp3", autoplay=True)
        else:
            st.success("✅ Area appears to be safe.")

elif st.session_state.page == "🎥 Video Detection":
    st.header("Video Stream Analysis")
    uploaded_video = st.file_uploader("Upload a video clip", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        
        if st.button("Run Model on Video"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.resize(frame, (640, 480))
                annotated_frame, alert = process_frame(frame, model)
                
                st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                
                if alert:
                    st.warning("🚨 **WILD ANIMAL SPOTTED!**")
            
            cap.release()
            os.remove(tfile.name)
            st.success("Video analysis completed.")

elif st.session_state.page == "🤳 Snapshot Detection":
    st.header("Real-time Snapshot")
    img_file_buffer = st.camera_input("Snapshot for environment analysis")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        with st.spinner("Analyzing snapshot..."):
            annotated_img, alert = process_frame(cv2_img, model)
            
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Snapshot Analysis")
        
        if alert:
            st.error("🚨 **DANGER!** Wild animal detected.")
            st.audio(ALERT_SOUND, format="audio/mp3", autoplay=True)
        else:
            st.success("✅ Area Clear.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & YOLOv8 (Original UI Mirror)")
