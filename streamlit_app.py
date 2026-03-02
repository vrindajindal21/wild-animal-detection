
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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #1a73e8;
        color: white;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🐯 Wild Animal Detection System")
st.markdown("---")

st.sidebar.image("https://img.icons8.com/clouds/100/tiger.png", width=100)
st.sidebar.header("Navigation")
menu = ["🏠 Home", "📸 Image Detection", "🎥 Video Detection", "🤳 Snapshot Detection"]
choice = st.sidebar.radio("Go to", menu)

model = load_model()

if choice == "🏠 Home":
    st.subheader("Welcome to the Wildlife Safety System")
    st.image("https://images.unsplash.com/photo-1546182990-dffeafbe841d?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", caption="Wild Forest", use_container_width=True)
    st.write("""
    This application uses state-of-the-art **YOLOv8** deep learning model to detect dangerous wild animals in images, videos, or snapshots.
    
    ### Detected Animals:
    - 🐯 Tiger, 🦁 Lion, 🐆 Leopard
    - 🐘 Elephant, 🦒 Giraffe, 🦓 Zebra
    - 🐻 Bear, 🐺 Wolf
    """)
    st.info("💡 **How it works:** Select a mode from the sidebar, upload your file, and the AI will analyze it in real-time. If it detects a dangerous animal, an alert will sound!")

elif choice == "📸 Image Detection":
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

elif choice == "🎥 Video Detection":
    st.header("Video Stream Analysis")
    uploaded_video = st.file_uploader("Upload a video clip", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        
        if st.button("Start Video Analysis"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resizing frame for faster processing in browsers
                frame = cv2.resize(frame, (640, 480))
                annotated_frame, alert = process_frame(frame, model)
                
                st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                
                if alert:
                    st.warning("🚨 **WILD ANIMAL SPOTTED!**")
            
            cap.release()
            os.remove(tfile.name)
            st.success("Video analysis completed.")

elif choice == "🤳 Snapshot Detection":
    st.header("Real-time Snapshot")
    img_file_buffer = st.camera_input("Smile & Stay Safe - Take a snapshot of your environment")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        with st.spinner("Analyzing snapshot..."):
            annotated_img, alert = process_frame(cv2_img, model)
            
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Snapshot Analysis")
        
        if alert:
            st.error("🚨 **DANGER!** Wild animal detected near your camera.")
            st.audio(ALERT_SOUND, format="audio/mp3", autoplay=True)
        else:
            st.success("✅ Snapshot indicates no immediate danger.")

st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit & YOLOv8")
