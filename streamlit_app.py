
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
    /* 5. History - Yellow */
    [data-testid="stVerticalBlock"] > div:nth-child(7) button { background-color: #FBC02D !important; }
    /* 6. Exit - Purple */
    [data-testid="stVerticalBlock"] > div:nth-child(8) button { background-color: #8E24AA !important; }

    /* Hide the Uploaded File List (The redundant blue bars) */
    [data-testid="stFileUploaderUploadedFileList"] {
        display: none !important;
    }

    header, footer, #MainMenu { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="original-title">🐾 Wild Animal Detection System</div>', unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = "main"
if 'uploader_version' not in st.session_state:
    st.session_state.uploader_version = 0
if 'history' not in st.session_state:
    st.session_state.history = []

# ================= NAVIGATION LOGIC =================
if st.session_state.page == "main":
    v = st.session_state.uploader_version
    # Holders for uploads
    img_holder = st.empty()
    vid_holder = st.empty()
    
    # 1/2. Styled File Uploaders with dynamic keys for clearing
    uploaded_images = img_holder.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=f"img_up_{v}")
    uploaded_video = vid_holder.file_uploader("Upload Videos", type=["mp4", "avi", "mov"], key=f"vid_up_{v}")
    
    if uploaded_images or uploaded_video:
        img_holder.empty()
        vid_holder.empty()
        
        if st.button("⬅️ Close & Back to Menu", key="back_btn"):
            st.session_state.uploader_version += 1
            st.rerun()
            
        st.markdown("---")
        
        if uploaded_images:
            for i, uploaded_file in enumerate(uploaded_images):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                annotated_img, alert = process_frame(image, model)
                
                # Show Result
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
                
                # AUTO-SAVE TO HISTORY if animal detected
                if alert:
                    from datetime import datetime
                    now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    st.session_state.history.append({
                        "image": cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "Auto-Saved Image Detection"
                    })
                    st.toast(f"🚨 Wild Animal Detected! Saved to History.")
                    st.audio(ALERT_SOUND, format="audio/mp3", autoplay=True, key=f"audio_img_{now_str}_{i}")

        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)
            
            # Holders for Video Display
            vid_frame = st.empty()
            audio_placeholder = st.empty()
            has_saved_video_detection = False 
            last_siren_time = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                import time
                curr_time = time.time()
                
                frame = cv2.resize(frame, (640, 480))
                annotated_frame, alert = process_frame(frame, model)
                
                # Update Video Frame
                vid_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                
                # AUTO-SAVE FIRST DETECTION in video
                if alert:
                    # Siren Throttle (Only play every 3.5 seconds)
                    if curr_time - last_siren_time > 3.5:
                        from datetime import datetime
                        now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        audio_placeholder.audio(ALERT_SOUND, format="audio/mp3", autoplay=True, key=f"audio_vid_{now_str}")
                        last_siren_time = curr_time
                        
                    if not has_saved_video_detection:
                        from datetime import datetime
                        st.session_state.history.append({
                            "image": cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "type": "Auto-Saved Video Detection"
                        })
                        st.toast("🚨 First Video Detection Saved to History!")
                        has_saved_video_detection = True

                # Small delay to keep UI responsive
                time.sleep(0.01)

            cap.release()
            os.remove(tfile.name)
            st.success("✅ Video Processing Complete!")
            
    else:
        # Show standard control buttons only when nothing is open
        if st.button("Start Live Detection"):
            st.session_state.page = "live"
            st.rerun()
        if st.button("Stop Live Detection"):
            st.rerun()
        if st.button("View Saved History"):
            st.session_state.page = "history"
            st.rerun()
        if st.button("Exit"):
            st.stop()

# ================= HISTORY PAGE =================
elif st.session_state.page == "history":
    st.markdown('<div class="original-title">📚 Activity History</div>', unsafe_allow_html=True)
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "main"
        st.rerun()

    if not st.session_state.history:
        st.info("No saved results yet. Start scanning to save your alerts!")
    else:
        tab1, tab2 = st.tabs(["📸 Detections", "⚙️ Manage"])
        
        with tab1:
            for item in reversed(st.session_state.history):
                with st.expander(f"🕒 {item['time']} - {item['type']}"):
                    st.image(item['image'], use_container_width=True)
        
        with tab2:
            if st.button("🗑️ Clear All History"):
                st.session_state.history = []
                st.toast("History cleared!")
                st.rerun()

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
        
        # AUTO-SAVE LIVE DETECTION
        if alert:
            from datetime import datetime
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            st.session_state.history.append({
                "image": cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "Auto-Saved Live Detection"
            })
            st.toast("🚨 Live Wild Animal Detected! Saved to History.")
            st.audio(ALERT_SOUND, format="audio/mp3", autoplay=True, key=f"live_{now_str}")

st.markdown("<br><br>", unsafe_allow_html=True)
