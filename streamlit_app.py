
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import tempfile

import base64

# ================= CONFIG =================
MODEL_PATH = "yolov8n.pt"
ALERT_SOUND = "siren-alert-96052.mp3"
WILD_CLASSES = ["bear", "elephant", "tiger", "lion", "leopard", "wolf", "giraffe", "zebra"]

# ================= HELPER FUNCTIONS =================
def play_siren():
    """Bypasses buggy st.audio using HTML/Base64 directly"""
    if os.path.exists(ALERT_SOUND):
        with open(ALERT_SOUND, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio autoplay="true" style="display:none;">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(md, unsafe_allow_html=True)

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
            # Add the RED ALERT text (Mirroring Tkinter GUI)
            cv2.putText(annotated_frame, "WILD ANIMAL ALERT!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
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
    [data-testid="stVerticalBlock"] > div:nth-child(1) [data-testid="stFileUploader"] button::after {
        content: "Upload Images" !important;
        font-size: 16px !important;
        color: white !important;
        font-weight: bold !important;
    }
    [data-testid="stVerticalBlock"] > div:nth-child(2) [data-testid="stFileUploader"] button::after {
        content: "Upload Videos" !important;
        font-size: 16px !important;
        color: white !important;
        font-weight: bold !important;
    }

    /* Assign Colors using highly specific paths */
    /* Images - Blue */
    [data-testid="stVerticalBlock"] > div:nth-child(1) [data-testid="stFileUploader"] button { background-color: #1976D2 !important; }
    /* Videos - Green */
    [data-testid="stVerticalBlock"] > div:nth-child(2) [data-testid="stFileUploader"] button { background-color: #4CAF50 !important; }
    /* Start - Orange */
    [data-testid="stVerticalBlock"] > div:nth-child(3) button { background-color: #FF9800 !important; }
    /* Stop - Red */
    [data-testid="stVerticalBlock"] > div:nth-child(4) button { background-color: #F44336 !important; }
    /* History - Yellow */
    [data-testid="stVerticalBlock"] > div:nth-child(5) button { background-color: #FBC02D !important; }
    /* Exit - Purple */
    [data-testid="stVerticalBlock"] > div:nth-child(6) button { background-color: #8E24AA !important; }

    /* Results Header Styling */
    .results-banner {
        background: #0D47A1;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    /* Hide the Uploaded File List */
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
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

# ================= NAVIGATION LOGIC =================
if st.session_state.page == "main":
    v = st.session_state.uploader_version
    
    # We use a container to ensure indices match for CSS nth-child styling
    with st.container():
        uploaded_images = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=f"img_up_{v}")
        uploaded_video = st.file_uploader("Upload Videos", type=["mp4", "avi", "mov"], key=f"vid_up_{v}")
        
        # Only show Main Menu buttons if NO files are currently being processed
        if not (uploaded_images or uploaded_video):
            # 1. Start Clicked
            if st.button("Start Live Detection"):
                st.session_state.page = "live"
                st.rerun()
            # 2. Stop Clicked
            if st.button("Stop Live Detection"):
                st.rerun()
            # 3. View History Clicked
            if st.button("View Saved History"):
                st.session_state.page = "history"
                st.rerun()
            # 4. Exit Clicked
            if st.button("Exit"):
                st.stop()
        else:
            # If files ARE uploaded, show the "Back" button at the top of results
            if st.button("⬅️ Finish & Return to Main Menu", key="back_btn"):
                st.session_state.uploader_version += 1
                st.session_state.processed_files = set() 
                st.rerun()
    
    if uploaded_images or uploaded_video:
        st.markdown('<div class="results-banner">🔍 SCANNING ACTIVE... SEE RESULTS BELOW</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        if uploaded_images:
            for i, uploaded_file in enumerate(uploaded_images):
                # Use filename + size as a unique ID to avoid duplicate saves on rerun
                file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                annotated_img, alert = process_frame(image, model)
                
                # Show Result in a labeled container
                with st.container(border=True):
                    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
                    if alert:
                        st.error("🚨 WILD ANIMAL DETECTED!")
                    else:
                        st.success("✅ No Dangerous Wildlife Detected")
                
                # SAVE TO HISTORY (Every Image Uploaded)
                if file_id not in st.session_state.processed_files:
                    from datetime import datetime
                    detection_type = "🚨 Image Detection" if alert else "✅ Image Scan (Safe)"
                    st.session_state.history.append({
                        "image": cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": detection_type
                    })
                    st.session_state.processed_files.add(file_id)
                    
                    if alert:
                        st.toast(f"🚨 Wild Animal Detected! Saved to History.")
                        play_siren()
                    else:
                        st.toast(f"📸 Image Saved to History.")

        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)
            
            # Holders for Video Display
            vid_frame = st.empty()
            siren_holder = st.empty() 
            last_save_time = 0
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
                
                # AUTO-SAVE TO HISTORY if animal detected
                if alert:
                    # Siren Cooldown: 10 seconds (avoids infinite/annoying overlap)
                    if curr_time - last_siren_time > 10.0:
                        with siren_holder:
                            play_siren()
                        last_siren_time = curr_time
                    
                    # History Save Cooldown: 2 seconds (Ensures 'Save All' major events)
                    if curr_time - last_save_time > 2.0:
                        from datetime import datetime
                        st.session_state.history.append({
                            "image": cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "type": "Video Detection"
                        })
                        # Memory Cap
                        if len(st.session_state.history) > 100: st.session_state.history.pop(0)
                        st.toast("🚨 Animal Saved to History!")
                        last_save_time = curr_time

                # Small delay to keep UI responsive
                time.sleep(0.01)

            cap.release()
            os.remove(tfile.name)
            st.success("✅ Video Processing Complete!")
            
        # --- ROBUST AUTO-SCROLL (FORCED) ---
        st.markdown("""
            <script>
                setTimeout(function() {
                    // Method 1: Scroll parent container
                    window.parent.postMessage({type: 'streamlit:scroll_to_bottom'}, '*');
                    
                    // Method 2: Direct scroll if permissions allow
                    try {
                        window.parent.window.scrollTo({
                            top: 1000, 
                            behavior: 'smooth'
                        });
                    } catch (e) {
                        // Method 3: Internal iframe scroll as fallback
                        window.scrollTo(0, document.body.scrollHeight);
                    }
                }, 500); 
            </script>
            """, unsafe_allow_html=True)
            
    pass

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
    st.markdown('<div class="original-title">📡 Live Mode - Camera Capture</div>', unsafe_allow_html=True)
    st.info("💡 Take a photo below to analyze the live feed for wild animals.")
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "main"
        st.rerun()

    img_file_buffer = st.camera_input("Scanner Active")
    if img_file_buffer:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        annotated_img, alert = process_frame(cv2_img, model)
        
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        
        # AUTO-SAVE LIVE DETECTION (Happens every time picture is snapped)
        if alert:
            from datetime import datetime
            st.session_state.history.append({
                "image": cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "Live Detection"
            })
            if len(st.session_state.history) > 100: st.session_state.history.pop(0)
            st.toast("🚨 Detected Animal Saved to History!")
            play_siren()

st.markdown("<br><br>", unsafe_allow_html=True)
