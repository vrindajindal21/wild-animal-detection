# 🐯 Wild Animal Detection System

A AI-powered web application built with **Streamlit** and **YOLOv8** for real-time wild animal detection in images, videos, and snapshots.

## 🚀 Features
- **🏠 Home Page**: Overview of the system.
- **📸 Image Detection**: Upload and analyze images for wild animals.
- **🎥 Video Detection**: Analyze video streams for dangerous wildlife.
- **🤳 Snapshot Detection**: Use your local camera to take snapshots & detect risk.
- **🚨 Audio Alert**: Siren sound triggers when an animal is detected.

## 🛠️ Tech Stack
- **Framework**: Streamlit
- **AI Model**: YOLOv8 (Ultralytics)
- **Image Processing**: OpenCV, Pillow
- **Programming Language**: Python

## 📦 How to Run Locally
1. Clone this repository (if hosted) or open the directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

## 🌐 Deployment Instructions (Streamlit Cloud)
1. **GitHub**: Push this directory to a new repository on GitHub.
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Wild Animal Detection System"
   git branch -M main
   # Add your remote URL
   git remote add origin https://github.com/YOUR_USERNAME/wild-animal-detection.git
   git push -u origin main
   ```
2. **Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/).
   - Connect your GitHub account.
   - Select the `wild-animal-detection` repository.
   - Set the Main file path to `streamlit_app.py`.
   - Click **Deploy**!

## 📁 Project Structure
- `streamlit_app.py`: The main web application.
- `app.py`: The original desktop (Tkinter) version.
- `yolov8n.pt`: Prematrained YOLOv8 model.
- `siren-alert-96052.mp3`: The alert siren sound.
- `requirements.txt`: List of Python dependencies.

---
Created by Antigravity AI助手
