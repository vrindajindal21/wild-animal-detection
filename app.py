import os
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import cv2
import pygame
import threading
from PIL import Image, ImageTk

# ================= CONFIG =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "yolov8n.pt")
ALERT_SOUND = os.path.join(SCRIPT_DIR, "siren-alert-96052.mp3")
WILD_CLASSES = ["bear", "elephant", "tiger", "lion", "leopard", "wolf", "giraffe", "zebra"]

pygame.mixer.init()

def play_alert():
    if os.path.exists(ALERT_SOUND):
        if not pygame.mixer.get_busy():
            pygame.mixer.music.load(ALERT_SOUND)
            pygame.mixer.music.play()

def safe_label(model, cls_id):
    return model.names.get(cls_id, "").lower()

def draw_alert_yolo(img, results, model):
    """Draw bounding boxes and alerts on frame/image"""
    annotated_img = results[0].plot()
    alert_triggered = False
    detected_this_frame = set()  # Track animals printed this frame

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = safe_label(model, cls_id)
        if label in WILD_CLASSES:
            alert_triggered = True
            x1, y1, _, _ = map(int, box.xyxy[0])
            cv2.putText(annotated_img, "WILD ANIMAL ALERT!", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            if label not in detected_this_frame:
                print("Wild Animal Detected:", label)
                detected_this_frame.add(label)

    return annotated_img, alert_triggered

# Helper to display images/videos in Tkinter window
def show_image_window(img, title="Image"):
    win = tk.Toplevel()
    win.title(title)
    win.geometry("800x600")
    win.resizable(True, True)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    canvas = tk.Canvas(win, width=img_tk.width(), height=img_tk.height())
    canvas.pack(expand=True, fill="both")
    canvas.img_tk = img_tk
    canvas.create_image(0, 0, anchor="nw", image=img_tk)

# ================= IMAGE DETECTION =================
def detect_images():
    try:
        model = YOLO(MODEL_PATH)
        files = filedialog.askopenfilenames(title="Select Images")
        if not files:
            return
        for img_path in files:
            results = model.predict(source=img_path, conf=0.5)
            img = cv2.imread(img_path)
            annotated_img, alert_triggered = draw_alert_yolo(img, results, model)
            if alert_triggered:
                play_alert()
            show_image_window(annotated_img, title=os.path.basename(img_path))
    except Exception as e:
        messagebox.showerror("Error", f"Image detection failed:\n{e}")

# ================= VIDEO DETECTION =================
def detect_videos():
    try:
        model = YOLO(MODEL_PATH)
        files = filedialog.askopenfilenames(title="Select Videos", filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if not files:
            return

        def play_video(video_path):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                messagebox.showerror("Error", f"Cannot open video: {video_path}")
                return

            alert_active = False
            win = tk.Toplevel()
            win.title(f"Video: {os.path.basename(video_path)}")
            win.geometry("800x600")
            win.resizable(True, True)
            canvas = tk.Canvas(win)
            canvas.pack(expand=True, fill="both")

            def update_frame():
                nonlocal alert_active
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    win.destroy()
                    return
                results = model.predict(source=frame.copy(), conf=0.5, verbose=False)
                annotated_img, wild_detected = draw_alert_yolo(frame, results, model)

                if wild_detected:
                    if not alert_active:
                        play_alert()
                        alert_active = True
                        cv2.putText(annotated_img, "WILD ANIMAL ALERT!", (10,50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255),3)
                else:
                    alert_active = False

                img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(img_pil)
                canvas.img_tk = img_tk
                canvas.create_image(0, 0, anchor="nw", image=img_tk)
                win.after(30, update_frame)

            update_frame()
            win.mainloop()

        for video_path in files:
            threading.Thread(target=play_video, args=(video_path,), daemon=True).start()
    except Exception as e:
        messagebox.showerror("Error", f"Video detection failed:\n{e}")

# ================= LIVE DETECTION =================
live_flag = threading.Event()
alert_active_live = False

def start_live(ip_camera_url=None):
    global alert_active_live
    try:
        model = YOLO(MODEL_PATH)
        cap = cv2.VideoCapture(0 if ip_camera_url is None else ip_camera_url)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera")
            return

        live_flag.set()
        win = tk.Toplevel()
        win.title("Live Detection")
        win.geometry("800x600")
        win.resizable(True, True)
        canvas = tk.Canvas(win)
        canvas.pack(expand=True, fill="both")

        def update_live():
            global alert_active_live
            if not live_flag.is_set():
                cap.release()
                win.destroy()
                return
            ret, frame = cap.read()
            if not ret:
                win.after(30, update_live)
                return

            results = model.predict(source=frame.copy(), conf=0.5, verbose=False)
            annotated_img, wild_detected = draw_alert_yolo(frame, results, model)

            if wild_detected:
                if not alert_active_live:
                    play_alert()
                    alert_active_live = True
                    cv2.putText(annotated_img, "WILD ANIMAL ALERT!", (10,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255),3)
            else:
                alert_active_live = False

            img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            canvas.img_tk = img_tk
            canvas.create_image(0, 0, anchor="nw", image=img_tk)
            win.after(30, update_live)

        update_live()
    except Exception as e:
        messagebox.showerror("Error", f"Live detection failed:\n{e}")

def stop_live():
    live_flag.clear()

# ================= GUI =================
root = tk.Tk()
root.title("🐯 Wild Animal Detection System")
root.geometry("600x600")
root.configure(bg="#E3F2FD")

tk.Label(root, text="🐾 Wild Animal Detection System", bg="#E3F2FD", fg="#0D47A1",
         font=("Arial", 18, "bold")).pack(pady=10)

tk.Button(root, text="Upload Images", bg="#1565C0", fg="white", width=25,
          command=lambda: threading.Thread(target=detect_images, daemon=True).start()).pack(pady=5)

tk.Button(root, text="Upload Videos", bg="#43A047", fg="white", width=25,
          command=lambda: threading.Thread(target=detect_videos, daemon=True).start()).pack(pady=5)

tk.Button(root, text="Start Live Detection", bg="orange", fg="white", width=25,
          command=lambda: threading.Thread(target=start_live, daemon=True).start()).pack(pady=5)

tk.Button(root, text="Stop Live Detection", bg="red", fg="white", width=25,
          command=stop_live).pack(pady=5)

tk.Button(root, text="Exit", bg="purple", fg="white", width=25, command=root.quit).pack(pady=5)

root.mainloop()
