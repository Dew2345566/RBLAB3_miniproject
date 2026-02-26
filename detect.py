from flask import Flask, Response
import cv2
import time
import os
import requests
import threading
from ultralytics import YOLO
from playsound import playsound

# Initialize the web server
app = Flask(__name__)

# ====== CONFIG ======
BOT_TOKEN = "8789739831:AAGsqTdYni6MDqBR_kj1YOAPKKoMpyv3MRk" # Get a fresh token from BotFather!
CHAT_ID = "7177779556"
CONF_THRESHOLD = 0.5
TARGET_CLASSES = ["Helmet-", "NO-Helmet"]
ALERT_SOUND_PATH = "alert-ascending-chime-betacut-1-00-02.mp3"
COOLDOWN_SECONDS = 15
# ====================

model = YOLO("best1.pt")

# Open webcam and OPTIMIZE FOR JETSON NANO (keeps it from lagging)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_alert_time = 0

def send_alert_background(frame_to_save):
    """Runs in the background so the video doesn't stutter"""
    print("🚨 Safety violation detected! Sending alert...")
    cv2.imwrite("detected.jpg", frame_to_save)
    
    caption = "⚠️ No helmet detected! Please be careful.\n\nSafety first! 🚧"
    
    if os.path.exists(ALERT_SOUND_PATH):
        try:
            playsound(ALERT_SOUND_PATH)
        except Exception as e:
            print("Sound error:", e)

    url = "https://api.telegram.org/bot{}/sendPhoto".format(BOT_TOKEN)
    data = {"chat_id": CHAT_ID, "caption": caption}
    try:
        with open("detected.jpg", "rb") as photo_file:
            requests.post(url, data=data, files={"photo": photo_file})
        print("📨 Alert sent successfully!")
    except Exception as e:
        print("❌ Failed to send Telegram alert.")

def generate_frames():
    global last_alert_time
    prev_time = time.time() # Used to calculate FPS
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Calculate FPS
        current_time_fps = time.time()
        fps = 1 / (current_time_fps - prev_time)
        prev_time = current_time_fps

        # Run YOLO inference
        results = model(frame, verbose=False)[0]
        found_no_helmet = False

        for box in results.boxes:
            conf = float(box.conf)
            cls = int(box.cls)
            label = model.names[cls]

            if conf > CONF_THRESHOLD and label in TARGET_CLASSES:
                if label == "Helmet-":
                    box_color = (0, 255, 0) # Green
                elif label == "NO-Helmet":
                    found_no_helmet = True
                    box_color = (0, 0, 255) # Red
                else:
                    continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                label_text = "{} {:.2f}".format(label, conf)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Draw the FPS counter on the frame (top-left corner)
        fps_text = "FPS: {:.1f}".format(fps)
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Trigger alert logic
        current_time = time.time()
        if found_no_helmet and (current_time - last_alert_time > COOLDOWN_SECONDS):
            last_alert_time = current_time
            threading.Thread(target=send_alert_background, args=(frame.copy(),)).start()

        # Convert the frame to JPEG format for the web browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame to the web server
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- WEB ROUTES ---
@app.route('/')
def index():
    # Upgraded, modern UI for the webpage
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Jetson AI Safety Monitor</title>
        <style>
            body { 
                background-color: #0f172a; 
                color: #f8fafc; 
                text-align: center; 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0;
                padding: 20px;
            }
            h1 { color: #38bdf8; margin-bottom: 5px; }
            p { color: #94a3b8; margin-top: 0; margin-bottom: 20px; font-size: 14px;}
            .video-container {
                display: inline-block;
                padding: 10px;
                background: #1e293b;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            }
            img { 
                width: 100%; 
                max-width: 800px; 
                border: 2px solid #334155; 
                border-radius: 8px; 
            }
            .status {
                display: inline-block;
                margin-top: 15px;
                padding: 8px 16px;
                background-color: #064e3b;
                color: #34d399;
                border-radius: 20px;
                font-weight: bold;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <h1>Helmet Detection AI</h1>
        <p>Live from Jetson Nano</p>
        
        <div class="video-container">
            <img src="/video_feed" alt="Live Video Feed Loading...">
        </div>
        <br>
        <div class="status">🟢 System Active & Monitoring</div>
    </body>
    </html>
    """
    return html

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("🌐 Web Server is LIVE!")
    print("👉 Look at your Jetson's IP address on port 5000 in your browser.")
    app.run(host='0.0.0.0', port=5000, threaded=True)