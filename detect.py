from ultralytics import YOLO
import cv2
import requests
import os
import time
from playsound import playsound 

# ====== CONFIG ======
BOT_TOKEN = "8789739831:AAGsqTdYni6MDqBR_kj1YOAPKKoMpyv3MRk" # Put your newly generated token here!
CHAT_ID = "7177779556"
CONF_THRESHOLD = 0.5

TARGET_CLASSES = ["Helmet-", "NO-Helmet"] 
ALERT_SOUND_PATH = "alert-ascending-chime-betacut-1-00-02.mp3"  

# New Config: Cooldown time to prevent Telegram spam (in seconds)
COOLDOWN_SECONDS = 15 
# ====================

# Load YOLO model 
model = YOLO("best1.pt")

# Open the default webcam (0 is usually the built-in camera)
cap = cv2.VideoCapture(0)

# Variables to keep track of time for the cooldown system
last_alert_time = 0 

print("Starting webcam feed... Press 'q' on your keyboard to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame from webcam. Is another app using it?")
        break

    results = model(frame, verbose=False)[0]

    found_helmet = False
    found_no_helmet = False

    for box in results.boxes:
        conf = float(box.conf)
        cls = int(box.cls)
        label = model.names[cls]

        if conf > CONF_THRESHOLD and label in TARGET_CLASSES:
            if label == "Helmet-":
                found_helmet = True
                box_color = (0, 255, 0)
            elif label == "NO-Helmet":
                found_no_helmet = True
                box_color = (0, 0, 255)
            else:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # ==== ALERT LOGIC ====
    current_time = time.time()
    
    if found_no_helmet:
        if current_time - last_alert_time > COOLDOWN_SECONDS:
            print("[ALERT] Safety violation detected! Sending alert...")
            
            cv2.imwrite("detected.jpg", frame)
            
            caption = (
                "[ALERT] No helmet detected! Please be careful.\n\n"
                "Safety first!\n\n"
                "Please make sure to wear your helmet at all times. Stay safe!"
            )

            if os.path.exists(ALERT_SOUND_PATH):
                try:
                    playsound(ALERT_SOUND_PATH)  
                except Exception as e:
                    print(f"Error playing sound: {e}")

            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
            data = {"chat_id": CHAT_ID, "caption": caption}
            
            with open("detected.jpg", "rb") as photo_file:
                requests.post(url, data=data, files={"photo": photo_file})
            
            print("Alert sent! Starting cooldown...")
            last_alert_time = current_time 
            
    cv2.imshow("Real-Time Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting program...")
        break

cap.release()
cv2.destroyAllWindows()