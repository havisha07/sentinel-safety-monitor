import cv2
import numpy as np
import csv
from ultralytics import YOLO
from datetime import datetime

# 1. Load the pretrained YOLO model (it will auto-download yolov8n.pt ~6MB)
model = YOLO("yolov8n.pt")

# 2. Define Restricted Zone (Adjust these coordinates later if needed)
ZONE_POLYGON = np.array([
    [200, 300], 
    [600, 300], 
    [700, 650], 
    [100, 650]
], np.int32)

ALERT_THRESHOLD_FRAMES = 5
intrusion_tracker = {}
event_logs = []

def log_event(track_id, obj_class, timestamp):
    event = {
        "Timestamp": timestamp,
        "Track_ID": track_id,
        "Object_Class": obj_class,
        "Event": "INTRUSION"
    }
    event_logs.append(event)
    print(f"[ALERT] {timestamp} - {obj_class} (ID: {track_id}) entered the restricted zone!")

def save_logs_to_csv():
    if event_logs:
        keys = event_logs[0].keys()
        with open("intrusion_log.csv", "w", newline="") as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(event_logs)
        print("Event logs saved to 'intrusion_log.csv'")

# 3. Load video
cap = cv2.VideoCapture("test_video.mp4") 

if not cap.isOpened():
    print("ERROR: Could not open test_video.mp4. Did you download it and put it in this exact folder?")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = str(datetime.now().strftime("%H:%M:%S"))

    # Run detection & tracking
    results = model.track(frame, persist=True, classes=[0, 2, 7], verbose=False)

    # Draw the Restricted Zone
    cv2.polylines(frame, [ZONE_POLYGON], True, (0, 255, 255), 2)
    cv2.putText(frame, "RESTRICTED ZONE", (150, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, ids, clss):
            x_min, y_min, x_max, y_max = map(int, box)
            class_name = model.names[cls]

            # Anchor Point: Bottom-Center (feet)
            anchor_x = int((x_min + x_max) / 2)
            anchor_y = int(y_max)
            cv2.circle(frame, (anchor_x, anchor_y), 5, (255, 0, 0), -1)

            # Spatial Logic: Is the anchor inside the polygon?
            is_inside = cv2.pointPolygonTest(ZONE_POLYGON, (anchor_x, anchor_y), False) >= 0

            # Temporal Logic: Count frames inside
            if is_inside:
                intrusion_tracker[track_id] = intrusion_tracker.get(track_id, 0) + 1
            else:
                intrusion_tracker[track_id] = 0

            # Trigger Event
            if intrusion_tracker[track_id] >= ALERT_THRESHOLD_FRAMES:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
                cv2.putText(frame, f"ALERT: {class_name} {track_id}", (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Privacy Safeguard: Blur face for Person class (0)
                if cls == 0: 
                    face_y_max = y_min + int((y_max - y_min) * 0.3)
                    face_y_max = min(face_y_max, frame.shape[0])
                    face_region = frame[y_min:face_y_max, x_min:x_max]
                    if face_region.size > 0:
                        frame[y_min:face_y_max, x_min:x_max] = cv2.GaussianBlur(face_region, (99, 99), 30)

                if intrusion_tracker[track_id] == ALERT_THRESHOLD_FRAMES:
                    log_event(track_id, class_name, timestamp)
            else:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {track_id}", (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Industrial Safety Zone Intrusion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
save_logs_to_csv()