import cv2
import numpy as np
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
from datetime import datetime
import winsound
import os
import time
import wave
import json
import csv
import traceback

# ==========================================
# THEME / COLOR PALETTE
# ==========================================
THEME = {
    "bg":           "#0a0a0a",
    "panel":        "#1a1a1a",
    "panel_alt":    "#141414",
    "border":       "#2d2d2d",
    "border_accent":"#8b0000",
    "header":       "#c0392b",
    "text":         "#d9d9d9",
    "text_muted":   "#808080",
    "safe":         "#00cc44",
    "warning":      "#ff8c00",
    "critical":     "#ff2222",
    "loitering":    "#b300ff",
    "btn_start":    "#1e8449",
    "btn_stop":     "#8b0000",
    "btn_zone":     "#34495e",
    "btn_sound":    "#6c3483",
    "btn_export":   "#b9770e",
}

# ==========================================
# TARGET CLASSES
# ==========================================
CLASS_OPTIONS = {
    0:  {"name": "Person",     "bgr": (0, 255, 0),   "hex": "#00ff00"},
    67: {"name": "Cell Phone", "bgr": (0, 0, 255),   "hex": "#ff0000"},
    63: {"name": "Laptop",     "bgr": (255, 128, 0), "hex": "#ff8000"},
}

enabled_classes = list(CLASS_OPTIONS.keys())
class_intrusions = {cid: 0 for cid in CLASS_OPTIONS.keys()}

# ==========================================
# CONFIGURATION
# ==========================================
ZONE_FILE = "zone.json"
ZONE_POLYGON = None 
ALERT_THRESHOLD_FRAMES = 5
CRITICAL_DWELL_SECONDS = 10
LOITERING_DWELL_SECONDS = 30
CRITICAL_BEEP_INTERVAL = 40
VIDEO_WIDTH = 700   # Reduced video size to fit everything on screen
VIDEO_HEIGHT = 500

intrusion_tracker = {}
track_entry_time = {}
track_severity = {}
is_running = False
temp_zone_points = []
drawing_zone = False
sound_enabled = True

total_intrusions = 0
active_trackers = 0
current_fps = 0.0
event_records = []

if not os.path.exists("evidence"):
    os.makedirs("evidence")

frame_queue = queue.Queue(maxsize=2)
log_queue = queue.Queue()
status_queue = queue.Queue()
metrics_queue = queue.Queue()
thumbnail_queue = queue.Queue()

# ==========================================
# ZONE PERSISTENCE
# ==========================================
def save_zone_to_file(zone_array):
    try:
        with open(ZONE_FILE, "w") as f:
            json.dump(zone_array.tolist(), f, indent=2)
    except Exception as e:
        print(f"[ZONE] Save error: {e}")

def load_zone_from_file():
    global ZONE_POLYGON
    if os.path.exists(ZONE_FILE):
        try:
            with open(ZONE_FILE, "r") as f:
                ZONE_POLYGON = np.array(json.load(f), np.int32)
            print(f"[ZONE] Loaded.")
        except Exception as e:
            print(f"[ZONE] Load error: {e}")

# ==========================================
# ALARM SOUND GENERATOR
# ==========================================
WARNING_FILE = "warning.wav"
CRITICAL_FILE = "critical.wav"

def generate_alarm_sounds():
    sample_rate = 44100
    duration = 0.35
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, False)
    tone = np.sign(np.sin(1000 * 2 * np.pi * t)) * 0.9
    fade = int(sample_rate * 0.01)
    tone[:fade] *= np.linspace(0, 1, fade)
    tone[-fade:] *= np.linspace(1, 0, fade)
    audio = (tone * 32767).astype(np.int16)
    with wave.open(WARNING_FILE, 'w') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate)
        w.writeframes(audio.tobytes())
    segments = []
    for freq in [1800, 800]:
        d = 0.25
        n = int(sample_rate * d)
        t = np.linspace(0, d, n, False)
        tone = np.sign(np.sin(freq * 2 * np.pi * t)) * 0.85
        fade = int(sample_rate * 0.01)
        tone[:fade] *= np.linspace(0, 1, fade)
        tone[-fade:] *= np.linspace(1, 0, fade)
        segments.append(tone)
    audio = np.concatenate(segments)
    audio = (audio * 32767).astype(np.int16)
    with wave.open(CRITICAL_FILE, 'w') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate)
        w.writeframes(audio.tobytes())
    print("[AUDIO] Alarm files generated.")

if not os.path.exists(WARNING_FILE) or not os.path.exists(CRITICAL_FILE):
    generate_alarm_sounds()

# ==========================================
# NON-BLOCKING AUDIO
# ==========================================
def play_sound_async(path):
    if not sound_enabled:
        return
    try:
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
    except Exception as e:
        print(f"[AUDIO] Error: {e}")

def play_warning(): play_sound_async(WARNING_FILE)
def play_critical(): play_sound_async(CRITICAL_FILE)

# ==========================================
# LOAD MODEL
# ==========================================
print("Loading YOLO Model...")
model = YOLO("yolov8n.pt")
print("YOLO Model Loaded!")
load_zone_from_file()

# ==========================================
# HELPERS
# ==========================================
def draw_zone_callback(event, x, y, flags, param):
    global temp_zone_points
    if event == cv2.EVENT_LBUTTONDOWN:
        temp_zone_points.append((x, y))

def log_event(track_id, obj_class, class_id, dwell_time, severity, snapshot_path):
    global total_intrusions
    timestamp = datetime.now().strftime("%H:%M:%S")
    if severity == "WARNING":
        total_intrusions += 1
        class_intrusions[class_id] = class_intrusions.get(class_id, 0) + 1
    event_records.append({
        "Timestamp": timestamp,
        "Track_ID": track_id,
        "Object_Class": obj_class,
        "Dwell_Time_Seconds": round(dwell_time, 2),
        "Severity": severity,
        "Snapshot": snapshot_path
    })
    log_msg = f"[{timestamp}] {severity} | {obj_class} ID:{track_id} | Dwell: {dwell_time:.1f}s"
    log_queue.put(log_msg)
    print(log_msg)

# ==========================================
# AI & CAMERA WORKER
# ==========================================
def video_processing_thread():
    global is_running, ZONE_POLYGON, active_trackers, current_fps
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    if not cap.isOpened():
        print(">>> ERROR: COULD NOT OPEN CAMERA <<<")
        return
    prev_time = time.time()
    critical_beep_counter = 0
    try:
        while is_running:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            current_time = time.time()
            current_fps = round(1 / (current_time - prev_time), 1)
            prev_time = current_time
            active_class_ids = list(enabled_classes) if enabled_classes else [0]
            results = model.track(frame, persist=True, classes=active_class_ids, verbose=False, conf=0.30)
            if ZONE_POLYGON is not None:
                cv2.polylines(frame, [ZONE_POLYGON], True, (0, 0, 200), 2)
                cv2.putText(frame, "RESTRICTED ZONE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
            current_status = "SYSTEM SECURE"
            status_color = (0, 200, 68)
            active_trackers = 0
            any_critical = False
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                clss = results[0].boxes.cls.cpu().numpy().astype(int)
                active_trackers = len(ids)
                for box, track_id, cls in zip(boxes, ids, clss):
                    x_min, y_min, x_max, y_max = map(int, box)
                    class_info = CLASS_OPTIONS.get(cls, {"name": f"Class{cls}", "bgr": (200, 200, 200)})
                    class_name = class_info["name"]
                    box_color = class_info["bgr"]
                    anchor_x = int((x_min + x_max) / 2)
                    anchor_y = int(y_max)
                    cv2.circle(frame, (anchor_x, anchor_y), 5, (100, 100, 100), -1)
                    is_inside = False
                    if ZONE_POLYGON is not None:
                        is_inside = cv2.pointPolygonTest(ZONE_POLYGON, (anchor_x, anchor_y), False) >= 0
                    if is_inside:
                        intrusion_tracker[track_id] = intrusion_tracker.get(track_id, 0) + 1
                        if track_id not in track_entry_time:
                            track_entry_time[track_id] = current_time
                    else:
                        intrusion_tracker[track_id] = 0
                        if track_id in track_entry_time:
                            del track_entry_time[track_id]
                        if track_id in track_severity:
                            del track_severity[track_id]
                    dwell_time = 0.0
                    if track_id in track_entry_time:
                        dwell_time = current_time - track_entry_time[track_id]
                    severity = "SAFE"
                    if intrusion_tracker[track_id] >= ALERT_THRESHOLD_FRAMES:
                        if dwell_time >= LOITERING_DWELL_SECONDS:
                            severity = "LOITERING"
                            current_status = "LOITERING DETECTED"
                            status_color = (179, 0, 255)
                        elif dwell_time >= CRITICAL_DWELL_SECONDS:
                            severity = "CRITICAL"
                            current_status = "CRITICAL INTRUSION"
                            status_color = (34, 34, 255)
                            any_critical = True
                        else:
                            severity = "WARNING"
                            if current_status == "SYSTEM SECURE":
                                current_status = "WARNING: INTRUSION"
                                status_color = (0, 140, 255)
                    if severity != "SAFE":
                        sev_color = {"WARNING": (0, 140, 255), "CRITICAL": (34, 34, 255), "LOITERING": (179, 0, 255)}[severity]
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), sev_color, 3)
                        label = f"{severity}: {class_name} {track_id} | {dwell_time:.1f}s"
                        cv2.putText(frame, label, (x_min, y_min - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, sev_color, 2)
                        if cls == 0:
                            face_y_max = y_min + int((y_max - y_min) * 0.3)
                            face_y_max = min(face_y_max, frame.shape[0])
                            face_region = frame[y_min:face_y_max, x_min:x_max]
                            if face_region.size > 0:
                                frame[y_min:face_y_max, x_min:x_max] = cv2.GaussianBlur(face_region, (99, 99), 30)
                        prev_severity = track_severity.get(track_id, "SAFE")
                        if severity != prev_severity:
                            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                            safe_name = class_name.replace(" ", "_")
                            snapshot_path = f"evidence/{severity}_{timestamp_str}_ID{track_id}_{safe_name}.jpg"
                            cv2.imwrite(snapshot_path, frame)
                            log_event(track_id, class_name, cls, dwell_time, severity, snapshot_path)
                            if severity == "WARNING":
                                play_warning()
                            thumbnail_queue.put(snapshot_path)
                            track_severity[track_id] = severity
                    else:
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                        cv2.putText(frame, f"{class_name} {track_id}", (x_min, y_min - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            if any_critical:
                critical_beep_counter += 1
                if critical_beep_counter >= CRITICAL_BEEP_INTERVAL:
                    play_critical()
                    critical_beep_counter = 0
            else:
                critical_beep_counter = 0
            status_queue.put((current_status, status_color))
            metrics_queue.put((current_fps, total_intrusions, active_trackers, dict(class_intrusions)))
            # Resize frame to fit in the GUI
            frame_resized = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            if frame_queue.full(): frame_queue.get() 
            frame_queue.put(frame_rgb)
    except Exception:
        print(">>> THREAD ERROR <<<"); traceback.print_exc()
    cap.release()

# ==========================================
# GUI
# ==========================================
class SafetyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SENTINEL — Industrial Safety Zone Monitor")
        self.root.geometry("1000x820")
        self.root.minsize(950, 750)
        self.root.configure(bg=THEME["bg"])
        
        # ==========================================
        # 1. HEADER (top)
        # ==========================================
        header_bar = tk.Frame(root, bg=THEME["bg"], height=55)
        header_bar.pack(side="top", fill="x", padx=15, pady=(10, 0))
        header_bar.pack_propagate(False)
        logo_frame = tk.Frame(header_bar, bg=THEME["bg"])
        logo_frame.pack(side="left")
        tk.Label(logo_frame, text="◉", font=("Arial", 24, "bold"), bg=THEME["bg"], fg=THEME["critical"]).pack(side="left", padx=(0, 8))
        tk.Label(logo_frame, text="SENTINEL", font=("Arial", 20, "bold"), bg=THEME["bg"], fg=THEME["text"]).pack(side="left")
        tk.Label(logo_frame, text="  INDUSTRIAL SAFETY ZONE MONITOR", font=("Arial", 8), bg=THEME["bg"], fg=THEME["text_muted"]).pack(side="left", pady=(8, 0))
        self.clock_label = tk.Label(header_bar, text="--:--:--", font=("Consolas", 12, "bold"), bg=THEME["bg"], fg=THEME["text_muted"])
        self.clock_label.pack(side="right")
        self.update_clock()
        
        # Red separator
        tk.Frame(root, bg=THEME["border_accent"], height=2).pack(side="top", fill="x", padx=15, pady=(5, 8))
        
        # ==========================================
        # 2. METRICS (top)
        # ==========================================
        metrics_frame = tk.Frame(root, bg=THEME["panel"], highlightbackground=THEME["border"], highlightthickness=1)
        metrics_frame.pack(side="top", fill="x", padx=15, pady=(0, 8))
        self.fps_card = self._make_metric_card(metrics_frame, "FPS", "0.0", THEME["safe"])
        self.fps_card.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        self.count_card = self._make_metric_card(metrics_frame, "TOTAL INTRUSIONS", "0", THEME["critical"])
        self.count_card.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        self.tracker_card = self._make_metric_card(metrics_frame, "ACTIVE TRACKERS", "0", THEME["warning"])
        self.tracker_card.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        self.fps_label = self.fps_card.value_label
        self.count_label = self.count_card.value_label
        self.tracker_label = self.tracker_card.value_label
        
        # ==========================================
        # 3. BUTTON BAR (packed to BOTTOM FIRST — always visible!)
        # ==========================================
        btn_frame = tk.Frame(root, bg=THEME["bg"])
        btn_frame.pack(side="bottom", fill="x", pady=(0, 12), padx=15)
        
        self.start_btn = self._make_button(btn_frame, "▶  START", THEME["btn_start"], self.start_camera)
        self.start_btn.pack(side="left", padx=3, expand=True, fill="x")
        self.stop_btn = self._make_button(btn_frame, "■  STOP", THEME["btn_stop"], self.stop_camera)
        self.stop_btn.pack(side="left", padx=3, expand=True, fill="x")
        self.stop_btn.config(state="disabled")
        self.zone_btn = self._make_button(btn_frame, "⚙  SET ZONE", THEME["btn_zone"], self.set_zone)
        self.zone_btn.pack(side="left", padx=3, expand=True, fill="x")
        self.mute_btn = self._make_button(btn_frame, "🔊  SOUND: ON", THEME["btn_sound"], self.toggle_sound)
        self.mute_btn.pack(side="left", padx=3, expand=True, fill="x")
        self.export_btn = self._make_button(btn_frame, "📄  EXPORT", THEME["btn_export"], self.export_report)
        self.export_btn.pack(side="left", padx=3, expand=True, fill="x")
        
        # ==========================================
        # 4. EVENT LOG (BOTTOM, above buttons)
        # ==========================================
        log_frame = tk.Frame(root, bg=THEME["bg"])
        log_frame.pack(side="bottom", fill="x", padx=15, pady=(0, 8))
        tk.Label(log_frame, text="◢ EVENT LOG", bg=THEME["bg"], fg=THEME["text_muted"], 
                 font=("Consolas", 9, "bold")).pack(anchor="w")
        self.log_text = tk.Text(log_frame, height=3, bg=THEME["panel_alt"], fg=THEME["safe"],
            font=("Consolas", 9), insertbackground=THEME["text"],
            highlightbackground=THEME["border"], highlightthickness=1,
            relief="flat", padx=8, pady=4)
        self.log_text.pack(fill="x", pady=(3, 0))
        
        # ==========================================
        # 5. STATUS BAR (BOTTOM, above log)
        # ==========================================
        self.status_label = tk.Label(root, text="◉ SYSTEM STANDBY", font=("Consolas", 14, "bold"),
            bg=THEME["panel"], fg=THEME["text_muted"], pady=6,
            highlightbackground=THEME["border"], highlightthickness=1)
        self.status_label.pack(side="bottom", fill="x", padx=15, pady=(0, 6))
        
        # ==========================================
        # 6. MAIN CONTENT (fills remaining space)
        # ==========================================
        main_frame = tk.Frame(root, bg=THEME["bg"])
        main_frame.pack(side="top", fill="both", expand=True, padx=15, pady=(0, 8))
        
        # Video (left)
        video_container = tk.Frame(main_frame, bg=THEME["border_accent"], padx=2, pady=2)
        video_container.pack(side="left", padx=(0, 10))
        self.video_label = tk.Label(video_container, bg="black", width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
        self.video_label.pack()
        
        # Right panel
        right_panel = tk.Frame(main_frame, bg=THEME["bg"])
        right_panel.pack(side="left", fill="both", expand=True)
        
        # Filter panel
        filter_content = self._make_panel(right_panel, "◢ DETECTION CLASSES")
        filter_content.master.pack(fill="x", pady=(0, 8))
        self.class_vars = {}
        for cid, info in CLASS_OPTIONS.items():
            var = tk.BooleanVar(value=True)
            self.class_vars[cid] = var
            tk.Checkbutton(filter_content, text=info["name"], variable=var, command=self.update_class_filters,
                bg=THEME["panel"], fg=info["hex"], selectcolor=THEME["bg"],
                activebackground=THEME["panel"], activeforeground=info["hex"],
                font=("Consolas", 10, "bold"), anchor="w", padx=8).pack(fill="x", pady=2)
        tk.Frame(filter_content, bg=THEME["border"], height=1).pack(fill="x", pady=6)
        tk.Label(filter_content, text="INTRUSION COUNTS", bg=THEME["panel"], fg=THEME["text_muted"], 
                 font=("Consolas", 8, "bold")).pack(anchor="w", padx=8)
        self.per_class_label = tk.Label(filter_content, text="", bg=THEME["panel"], fg=THEME["text"], 
                                         font=("Consolas", 9), justify="left", anchor="w")
        self.per_class_label.pack(fill="x", padx=8, pady=(2, 6))
        
        # Evidence panel
        evidence_content = self._make_panel(right_panel, "◢ RECENT EVIDENCE")
        evidence_content.master.pack(fill="x")
        self.thumbnail_labels = []
        for i in range(3):
            lbl = tk.Label(evidence_content, bg=THEME["panel_alt"], width=18, height=6,
                          highlightbackground=THEME["border"], highlightthickness=1)
            lbl.pack(pady=2, padx=5)
            self.thumbnail_labels.append(lbl)
        
        self.update_gui()

    def _make_metric_card(self, parent, title, value, color):
        card = tk.Frame(parent, bg=THEME["panel_alt"], padx=15, pady=6)
        tk.Label(card, text=title, bg=THEME["panel_alt"], fg=THEME["text_muted"], 
                 font=("Consolas", 8, "bold")).pack(anchor="w")
        value_label = tk.Label(card, text=value, bg=THEME["panel_alt"], fg=color, 
                              font=("Consolas", 18, "bold"))
        value_label.pack(anchor="w")
        card.value_label = value_label
        return card

    def _make_panel(self, parent, title_text):
        outer = tk.Frame(parent, bg=THEME["panel"], highlightbackground=THEME["border"], highlightthickness=1)
        title_bar = tk.Frame(outer, bg=THEME["bg"], height=24)
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)
        tk.Label(title_bar, text=title_text, bg=THEME["bg"], fg=THEME["header"],
                 font=("Consolas", 9, "bold")).pack(side="left", padx=8, pady=4)
        tk.Frame(title_bar, bg=THEME["border_accent"], width=3).pack(side="left", fill="y")
        content = tk.Frame(outer, bg=THEME["panel"], padx=5, pady=8)
        content.pack(fill="both", expand=True)
        return content

    def _make_button(self, parent, text, bg_color, command):
        return tk.Button(parent, text=text, command=command, bg=bg_color, fg="white",
            font=("Consolas", 10, "bold"), padx=10, pady=8, relief="flat", bd=0,
            activebackground=bg_color, activeforeground="white", cursor="hand2")

    def update_clock(self):
        self.clock_label.config(text=datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.root.after(1000, self.update_clock)

    def update_class_filters(self):
        global enabled_classes
        enabled_classes = [cid for cid, var in self.class_vars.items() if var.get()]

    def toggle_sound(self):
        global sound_enabled
        sound_enabled = not sound_enabled
        if sound_enabled:
            self.mute_btn.config(text="🔊  SOUND: ON", bg=THEME["btn_sound"])
        else:
            self.mute_btn.config(text="🔇  SOUND: OFF", bg="#3d3d3d")

    def export_report(self):
        if not event_records:
            messagebox.showinfo("Export", "No events to export yet.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV Files", "*.csv")],
            initialfile=f"safety_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        if not file_path: return
        try:
            with open(file_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["Timestamp", "Track_ID", "Object_Class", "Dwell_Time_Seconds", "Severity", "Snapshot"])
                writer.writeheader()
                writer.writerows(event_records)
            messagebox.showinfo("Export", f"Report saved!\n{len(event_records)} events exported.")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def start_camera(self):
        global is_running
        if not is_running:
            is_running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.zone_btn.config(state="disabled")
            self.status_label.config(text="◉ SYSTEM ACTIVE", bg=THEME["panel"], fg=THEME["safe"])
            self.log_text.insert(tk.END, "[SYSTEM] Camera started.\n")
            self.log_text.see(tk.END)
            self.thread = threading.Thread(target=video_processing_thread, daemon=True)
            self.thread.start()

    def stop_camera(self):
        global is_running
        is_running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.zone_btn.config(state="normal")
        self.status_label.config(text="◉ SYSTEM STANDBY", bg=THEME["panel"], fg=THEME["text_muted"])

    def set_zone(self):
        global ZONE_POLYGON, temp_zone_points, drawing_zone
        self.stop_camera()
        temp_zone_points = []
        drawing_zone = True
        cv2.namedWindow("Draw Zone")
        cv2.setMouseCallback("Draw Zone", draw_zone_callback)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while drawing_zone:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            for i in range(len(temp_zone_points)):
                cv2.circle(frame, temp_zone_points[i], 5, (0, 0, 255), -1)
                if i > 0:
                    cv2.line(frame, temp_zone_points[i-1], temp_zone_points[i], (0, 0, 200), 2)
            cv2.putText(frame, "Click points. ENTER to save. ESC to cancel.", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
            cv2.imshow("Draw Zone", frame)
            key = cv2.waitKey(1)
            if key == 13: 
                if len(temp_zone_points) >= 3:
                    ZONE_POLYGON = np.array(temp_zone_points, np.int32)
                    save_zone_to_file(ZONE_POLYGON)
                break
            elif key == 27: break
        cap.release()
        cv2.destroyWindow("Draw Zone")
        drawing_zone = False
        self.start_camera()

    def update_gui(self):
        if not frame_queue.empty():
            frame = frame_queue.get()
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        if not status_queue.empty():
            status, color = status_queue.get()
            hex_color = '#%02x%02x%02x' % (color[2], color[1], color[0])
            if "CRITICAL" in status or "LOITERING" in status or "WARNING" in status:
                self.status_label.config(text=f"⚠ {status}", bg=hex_color, fg="white")
            else:
                self.status_label.config(text=f"◉ {status}", bg=THEME["panel"], fg=THEME["safe"])

        if not metrics_queue.empty():
            fps, total, trackers, per_class = metrics_queue.get()
            self.fps_label.config(text=str(fps))
            self.count_label.config(text=str(total))
            self.tracker_label.config(text=str(trackers))
            txt = "\n".join([f"  {CLASS_OPTIONS[cid]['name']:<12} {per_class.get(cid,0)}" for cid in CLASS_OPTIONS])
            self.per_class_label.config(text=txt)

        if not log_queue.empty():
            log_msg = log_queue.get()
            self.log_text.insert(tk.END, log_msg + "\n")
            self.log_text.see(tk.END)

        if not thumbnail_queue.empty():
            new_snapshot = thumbnail_queue.get()
            try:
                if os.path.exists(new_snapshot) and os.path.getsize(new_snapshot) > 0:
                    for i in range(len(self.thumbnail_labels)-1, 0, -1):
                        prev_img = self.thumbnail_labels[i-1].cget("image")
                        self.thumbnail_labels[i].config(image=prev_img)
                        self.thumbnail_labels[i].image = getattr(self.thumbnail_labels[i-1], "image", None)
                    img = Image.open(new_snapshot)
                    img.thumbnail((140, 90))
                    photo = ImageTk.PhotoImage(img)
                    self.thumbnail_labels[0].config(image=photo)
                    self.thumbnail_labels[0].image = photo
            except Exception as e:
                print(f"Thumbnail error: {e}")

        self.root.after(30, self.update_gui)

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    app = SafetyApp(root)
    root.mainloop()