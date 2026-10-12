<div align="center">

# ◉ SENTINEL
### AI-Powered Industrial Safety Zone Intrusion Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=flat)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Real-time detection of unauthorized personnel, cell phones, and laptops in restricted industrial zones.**

Built for JIGNASA 2026 Hackathon | Round 3 Finalist

</div>

---

## 🎯 The Problem

In industrial environments, restricted zones around heavy machinery, high-voltage equipment, and chemical storage areas are critical safety boundaries. Manual monitoring is:
- **Expensive** (requires dedicated CCTV operators)
- **Unreliable** (human attention degrades within 20 minutes)
- **Non-compliant** (no audit trail for safety inspections)

Furthermore, **cell phone usage in hazardous zones** is a leading cause of distraction-related accidents, yet it's nearly impossible to enforce manually.

## 💡 Our Solution

**SENTINEL** is a computer-vision-based intrusion detection system that:
1. Detects restricted objects in real time using YOLOv8
2. Maps them to a user-drawn polygonal zone
3. Tracks dwell time and escalates through 3 severity levels
4. Blurs faces for privacy compliance
5. Logs every incident with photographic evidence and CSV audit trail

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Multi-Class Detection** | Person, Cell Phone, Laptop — with per-class toggle |
| ⚙️ **Dynamic Zone Drawing** | Draw the restricted polygon with your mouse at runtime |
| ⏱️ **3-Tier Escalation** | WARNING (5 frames) → CRITICAL (10s) → LOITERING (30s) |
| 🔒 **Privacy-First** | Gaussian blur on all detected person faces |
| 📸 **Evidence Snapshots** | Auto-saves timestamped images with severity + track ID |
| 📊 **Live Metrics Dashboard** | FPS, total intrusions, active trackers, per-class counts |
| 🔔 **Multi-Level Audio** | Soft warning beep + looping critical siren |
| 📄 **CSV Audit Report** | One-click export of full incident history |
| 💾 **Zone Persistence** | Zone auto-saves to `zone.json` and reloads on startup |

---

## 📸 Screenshots

<table>
<tr>
<td><img src="screenshots/dashboard.png" width="400"/></td>
<td><img src="screenshots/evidence.png" width="400"/></td>
</tr>
<tr>
<td align="center"><i>Live detection dashboard</i></td>
<td align="center"><i>Privacy-blurred evidence snapshot</i></td>
</tr>
</table>

---

## 🏗️ Architecture

┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Webcam │────▶│ YOLOv8 │────▶│ ByteTrack │
│ (OpenCV) │ │ Detector │ │ Tracker │
└──────────────┘ └──────────────┘ └──────┬───────┘
│
▼
┌──────────────────────────────────────────────────────┐
│ CUSTOM SAFETY LOGIC LAYER │
│ • Zone Mapper (point-in-polygon test) │
│ • Temporal Filter (N-frame confirmation) │
│ • Severity Escalator (WARNING → CRITICAL → LOITER) │
│ • Privacy Filter (Gaussian blur on faces) │
│ • Evidence Snapshotter (auto-save JPEG) │
└──────────────────┬───────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────┐
│ TKINTER GUI (Multi-threaded) │
│ • Live video feed • Metrics dashboard │
│ • Audio alerts • Event log │
│ • Thumbnail strip • CSV export │
└──────────────────────────────────────────────────────┘


---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Object Detection** | YOLOv8n (Ultralytics) |
| **Tracking** | ByteTrack (built into Ultralytics) |
| **Vision Processing** | OpenCV 4.x |
| **GUI** | Tkinter |
| **Audio** | winsound + custom-generated WAV alarms |
| **Threading** | Python `threading` + `queue.Queue` for lock-free IPC |

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- Windows 10/11 (uses `winsound` for audio)
- Webcam

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/sentinel-safety-monitor.git
cd sentinel-safety-monitor

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py

🎮 Usage
Launch: Run python app.py

Draw Zone: Click ⚙ SET ZONE, then click 4 points on the floor to define the restricted area. Press ENTER to save.

Start Monitoring: Click ▶ START

Test Detection: Walk into the zone — the system will escalate from WARNING → CRITICAL → LOITERING based on dwell time.

Export Report: Click 📄 EXPORT to save a CSV audit trail.

Keyboard Shortcuts (during zone drawing)
Left Click — Add a point to the polygon

ENTER — Save the zone (also persists to zone.json)

ESC — Cancel

📊 How It Works
The Core Intrusion Logic (Pseudocode)
python
for each detected object:
    # 1. Get the anchor point (bottom-center = feet on ground)
    anchor = ((x_min + x_max) / 2, y_max)
    
    # 2. Spatial check: is the anchor inside the polygon?
    is_inside = cv2.pointPolygonTest(ZONE_POLYGON, anchor, False) >= 0
    
    # 3. Temporal filter: N-frame confirmation to avoid flicker
    if is_inside:
        counter[track_id] += 1
    else:
        counter[track_id] = 0
    
    # 4. Severity escalation based on dwell time
    if counter[track_id] >= 5:
        dwell = now - entry_time[track_id]
        if dwell >= 30: severity = "LOITERING"
        elif dwell >= 10: severity = "CRITICAL"
        else: severity = "WARNING"
Why Bottom-Center, Not Center?
We use the bottom-center point (where the person's feet are) as the anchor, not the geometric center of the bounding box. This is critical because:

A person leaning over the line shouldn't trigger an alarm

Physical presence = feet on the ground

Reduces false positives from perspective distortion

📈 Roadmap
□ Forklift detection — Fine-tune YOLOv8 on a custom forklift dataset
□ Multi-camera support — Fuse feeds from multiple viewpoints
□ Web dashboard — Migrate to Flask/Streamlit for remote monitoring
□ Edge deployment — Optimize for NVIDIA Jetson Nano
□ PPE detection — Helmet, vest, goggles compliance
□ Directional alerts — Only trigger on zone entry, not exit
🤝 Contributing
Contributions are welcome! Please open an issue first to discuss any major changes.

📜 License
This project is licensed under the MIT License — see LICENSE for details.

🙏 Acknowledgements
Ultralytics for YOLOv8

COCO Dataset for pretrained weights

JIGNASA 2026 for the problem statement
