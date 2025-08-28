# AI-Powered Crowd Monitoring & Emergency Response Platform

Real-time intelligent surveillance platform for people detection, tracking, density estimation, and anomaly detection (fire, smoke, fallen individuals) with heatmaps (3D plane analysis) and a mini‑map for congestion prediction. Built with **YOLOv8**, **OpenCV**, **Flask**, and **NumPy**.

---

## 🔍 Overview
This project provides a web-based dashboard that ingests CCTV or recorded video, detects/ tracks people, estimates density, surfaces anomalies, and visualizes crowd dynamics via heatmaps and a live mini‑map. It is designed for public safety scenarios such as events, transit hubs, and campuses.

---

## ✨ Key Features
- **People Detection & Tracking**: YOLOv8 + centroid tracker with persistent IDs
- **Density Estimation**: Grid-based density levels (LOW/MEDIUM/HIGH)
- **Anomaly Detection**: Fire, smoke, and fallen-individual detection
- **Heatmap (3D Plane Analysis)**: Spatial density visualization from 2D CCTV
- **Mini‑Map + Congestion Prediction**: Top‑down projection with early crowding alerts
- **Live Dashboard**: Flask APIs streaming annotated frames and JSON analytics

---

## 🧱 System Architecture
1. **Video Input** → CCTV / file
2. **Preprocess** → resize/normalize frames
3. **Detection** → YOLOv8 (person class)
4. **Tracking** → centroid-based ID assignment, motion smoothing
5. **Analytics** → density grid, heatmap, anomaly detection (fire/smoke/fallen)
6. **Projection** → 3D plane mapping + mini‑map
7. **Serve** → Flask endpoints for images + JSON
8. **Dashboard** → web UI with overlays and alerts


---

## 🛠️ Tech Stack
- **Backend**: Python, Flask
- **CV/AI**: OpenCV, YOLOv8 (Ultralytics), NumPy
- **Frontend**: HTML, CSS, JavaScript
- **Concurrency**: Python threading
- **Optional**: GPU acceleration (CUDA) for faster inference

---

## 🚀 Quick Start

### 1) Environment
```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

pip install -U pip wheel setuptools
pip install -r requirements.txt  # create this if not present (see below)
```

**Minimal `requirements.txt`:**
```
flask
opencv-python
ultralytics
numpy
```
> If you use GPU, install the correct **torch** + **cuda** build from PyTorch website first, then `pip install ultralytics`.

### 2) Model Weights
- Default YOLOv8 weights: `yolov8n.pt` (fast) or `yolov8s.pt` (balanced).
- Place custom weights in `weights/` and update the path in the code/config.

### 3) Run
```bash
# set any necessary environment variables (optional)
export FLASK_ENV=production
export PYTHONUNBUFFERED=1

# start the app
python app.py
# or (if using Flask app factory)
flask run --host 0.0.0.0 --port 5000
```

Open the dashboard at: `http://localhost:5000/`

---

## ⚙️ Configuration
Create `.env` (optional):
```
VIDEO_SOURCE=0             # 0 for webcam, or path to video file
MODEL_PATH=weights/yolov8n.pt
FRAME_WIDTH=960            # resize width for speed
DENSITY_THRESHOLDS=6,15    # LOW<6, 6-14 MED, >=15 HIGH
```
The app can also accept query/body params on start endpoints (see APIs).

---

## 📡 API Endpoints (examples)
- `POST /start_cctv_stream`  
  **Body (json)**: `{{ "video_source": 0 }}` or `{{ "video_source": "data/crowd.mp4" }}`
- `GET /get_cctv_frame`  
  Returns latest annotated frame as Base64 + analytics JSON
- `POST /stop_cctv_stream`  
  Stops the video worker

**cURL examples**
```bash
curl -X POST http://localhost:5000/start_cctv_stream -H "Content-Type: application/json" -d '{"video_source": 0}'

curl http://localhost:5000/get_cctv_frame

curl -X POST http://localhost:5000/stop_cctv_stream
```

---

## 🗂️ Project Structure
```
.
├── app.py
├── weights/
│   └── yolov8n.pt
├── static/
│   └── js/, css/, images/
├── templates/
│   └── index.html
├── data/
│   └── samples/
├── docs/
│   ├── architecture.png
│   └── results/
│       ├── detection.png
│       ├── heatmap.png
│       └── minimap.png
└── README.md
```

---

## 🧭 Roadmap / Future Enhancements
- **Emergency Routing Engine**: dynamic evacuation paths
- **NLP Command Interface**: voice/text commands for operators
- **Post-Event Analytics**: detailed reports & heatmap time‑lapse exports
- Multi-camera fusion and re‑identification
- Role-based access control & audit logging

---

## 🙌 Acknowledgements
- **SIFY Technologies Ltd** – Internship opportunity and guidance
- **Mentor**: *Senthil Kumar*
- **SRM Institute of Science and Technology, Kattankulathur*

---

## 📝 License
Add your preferred license (e.g., MIT) in `LICENSE`.

---

## 👤 Author
**Raamiz Hussain Shikoh**  
B.Tech Computer Science, SRM Institute of Science and Technology, Kattankulathur




