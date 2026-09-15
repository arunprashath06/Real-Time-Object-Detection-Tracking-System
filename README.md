# Real-Time Object Detection, Multi-Object Tracking and Spatial Analytics System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green.svg)](https://github.com/ultralytics/ultralytics)
[![Tracking](https://img.shields.io/badge/Tracker-ByteTrack-orange.svg)](https://github.com/ifzhang/ByteTrack)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An edge-deployable, high-throughput computer vision pipeline engineered for real-time video surveillance, robotic perception, and industrial safety automation. The system integrates deep learning object detection (YOLOv8), multi-object tracking (ByteTrack), and spatial telemetry analytics including directional line-crossing counters and geofenced hazard zone intrusion detection.

---

## 📌 Architecture Overview

```
                          ┌────────────────────────┐
                          │   Video Input Stream   │
                          │ (RTSP / Webcam / File) │
                          └───────────┬────────────┘
                                      │ (BGR Frames)
                                      ▼
                          ┌────────────────────────┐
                          │ Object Detection Layer │
                          │  (YOLOv8 with TensorRT │
                          │   and FP16 Optimization) │
                          └───────────┬────────────┘
                                      │ (xyxy, class_id, conf)
                                      ▼
                          ┌────────────────────────┐
                          │ Multi-Object Tracker   │
                          │ (ByteTrack Association │
                          │  and Persistent TrackID) │
                          └───────────┬────────────┘
                                      │ (Active Tracks and Centroids)
                                      ▼
                 ┌────────────────────┴────────────────────┐
                 │                                         │
                 ▼                                         ▼
   ┌───────────────────────────┐             ┌───────────────────────────┐
   │    Spatial Telemetry      │             │  Visualization and Overlay  │
   │  - Bi-directional Tripwire│             │  - Dynamic Bounding Boxes │
   │  - Polygon Zone Intrusion │             │  - Trajectory Trail HUD   │
   │  - Trajectory Analysis    │             │  - Live Throughput (FPS)  │
   └─────────────┬─────────────┘             └─────────────┬─────────────┘
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      ▼
                          ┌────────────────────────┐
                          │ Interactive Dashboard  │
                          │  (Streamlit and OpenCV)  │
                          └────────────────────────┘
```

---

## 🚀 Key Features

- **High-Throughput Detection:** Leverages Ultralytics YOLOv8 architectures (
ano, small, medium, x-large) with GPU acceleration (CUDA and half-precision FP16), achieving sub-10ms latency and 90+ FPS on dedicated hardware.
- **Robust Multi-Object Tracking (ByteTrack):** Implements two-stage data association using Kalman filtering and Hungarian matching over IoU (Intersection over Union). Handles severe occlusions, motion blur, and low-confidence detections without track fragmentation.
- **Directional Virtual Tripwire:** Vector cross-product mathematics track object trajectories between sequential frames (-1$ to $), accurately tallying bi-directional inflows and outflows.
- **Restricted Safety Geofencing:** Polygon zone intrusion detection based on the ray-casting point-in-polygon algorithm, generating real-time visual alerts for industrial workcell safety and restricted-area monitoring.
- **Dual Runtime Interfaces:**
  - **Interactive Streamlit Web Dashboard:** Designed for cloud deployment, video upload analysis, and real-time parameter tuning (confidence threshold, NMS IoU).
  - **Native OpenCV CLI Streamer:** Tailored for edge devices and live webcam feeds with minimal rendering latency.

---

## 🛠️ Tech Stack

- **Core Vision and ML:** Python 3.11, Ultralytics YOLOv8, PyTorch, TorchVision, Supervision
- **Image and Stream Processing:** OpenCV (cv2), NumPy
- **Spatial Analytics and Telemetry:** Shapely, SciPy, Lapx
- **Web UI and Visualization:** Streamlit, Plotly, Pandas
- **Hardware Acceleration:** NVIDIA CUDA 12.4, Tensor Cores

---

## ⚡ Performance Benchmarks

*Evaluated on NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM) with FP16 Tensor Optimization:*

| Model Variant | Resolution | Parameters | Inference Latency | P95 Latency | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8-Nano (yolov8n)** | 640 x 640 | 3.2M | **10.4 ms** | 15.2 ms | **96.1 FPS** |
| **YOLOv8-Nano (yolov8n)** | 1280 x 720 (HD) | 3.2M | **9.7 ms** | 14.8 ms | **102.5 FPS** |
| **YOLOv8-Small (yolov8s)** | 640 x 640 | 11.2M | **14.2 ms** | 19.8 ms | **70.4 FPS** |
| **YOLOv8-Medium (yolov8m)**| 640 x 640 | 25.9M | **20.3 ms** | 28.1 ms | **49.2 FPS** |

---

## 📂 Project Structure

```
Real-Time-Object-Detection-Tracking-System/
│
├── core/
│   ├── __init__.py
│   ├── detector.py          # Modular YOLO detection wrapper with device auto-resolution
│   ├── tracker.py           # ByteTrack multi-object tracking and trajectory caching
│   ├── analytics.py         # Vector tripwire counting and polygon intrusion logic
│   └── visualizer.py        # Stream visualizer rendering HUD, traces, and alert banners
│
├── data/
│   └── samples/             # Sample traffic and surveillance clips for testing
│
├── benchmarks/
│   └── benchmark.py         # Hardware latency, P50/P95 distribution, and FPS profiler
│
├── app.py                   # Streamlit web application for cloud and local dashboard
├── main.py                  # Standalone low-latency OpenCV runner
├── requirements.txt         # Production dependencies for cloud deployment
├── run_dashboard.bat        # Windows one-click dashboard launcher
└── run_webcam.bat           # Windows one-click webcam launcher
```

---

## 💻 Installation and Quick Start

### 1. Clone the Repository
```ash
git clone https://github.com/arunprashath06/Real-Time-Object-Detection-Tracking-System.git
cd Real-Time-Object-Detection-Tracking-System
```

### 2. Create and Activate a Virtual Environment
```ash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```ash
pip install -r requirements.txt
```

### 4. Launch the Application

**Option A — Web Dashboard (Streamlit):**
```ash
streamlit run app.py
```

**Option B — Native High-Speed CLI Stream:**
```ash
# Live Webcam
python main.py --source 0 --model yolov8s.pt --conf 0.40

# Custom Video File
python main.py --source path/to/video.mp4 --model yolov8m.pt
```

**Option C — Run Hardware Benchmark:**
```ash
python benchmarks/benchmark.py
```

---

## 📄 License
Distributed under the MIT License. See LICENSE for more information.
