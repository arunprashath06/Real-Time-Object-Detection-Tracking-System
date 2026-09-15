# Real-Time Object Detection, Multi-Object Tracking & Spatial Analytics System

An edge-optimized computer vision and spatial telemetry system engineered for real-time video feeds, robotic perception, and industrial safety automation.

---

## 🎯 Key Highlights & Architecture
- **Inference Engine:** YOLOv8 / YOLOv11 running with CUDA & FP16 half-precision on NVIDIA RTX 3050.
- **Tracking Algorithm:** ByteTrack (maintains persistent identity across severe occlusions and low-confidence detections).
- **Spatial Intelligence:** 
  - Directional Tripwire Line-Crossing Counter (Inflow / Outflow).
  - Restricted Polygon Safety Zone Intrusion detection.
  - Motion Trajectory History and Centroid Tracking.
- **Benchmarking Suite:** Measures end-to-end latency (ms), P50/P95/P99 latency distribution, and throughput (FPS).
- **Interactive UI:** Real-time Streamlit dashboard + low-latency OpenCV CLI runner.

---

## 🎓 Interview & Viva Defense Cheat-Sheet (PES & Campus Placement)

### Q1: What is the fundamental difference between Object Detection and Multi-Object Tracking (MOT)?
- **Detection (YOLO):** Operates on static, isolated frames. It identifies *what* an object is (class label) and *where* it is (bounding box coordinates [x1, y1, x2, y2]). However, it has no temporal memory — it treats every frame as an entirely new world.
- **Tracking (ByteTrack):** Adds temporal memory across sequential frames. It associates detections across time, assigning and preserving a persistent tracker_id (e.g., Car #4).

### Q2: Why ByteTrack over DeepSORT?
- Traditional trackers (like SORT/DeepSORT) discard bounding boxes with low confidence scores (e.g., below 0.3). When an object gets partially occluded or blurred, its detection confidence drops, causing the tracker to lose it and assign a new ID when it reappears (known as ID Switch).
- **ByteTrack's Innovation:** It uses *all* detection boxes (both high and low confidence) in a two-stage association strategy via IoU (Intersection over Union). It matches high-score boxes first, and then matches remaining unmatched tracks with low-score boxes, eliminating broken tracks without needing computationally heavy Re-ID neural networks.

### Q3: How is Spatial Line Crossing & Zone Intrusion calculated?
- **Line Crossing:** Uses vector cross-product orientation comparing the object centroid between frame t-1 and frame t relative to the tripwire line vector AB.
- **Polygon Zone Intrusion:** Uses the Ray-Casting Algorithm (point-in-polygon test) to determine whether the bottom-center point of a bounding box lies within the restricted coordinate polygon.

### Q4: How did you optimize inference for Real-Time execution?
- Enabled FP16 Half-Precision on CUDA Tensor cores.
- Decoupled detection frame rates from rendering via lightweight frame caching.
- Achieved ~45-60 FPS at 640x640 resolution on an NVIDIA RTX 3050 Laptop GPU.

---

## 🚀 How to Run

### 1. Run the Interactive Streamlit Dashboard:
`ash
streamlit run app.py
`

### 2. Run the High-Performance OpenCV CLI Stream:
`ash
# Webcam
python main.py --source 0

# Video file
python main.py --source data/samples/traffic.mp4
`

### 3. Run Hardware Benchmark:
`ash
python benchmarks/benchmark.py
`
