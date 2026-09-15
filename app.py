import cv2
import time
import numpy as np
import pandas as pd
import streamlit as st
import tempfile
import os
import torch
from PIL import Image
from core import MultiObjectTracker, SpatialAnalytics, StreamVisualizer

st.set_page_config(
    page_title="AI Vision and Telemetry System",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Real-Time Object Detection and Tracking System")
st.caption("Edge-accelerated Computer Vision Pipeline with ByteTrack and Spatial Telemetry")

# Detect if running on Streamlit Cloud (CPU-only)
IS_CLOUD = not torch.cuda.is_available()

# ─── Sidebar Controls ──────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Pipeline Configuration")

if IS_CLOUD:
    available_models = ["yolov8s.pt", "yolov8n.pt"]
    st.sidebar.info("☁️ Running on Streamlit Cloud (CPU mode). yolov8s gives best accuracy; yolov8n is faster.")
else:
    available_models = ["yolov8n.pt", "yolov8s.pt"]
    if os.path.exists("yolov8m.pt"):
        available_models.append("yolov8m.pt")
    if os.path.exists("yolov8x.pt"):
        available_models.append("yolov8x.pt")

model_type   = st.sidebar.selectbox("Model Architecture", available_models, index=0)
conf_thresh  = st.sidebar.slider("Detection Confidence", 0.10, 0.90, 0.25, 0.05)
iou_thresh   = st.sidebar.slider("NMS IoU Threshold",    0.20, 0.80, 0.50, 0.05)

# ─── Input Source ──────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Input Source")

source_options = ["Demo Video (Preloaded)", "Upload Video (MP4/AVI)", "Browser Camera (Live)"]
if not IS_CLOUD:
    source_options.append("Local Webcam (OpenCV)")

video_source = st.sidebar.radio("Select Source", source_options)

# ─── KPI Row ───────────────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
fps_metric  = kpi1.empty()
in_metric   = kpi2.empty()
out_metric  = kpi3.empty()
zone_metric = kpi4.empty()

frame_placeholder = st.empty()

# ─── Model loader (cached so it only loads once per session) ───────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(model_path, conf, iou, device):
    tracker    = MultiObjectTracker(model_path=model_path, conf_thresh=conf, iou_thresh=iou, device=device)
    analytics  = SpatialAnalytics()
    visualizer = StreamVisualizer(class_names=tracker.get_classes_map())
    return tracker, analytics, visualizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════════════════════
# BROWSER CAMERA MODE  (uses st.camera_input — works on Streamlit Cloud)
# ═══════════════════════════════════════════════════════════════════════════════
if video_source == "Browser Camera (Live)":
    st.info(
        "📷 **Browser Camera** — Click **Allow** when your browser asks for camera permission. "
        "Each captured frame is run through the YOLO detector automatically."
    )

    camera_frame = st.camera_input("Point your camera and capture a frame")

    if camera_frame is not None:
        with st.spinner(f"Loading {model_type} on {device.upper()} …"):
            tracker, analytics, visualizer = load_pipeline(model_type, conf_thresh, iou_thresh, device)

        # Convert the uploaded image bytes → numpy BGR frame
        img = Image.open(camera_frame).convert("RGB")
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        t0 = time.time()
        tracked, inference_ms = tracker.track(frame)
        analytics_result = analytics.update(tracked)
        fps = 1.0 / (time.time() - t0)

        annotated = visualizer.annotate(
            frame=frame,
            tracked_detections=tracked,
            analytics_result=analytics_result,
            fps=fps,
            inference_ms=inference_ms
        )

        fps_metric.metric("Pipeline Throughput",    f"{fps:.1f} FPS")
        in_metric.metric("Inflow Count (Tripwire)", analytics_result.get("in_count", 0))
        out_metric.metric("Outflow Count (Tripwire)", analytics_result.get("out_count", 0))
        zone_status = "⚠️ ALERT" if analytics_result.get("zone_intrusion_active") else "✅ Normal"
        zone_metric.metric("Restricted Zone Status", zone_status)

        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        st.caption("📸 Capture another frame above to keep detecting.")

# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO FILE MODES  (Demo or Upload or Local Webcam)
# ═══════════════════════════════════════════════════════════════════════════════
else:
    video_path = None

    if video_source == "Demo Video (Preloaded)":
        demo_file = "data/samples/sample_traffic.mp4"
        if os.path.exists(demo_file):
            video_path = demo_file
            st.sidebar.success("Loaded preloaded traffic simulation demo.")
        else:
            st.sidebar.warning("Demo video not found. Please upload a video instead.")

    elif video_source == "Upload Video (MP4/AVI)":
        uploaded_file = st.sidebar.file_uploader("Upload video (MP4/AVI/MOV)", type=["mp4", "avi", "mov"])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_path = tfile.name

    elif video_source == "Local Webcam (OpenCV)":
        video_path = 0
        st.sidebar.warning("Webcam via OpenCV only works when running locally (not on Streamlit Cloud).")

    col_btn1, col_btn2 = st.sidebar.columns(2)
    start_btn = col_btn1.button("▶ Start Stream", use_container_width=True)
    stop_btn  = col_btn2.button("⏹ Stop Stream",  use_container_width=True)

    if start_btn and video_path is not None:
        cap = cv2.VideoCapture(video_path)

        with st.spinner(f"Loading {model_type} on {device.upper()} …"):
            tracker, analytics, visualizer = load_pipeline(model_type, conf_thresh, iou_thresh, device)

        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if video_source == "Demo Video (Preloaded)":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                st.info("End of video reached.")
                break

            tracked, inference_ms = tracker.track(frame)
            analytics_result = analytics.update(tracked)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            annotated = visualizer.annotate(
                frame=frame,
                tracked_detections=tracked,
                analytics_result=analytics_result,
                fps=fps,
                inference_ms=inference_ms
            )

            fps_metric.metric("Pipeline Throughput",      f"{fps:.1f} FPS")
            in_metric.metric("Inflow Count (Tripwire)",   analytics_result.get("in_count", 0))
            out_metric.metric("Outflow Count (Tripwire)", analytics_result.get("out_count", 0))
            zone_status = "⚠️ ALERT" if analytics_result.get("zone_intrusion_active") else "✅ Normal"
            zone_metric.metric("Restricted Zone Status",  zone_status)

            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            if stop_btn:
                break

        cap.release()

    elif start_btn and video_path is None:
        st.warning("⚠️ Please select or upload a video source before starting.")
