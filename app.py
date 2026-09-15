import cv2
import time
import numpy as np
import pandas as pd
import streamlit as st
import tempfile
import os
import torch
from core import MultiObjectTracker, SpatialAnalytics, StreamVisualizer

st.set_page_config(
    page_title="AI Vision & Telemetry System",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Real-Time Object Detection & Tracking System")
st.caption("Edge-accelerated Computer Vision Pipeline with ByteTrack & Spatial Telemetry")

# Sidebar Controls
st.sidebar.header("⚙️ Pipeline Configuration")

# Model availability check (auto-downloads if not cached)
available_models = ["yolov8n.pt", "yolov8s.pt"]
if os.path.exists("yolov8m.pt"):
    available_models.insert(0, "yolov8m.pt")
    
model_type = st.sidebar.selectbox("Model Architecture", available_models, index=0)
conf_thresh = st.sidebar.slider("Detection Confidence", min_value=0.10, max_value=0.90, value=0.35, step=0.05)
iou_thresh = st.sidebar.slider("NMS IoU Threshold", min_value=0.20, max_value=0.80, value=0.50, step=0.05)

# Input Source selection
video_source = st.sidebar.radio(
    "Input Source",
    ["Demo Video (Preloaded)", "Upload Video (MP4/AVI)", "Live Webcam"]
)

# Main Dashboard Layout
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
fps_metric = kpi1.empty()
in_metric = kpi2.empty()
out_metric = kpi3.empty()
zone_metric = kpi4.empty()

frame_placeholder = st.empty()

video_path = None
if video_source == "Demo Video (Preloaded)":
    demo_file = "data/samples/sample_traffic.mp4"
    if os.path.exists(demo_file):
        video_path = demo_file
        st.sidebar.success("Loaded preloaded traffic simulation demo.")
    else:
        st.sidebar.warning("Demo video not found, please upload one.")
elif video_source == "Upload Video (MP4/AVI)":
    uploaded_file = st.sidebar.file_uploader("Upload video (MP4/AVI/MOV)", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name
elif video_source == "Live Webcam":
    video_path = 0
    st.sidebar.info("Webcam mode works best on local machine. For cloud, use Demo or Upload.")

col_btn1, col_btn2 = st.sidebar.columns(2)
start_btn = col_btn1.button("▶ Start Stream", use_container_width=True)
stop_btn = col_btn2.button("⏹ Stop Stream", use_container_width=True)

if start_btn and video_path is not None:
    cap = cv2.VideoCapture(video_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    tracker = MultiObjectTracker(
        model_path=model_type,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        device=device
    )
    analytics = SpatialAnalytics()
    visualizer = StreamVisualizer(class_names=tracker.get_classes_map())
    
    prev_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # If demo video, loop back for continuous display
            if video_source == "Demo Video (Preloaded)":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            st.info("End of video reached.")
            break
            
        # Inference & Tracking
        tracked, inference_ms = tracker.track(frame)
        analytics_result = analytics.update(tracked)
        
        # FPS Calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        # Visualization
        annotated_frame = visualizer.annotate(
            frame=frame,
            tracked_detections=tracked,
            analytics_result=analytics_result,
            fps=fps,
            inference_ms=inference_ms
        )
        
        # Update metrics
        fps_metric.metric(label="Pipeline Throughput", value=f"{fps:.1f} FPS")
        in_metric.metric(label="Inflow Count (Tripwire)", value=f"{analytics_result.get('in_count', 0)}")
        out_metric.metric(label="Outflow Count (Tripwire)", value=f"{analytics_result.get('out_count', 0)}")
        
        zone_status = "⚠️ ALERT" if analytics_result.get("zone_intrusion_active") else "✅ Normal"
        zone_metric.metric(label="Restricted Zone Status", value=zone_status)
        
        # Stream frame
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        if stop_btn:
            break
            
    cap.release()
