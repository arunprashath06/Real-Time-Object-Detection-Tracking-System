import cv2
import time
import argparse
import torch
from core import MultiObjectTracker, SpatialAnalytics, StreamVisualizer

def run_pipeline(source: str = "0", model_path: str = "yolov8n.pt", conf_thresh: float = 0.35):
    video_src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(video_src)
    
    if not cap.isOpened():
        print(f"[Error] Cannot open video source: {source}")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[Pipeline] Initializing YOLO + ByteTrack on {device}...")
    
    tracker = MultiObjectTracker(model_path=model_path, conf_thresh=conf_thresh, device=device)
    analytics = SpatialAnalytics()
    visualizer = StreamVisualizer(class_names=tracker.get_classes_map())

    prev_time = time.time()
    fps_smooth = 0.0

    print("[Pipeline] Live stream started. Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Pipeline] End of video stream.")
            break
            
        # Unified Detection + ByteTrack Tracking
        tracked_detections, inference_ms = tracker.track(frame)
        
        # Spatial Telemetry (Tripwires & Zone Intrusion)
        analytics_result = analytics.update(tracked_detections)
        
        # FPS Calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        fps_smooth = (0.9 * fps_smooth) + (0.1 * fps) if fps_smooth > 0 else fps
        
        # Render visual overlays
        annotated_frame = visualizer.annotate(
            frame=frame,
            tracked_detections=tracked_detections,
            analytics_result=analytics_result,
            fps=fps_smooth,
            inference_ms=inference_ms
        )
        
        cv2.imshow("Real-Time Object Detection & Tracking System", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("[Pipeline] Stream closed cleanly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Detection & Tracking")
    parser.add_argument("--source", type=str, default="0", help="Webcam (0) or path to video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO checkpoint")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()
    
    run_pipeline(source=args.source, model_path=args.model, conf_thresh=args.conf)
