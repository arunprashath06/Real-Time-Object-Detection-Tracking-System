import numpy as np
from ultralytics import YOLO
from typing import Dict, Any, List, Optional
import supervision as sv

class MultiObjectTracker:
    """
    High-performance Multi-Object Tracker powered natively by YOLOv8 + ByteTrack.
    Preserves persistent track IDs across frame occlusions and records motion trajectories.
    """
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.5,
        device: str = "cuda:0"
    ):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        
        # Motion history: {track_id: [(x, y), ...]}
        self.trajectories: Dict[int, List[tuple]] = {}
        self.max_trajectory_length = 30

    def track(self, frame: np.ndarray) -> tuple[sv.Detections, float]:
        """
        Runs native ByteTrack tracking on a BGR video frame.
        Returns:
            (supervision.Detections, inference_ms)
        """
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False
        )
        
        result = results[0]
        inference_ms = result.speed.get("inference", 0.0)
        boxes_data = result.boxes
        
        if boxes_data is None or len(boxes_data) == 0 or boxes_data.id is None:
            return sv.Detections.empty(), inference_ms
            
        boxes = boxes_data.xyxy.cpu().numpy()
        confidences = boxes_data.conf.cpu().numpy()
        class_ids = boxes_data.cls.cpu().numpy().astype(int)
        tracker_ids = boxes_data.id.cpu().numpy().astype(int)
        
        sv_detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids,
            tracker_id=tracker_ids
        )
        
        # Record centroids into trajectories
        for idx, track_id in enumerate(tracker_ids):
            track_id = int(track_id)
            xyxy = boxes[idx]
            cx = int((xyxy[0] + xyxy[2]) / 2.0)
            cy = int((xyxy[1] + xyxy[3]) / 2.0)
            
            if track_id not in self.trajectories:
                self.trajectories[track_id] = []
            self.trajectories[track_id].append((cx, cy))
            
            if len(self.trajectories[track_id]) > self.max_trajectory_length:
                self.trajectories[track_id].pop(0)
                
        return sv_detections, inference_ms

    def get_classes_map(self) -> Dict[int, str]:
        return self.model.names
