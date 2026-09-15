import torch
from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any, Optional

class ObjectDetector:
    """
    Production-grade YOLO detector with hardware auto-detection (CUDA / CPU),
    half-precision (FP16) acceleration, and customizable class filtering.
    """
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: Optional[str] = None,
        classes: Optional[List[int]] = None
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        
        # Hardware acceleration resolution
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[Detector] Initializing {model_path} on device: {self.device}")
        self.model = YOLO(model_path)
        
        # Move model to target device
        self.model.to(self.device)
        self.is_cuda = "cuda" in self.device
        
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Runs inference on a single BGR image frame (from OpenCV).
        
        Returns:
            Dict containing:
            - 'boxes': Nx4 array of bounding boxes [x1, y1, x2, y2]
            - 'confidences': N array of confidence scores [0.0 - 1.0]
            - 'class_ids': N array of integer class IDs
            - 'class_names': list of string labels
            - 'inference_ms': inference time in milliseconds
        """
        # Run inference with FP16 half precision enabled on CUDA for max FPS
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            half=self.is_cuda,
            verbose=False
        )
        
        result = results[0]
        boxes_data = result.boxes
        
        if boxes_data is None or len(boxes_data) == 0:
            return {
                "boxes": np.empty((0, 4), dtype=np.float32),
                "confidences": np.empty((0,), dtype=np.float32),
                "class_ids": np.empty((0,), dtype=np.int32),
                "class_names": [],
                "inference_ms": result.speed.get("inference", 0.0)
            }
            
        boxes = boxes_data.xyxy.cpu().numpy()
        confidences = boxes_data.conf.cpu().numpy()
        class_ids = boxes_data.cls.cpu().numpy().astype(int)
        class_names = [self.model.names[cid] for cid in class_ids]
        
        return {
            "boxes": boxes,
            "confidences": confidences,
            "class_ids": class_ids,
            "class_names": class_names,
            "inference_ms": result.speed.get("inference", 0.0)
        }

    def get_classes_map(self) -> Dict[int, str]:
        """Returns the dictionary mapping class IDs to class names."""
        return self.model.names
