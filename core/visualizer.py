import cv2
import numpy as np
import supervision as sv
from typing import Dict, Any, List

class StreamVisualizer:
    """
    Renders bounding boxes, persistent track IDs,
    motion trajectory trails, safety zones, and a modern heads-up telemetry HUD.
    """
    def __init__(self, class_names: Dict[int, str]):
        self.class_names = class_names
        
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)
        
    def annotate(
        self,
        frame: np.ndarray,
        tracked_detections: sv.Detections,
        analytics_result: Dict[str, Any],
        fps: float,
        inference_ms: float
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        
        # Draw detections if any exist
        if len(tracked_detections) > 0:
            if tracked_detections.tracker_id is not None:
                annotated_frame = self.trace_annotator.annotate(
                    scene=annotated_frame,
                    detections=tracked_detections
                )
            
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame,
                detections=tracked_detections
            )
            
            labels = []
            for item in tracked_detections:
                xyxy = item[0]
                conf = item[2]
                class_id = item[3]
                tracker_id = item[4]
                name = self.class_names.get(class_id, str(class_id))
                if tracker_id is not None:
                    labels.append(f"#{tracker_id} {name} {conf:.2f}")
                else:
                    labels.append(f"{name} {conf:.2f}")
                    
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame,
                detections=tracked_detections,
                labels=labels
            )
            
        # Draw restricted safety zone polygon
        zone_color = (0, 0, 255) if analytics_result.get("zone_intrusion_active", False) else (0, 255, 0)
        overlay = annotated_frame.copy()
        zone_poly = np.array([[400, 200], [880, 200], [880, 520], [400, 520]], dtype=np.int32)
        cv2.fillPoly(overlay, [zone_poly], zone_color)
        cv2.addWeighted(overlay, 0.25, annotated_frame, 0.75, 0, annotated_frame)
        cv2.polylines(annotated_frame, [zone_poly], isClosed=True, color=zone_color, thickness=2)
        
        # Draw Virtual Tripwire Line
        cv2.line(annotated_frame, (100, 360), (1180, 360), (255, 200, 0), 2)
        cv2.putText(annotated_frame, "Tripwire Boundary", (110, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        
        # Heads-Up Display (HUD) Banner
        h, w = annotated_frame.shape[:2]
        hud_bg = np.zeros((70, w, 3), dtype=np.uint8)
        cv2.addWeighted(annotated_frame[0:70, 0:w], 0.3, hud_bg, 0.7, 0, annotated_frame[0:70, 0:w])
        
        # Telemetry Text
        fps_text = f"FPS: {fps:.1f} ({inference_ms:.1f}ms)"
        in_out_text = f"Crossings: In {analytics_result.get('in_count', 0)} | Out {analytics_result.get('out_count', 0)}"
        zone_text = f"Zone Intrusion: {'[ALERT!]' if analytics_result.get('zone_intrusion_active') else 'CLEAR'}"
        zone_color_txt = (0, 0, 255) if analytics_result.get('zone_intrusion_active') else (0, 255, 0)
        
        cv2.putText(annotated_frame, "Real-Time AI Vision Pipeline", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated_frame, fps_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
        cv2.putText(annotated_frame, in_out_text, (380, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated_frame, zone_text, (750, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, zone_color_txt, 2)
        
        return annotated_frame
