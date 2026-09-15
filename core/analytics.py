import numpy as np
import supervision as sv
from typing import Dict, Any, List, Tuple, Optional

class SpatialAnalytics:
    """
    Spatial intelligence layer calculating:
    1. Directional Virtual Line Crossing (e.g. Entry/Exit traffic counting)
    2. Restricted Polygon Zone Intrusion (e.g. Industrial / Robotic safety boundaries)
    3. Trajectory and Velocity telemetry
    """
    def __init__(
        self,
        line_start: Tuple[int, int] = (100, 360),
        line_end: Tuple[int, int] = (1180, 360),
        zone_polygon: Optional[np.ndarray] = None
    ):
        self.line_start = sv.Point(x=line_start[0], y=line_start[1])
        self.line_end = sv.Point(x=line_end[0], y=line_end[1])
        
        # Line counter for bi-directional crossing
        self.line_zone = sv.LineZone(start=self.line_start, end=self.line_end)
        
        # Default safety polygon if none provided
        if zone_polygon is None:
            self.zone_polygon = np.array([
                [400, 200],
                [880, 200],
                [880, 520],
                [400, 520]
            ], dtype=np.int32)
        else:
            self.zone_polygon = np.array(zone_polygon, dtype=np.int32)
            
        self.polygon_zone = sv.PolygonZone(polygon=self.zone_polygon)
        
    def update(self, tracked_detections: sv.Detections) -> Dict[str, Any]:
        """
        Updates line crossing counts and polygon zone intrusions with active detections.
        """
        # Trigger line zone state
        cross_in, cross_out = self.line_zone.trigger(detections=tracked_detections)
        
        # Trigger polygon zone state
        zone_mask = self.polygon_zone.trigger(detections=tracked_detections)
        intruding_count = int(np.sum(zone_mask))
        
        # Extract intruding track IDs
        intruding_ids = []
        if tracked_detections.tracker_id is not None and len(zone_mask) > 0:
            for idx, is_inside in enumerate(zone_mask):
                if is_inside:
                    intruding_ids.append(int(tracked_detections.tracker_id[idx]))
                    
        return {
            "in_count": self.line_zone.in_count,
            "out_count": self.line_zone.out_count,
            "total_crossings": self.line_zone.in_count + self.line_zone.out_count,
            "zone_intrusion_active": intruding_count > 0,
            "zone_occupancy": intruding_count,
            "intruding_ids": intruding_ids,
            "zone_mask": zone_mask
        }
        
    def reset(self):
        """Resets counters."""
        self.line_zone.in_count = 0
        self.line_zone.out_count = 0
