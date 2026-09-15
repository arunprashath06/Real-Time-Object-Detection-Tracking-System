import cv2
import numpy as np
import os

def create_sample_video(path="data/samples/sample_traffic.mp4", duration_sec=10, fps=30):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    total_frames = duration_sec * fps
    
    cars = [
        {"x": 150, "y": 100, "vx": 4, "vy": 3, "color": (40, 40, 220), "size": (120, 60), "label": "Car A"},
        {"x": 1100, "y": 600, "vx": -5, "vy": -3, "color": (220, 100, 40), "size": (130, 70), "label": "Car B"},
        {"x": 450, "y": 50, "vx": 1, "vy": 4, "color": (50, 200, 50), "size": (90, 50), "label": "Van"},
        {"x": 900, "y": 650, "vx": -3, "vy": -4, "color": (200, 50, 200), "size": (110, 55), "label": "Sedan"}
    ]
    
    for f in range(total_frames):
        # Road asphalt background
        frame = np.full((height, width, 3), 45, dtype=np.uint8)
        
        # Road lane markings
        for y in range(0, height, 40):
            cv2.line(frame, (width // 2, y), (width // 2, y + 20), (220, 220, 220), 3)
            
        cv2.line(frame, (100, 0), (100, height), (200, 200, 50), 3)
        cv2.line(frame, (width - 100, 0), (width - 100, height), (200, 200, 50), 3)
        
        for c in cars:
            c["x"] += c["vx"]
            c["y"] += c["vy"]
            
            # wrap around borders for continuous motion
            if c["y"] > height: c["y"] = -c["size"][1]
            if c["y"] < -c["size"][1]: c["y"] = height
            if c["x"] > width - 150: c["vx"] *= -1
            if c["x"] < 150: c["vx"] *= -1
            
            x, y = int(c["x"]), int(c["y"])
            w, h = c["size"]
            
            # Draw car body
            cv2.rectangle(frame, (x, y), (x + w, y + h), c["color"], -1)
            cv2.rectangle(frame, (x + 10, y + 10), (x + w - 10, y + h - 10), (20, 20, 20), -1) # roof/windshield
            cv2.putText(frame, c["label"], (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        out.write(frame)
        
    out.release()
    print(f"Sample demo video created at {path}")

if __name__ == "__main__":
    create_sample_video()
