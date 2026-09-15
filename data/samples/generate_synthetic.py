import cv2
import numpy as np

def generate_sample_video(filename="data/samples/traffic_sim.mp4", num_frames=300, width=1280, height=720):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
    
    np.random.seed(42)
    # Generate 4 moving objects (representing cars/people)
    objects = [
        {'pos': [100, 200], 'vel': [4, 1], 'color': (0, 0, 255), 'size': (80, 50), 'label': 'Car 1'},
        {'pos': [1100, 450], 'vel': [-5, -1], 'color': (255, 0, 0), 'size': (90, 55), 'label': 'Car 2'},
        {'pos': [600, 100], 'vel': [0, 3], 'color': (0, 255, 0), 'size': (40, 40), 'label': 'Pedestrian'},
        {'pos': [200, 600], 'vel': [3, -2], 'color': (255, 255, 0), 'size': (75, 45), 'label': 'Car 3'},
    ]
    
    for f in range(num_frames):
        frame = np.full((height, width, 3), 35, dtype=np.uint8)
        
        # Draw road / background markers
        cv2.line(frame, (0, 360), (width, 360), (70, 70, 70), 2)
        cv2.line(frame, (0, 400), (width, 400), (70, 70, 70), 2)
        
        for obj in objects:
            obj['pos'][0] += obj['vel'][0]
            obj['pos'][1] += obj['vel'][1]
            
            # Bounce at borders
            if obj['pos'][0] <= 50 or obj['pos'][0] >= width - 100:
                obj['vel'][0] *= -1
            if obj['pos'][1] <= 50 or obj['pos'][1] >= height - 100:
                obj['vel'][1] *= -1
                
            x, y = int(obj['pos'][0]), int(obj['pos'][1])
            w, h = obj['size']
            cv2.rectangle(frame, (x, y), (x + w, y + h), obj['color'], -1)
            cv2.putText(frame, obj['label'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        out.write(frame)
        
    out.release()
    print(f"Sample generated at {filename}")

if __name__ == "__main__":
    import os
    os.makedirs("data/samples", exist_ok=True)
    # We will run this once opencv is installed
