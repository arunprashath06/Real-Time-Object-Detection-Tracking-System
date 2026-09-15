import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import numpy as np
from core.detector import ObjectDetector

def run_benchmark(num_warmup=15, num_iterations=50, resolutions=[(640, 640), (1280, 720)]):
    print("=" * 60)
    print("🚀 HARDWARE INFERENCE & LATENCY BENCHMARK")
    print("=" * 60)
    
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device Target: {device_name}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    detector = ObjectDetector(model_path="yolov8n.pt")
    
    for (w, h) in resolutions:
        print(f"\n--- Benchmarking Resolution: {w}x{h} ---")
        dummy_frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        
        # Warmup passes
        for _ in range(num_warmup):
            _ = detector.detect(dummy_frame)
            
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = detector.detect(dummy_frame)
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)
            
        latencies = np.array(latencies)
        avg_latency = np.mean(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        fps = 1000.0 / avg_latency
        
        print(f"  • Average Latency : {avg_latency:.2f} ms")
        print(f"  • Throughput (FPS): {fps:.1f} FPS")
        print(f"  • P50 Latency     : {p50:.2f} ms")
        print(f"  • P95 Latency     : {p95:.2f} ms")
        print(f"  • P99 Latency     : {p99:.2f} ms")
        
    print("\nBenchmark completed successfully.")

if __name__ == "__main__":
    run_benchmark()
