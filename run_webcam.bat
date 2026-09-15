@echo off
echo ==========================================================
echo  Starting Real-Time OpenCV Camera Stream (Tuned Sensitivity)
echo ==========================================================
echo Running YOLOv8 Medium with Optimized 0.30 Confidence...
.\venv\Scripts\python.exe main.py --source 0 --model yolov8m.pt --conf 0.30
pause
