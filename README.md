Vehicle Detection, Classification & Speed Estimation
Real-time computer vision system that detects moving vehicles (cars, trucks, buses, motorcycles)
from live/recorded traffic camera feeds, classifies their type, and estimates their speed — built
for smart traffic monitoring and surveillance use cases.


Overview:

This project processes video streams from traffic cameras (CCTV, dashcams, or live RTSP feeds) to:
Detect moving vehicles in each frame
Classify them into categories — car, truck, bus, motorcycle, etc.
Track each vehicle across frames using a unique ID
Estimate speed (km/h) by measuring how far a vehicle travels between frames, calibrated against real-world distance
Flag overspeeding vehicles against a configurable speed limit
Log and visualize results (CSV export + on-screen overlay + optional dashboard)
Designed to work on both recorded footage and live camera feeds, making it suitable for smart city / ITS (Intelligent Transport System) applications.


 Features:

✅ Real-time vehicle detection using YOLOv8
✅ Multi-class classification (car / truck / bus / motorcycle / auto-rickshaw)
✅ Object tracking with persistent IDs (ByteTrack / DeepSORT)
✅ Speed estimation via pixel-to-real-world calibration
✅ Overspeeding alert system with configurable threshold
✅ Live annotated video output (bounding boxes, class label, speed, ID)
✅ CSV/JSON logging of every detected vehicle event
✅ Works on video files, webcam, and RTSP live streams
✅ (Optional) Streamlit dashboard for live analytics


How It Works:

Detection — Each frame is passed through a YOLOv8 model fine-tuned/pretrained on vehicle classes (COCO-based, optionally fine-tuned on Indian traffic data for auto-rickshaws etc.)
Tracking — Detected boxes are linked across frames using a tracker (ByteTrack), assigning a consistent ID to each vehicle so it isn't re-counted.
Speed Estimation —

The camera view is calibrated once by mapping known real-world reference points (e.g. lane markings, a fixed distance on the road) to pixel coordinates using a perspective transform.
The vehicle's centroid position is tracked frame-to-frame; using the frame rate (FPS) and the calibrated pixel-to-meter ratio, distance travelled is converted into speed (km/h).
A rolling average over several frames smooths out noise.



Output — Each vehicle is annotated with its class, tracking ID, and estimated speed, and the event is logged.
