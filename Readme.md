# Smoke & Fire Detection System

Real-time smoke and fire detection using YOLOv11 and OpenCV. Monitors RTSP camera feeds or video files with persistent alert mechanisms.

**Dataset**: [Smoke & Fire Detection YOLO Dataset](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo)

## Requirements

```bash
pip install ultralytics opencv-python numpy
```

GPU with CUDA support recommended for training.

## Quick Start

### 1. Download Dataset

```bash
kaggle datasets download -d sayedgamal99/smoke-fire-detection-yolo
unzip smoke-fire-detection-yolo.zip -d data/
```

### 2. Train Model

```bash
# Basic training
python train.py

# Custom parameters
python train.py --epochs 100 --batch 16 --device cuda:0
```

### 3. Run Detection

```bash
python detect_rtsp.py
```

Configure in `detect_rtsp.py`:
```python
WEIGHTS = "runs/train/smoke_fire/exp2/weights/best.pt"
RTSP_OR_VIDEO = "videos/Fire_test.mp4"  # or "rtsp://camera_ip/stream"
ALERT_SECONDS = 3.0      # Alert persistence threshold
CONF_THRESH = 0.35       # Confidence threshold
```

## Key Features

- Detects smoke and fire in real-time video streams
- Triggers alerts when detections persist for configurable duration
- Saves snapshots of alert events
- Displays FPS and bounding boxes with confidence scores
- Supports RTSP cameras and video files

## Configuration

Edit `detect_rtsp.py` to customize:
- `SAVE_SNAPSHOT`: Save alert images (default: True)
- `DISPLAY_WINDOW`: Show video window (default: True, set False for headless)
- `ALERT_SECONDS`: Detection persistence for alerts (default: 3.0)
- `CONF_THRESH`: Detection confidence threshold (default: 0.35)

## Output

- Trained weights: `runs/train/smoke_fire/exp/weights/best.pt`
- Alert snapshots: `detections_snapshots/alert_{timestamp}_{class}.jpg`

## Credits

Dataset by [Sayed Gamal](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo)