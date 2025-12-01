# detect_rtsp.py
import cv2
import time
import os
from collections import deque
from ultralytics import YOLO
import numpy as np

# ---------- CONFIG ----------
WEIGHTS = "runs/train/smoke_fire/exp2/weights/best.pt"  # update after training
RTSP_OR_VIDEO = "videos/Fire_test.mp4"  # or path to video file
OUTPUT_DIR = "detections_snapshots"
SAVE_SNAPSHOT = True
DISPLAY_WINDOW = True  # set False for headless servers
ALERT_SECONDS = 3.0  # seconds a detection must persist to trigger alert
CONF_THRESH = 0.35
IOU_THRESH = 0.25
MAX_HISTORY = 30  # for persistence checks
FPS_CALC_AVG = 0.9

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- helper ----------
def draw_boxes(frame, boxes, scores, cls_ids, class_names):
    for (x1,y1,x2,y2), s, cid in zip(boxes, scores, cls_ids):
        label = f"{class_names[int(cid)]} {s:.2f}"
        # box
        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 2)
        # label background
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (int(x1), int(y1)-h_text-6), (int(x1)+w_text, int(y1)), (0,0,255), -1)
        cv2.putText(frame, label, (int(x1), int(y1)-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

# ---------- main ----------
def main():
    model = YOLO(WEIGHTS)
    # load class names from model (or fallback)
    try:
        class_names = model.model.names
    except Exception:
        class_names = {0: 'smoke', 1: 'fire'}

    # history of detection timestamps, one deque per class id
    detect_history = {0: deque(maxlen=MAX_HISTORY), 1: deque(maxlen=MAX_HISTORY)}

    cap = cv2.VideoCapture(RTSP_OR_VIDEO)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open stream: {RTSP_OR_VIDEO}")

    # For FPS smoothing
    fps = 0.0
    t_prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or cannot fetch frame.")
            break

        # Ultralytics supports streaming to process batches efficiently.
        # But for simple real-time usage, we predict per-frame.
        results = model.predict(source=[frame], conf=CONF_THRESH, iou=IOU_THRESH, max_det=10, device='cuda:0', verbose=False)
        # results is a list with one element per source
        r = results[0]

        # r.boxes.xyxy: tensor Nx4, r.boxes.conf, r.boxes.cls
        boxes = []
        scores = []
        cls_ids = []
        if hasattr(r, 'boxes') and len(r.boxes) > 0:
            for box in r.boxes:
                # xyxy is (1, 4) tensor -> flatten to 4 values
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cid = int(box.cls[0].cpu().numpy())

                boxes.append((x1, y1, x2, y2))
                scores.append(conf)
                cls_ids.append(cid)

                # update history
                detect_history.setdefault(cid, deque(maxlen=MAX_HISTORY)).append(time.time())


        # draw detections
        draw_boxes(frame, boxes, scores, cls_ids, class_names)

        # check persisted detection for alert
        alerting = False
        alert_classes = []
        now = time.time()
        for cid, dq in detect_history.items():
            # if any timestamp within last ALERT_SECONDS
            if len(dq) > 0:
                # check streak: any timestamp >= now - ALERT_SECONDS
                if any(t >= now - ALERT_SECONDS for t in dq):
                    alerting = True
                    alert_classes.append(class_names.get(cid, str(cid)))

        if alerting:
            cv2.putText(frame, f"ALERT: {','.join(alert_classes)}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
            if SAVE_SNAPSHOT:
                fname = os.path.join(OUTPUT_DIR, f"alert_{int(now)}_{','.join(alert_classes)}.jpg")
                cv2.imwrite(fname, frame)
                print("Saved alert snapshot:", fname)
                # Optional: integrate with email/SMS/HTTP webhook here

        # FPS
        t_now = time.time()
        dt = t_now - t_prev
        t_prev = t_now
        if dt > 0:
            fps = FPS_CALC_AVG * fps + (1.0 - FPS_CALC_AVG) * (1.0 / dt)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20,frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        if DISPLAY_WINDOW:
            cv2.imshow("Smoke & Fire Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if DISPLAY_WINDOW:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
