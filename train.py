# train.py
from ultralytics import YOLO
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data.yaml', help='path to data yaml')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='base model weights (e.g. yolo11n.pt)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--project', type=str, default='runs/train/smoke_fire')
    parser.add_argument('--name', type=str, default='exp')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.project, exist_ok=True)
    # Create model from pretrained weights (or a model string)
    model = YOLO(args.model)

    # Train
    model.train(data=args.data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                project=args.project,
                name=args.name,
                workers=8,
                optimizer='Adam')  # try Adam or SGD

if __name__ == '__main__':
    main()
