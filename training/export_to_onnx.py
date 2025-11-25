"""
Export a trained YOLOv8 model (best.pt) to ONNX format using ultralytics.YOLO
Requirements: ultralytics installed and best.pt available at training/runs_train/<exp>/weights/best.pt

Usage:
    python export_to_onnx.py --weights runs_train/exp_cpu/weights/best.pt --output models/best.onnx --imgsz 416

This script ensures a reproducible export path and prints the exported file size.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def export(weights: str, out: str, imgsz: int = 416, opset: int = 12):
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {weights_path} → {out_path} (imgsz={imgsz}, opset={opset})")
    model = YOLO(str(weights_path))
    model.export(format='onnx', imgsz=imgsz, opset=opset, verbose=True, save=True, file=out_path.name, dir=str(out_path.parent))

    size_mb = out_path.stat().st_size / (1024*1024)
    print(f"Export complete: {out_path} ({size_mb:.2f} MB)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs_train/exp/weights/best.pt', help='Path to best.pt')
    parser.add_argument('--output', type=str, default='models/best.onnx', help='Output onnx path')
    parser.add_argument('--imgsz', type=int, default=416, help='Image size')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset to use')
    args = parser.parse_args()
    export(args.weights, args.output, args.imgsz, args.opset)
