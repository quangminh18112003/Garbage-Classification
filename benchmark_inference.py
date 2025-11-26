"""
benchmark_inference.py
---------------------
Benchmark ONNX and PyTorch model inference speed (FPS) on sample images.
Logs results to benchmark_results.txt.

Usage:
    python benchmark_inference.py --model_type [onnx|pt] --model_path <path> --image_dir <dir> [--device cpu|cuda]

Example:
    python benchmark_inference.py --model_type onnx --model_path training/best.onnx --image_dir dataset_split/images/test
    python benchmark_inference.py --model_type pt --model_path training/best.pt --image_dir dataset_split/images/test --device cpu
"""
import argparse
import os
import time
import glob
import numpy as np
from PIL import Image

RESULTS_FILE = "benchmark_results.txt"

def load_images(image_dir, max_images=100):
    exts = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
    files = files[:max_images]
    images = []
    for f in files:
        img = Image.open(f).convert("RGB")
        images.append(np.array(img))
    return images

def benchmark_onnx(model_path, images):
    import onnxruntime as ort
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    # Assume YOLOv8 input size
    def preprocess(img):
        img = np.array(img)
        img = np.resize(img, (640, 640, 3))
        img = img.transpose(2, 0, 1)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)
        return img
    times = []
    for img in images:
        inp = preprocess(img)
        start = time.time()
        session.run(None, {input_name: inp})
        times.append(time.time() - start)
    return times

def benchmark_pt(model_path, images, device="cpu"):
    import torch
    from ultralytics import YOLO
    model = YOLO(model_path)
    model.to(device)
    def preprocess(img):
        img = np.array(img)
        img = np.resize(img, (640, 640, 3))
        img = img.transpose(2, 0, 1)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)
        return torch.from_numpy(img).to(device)
    times = []
    for img in images:
        inp = preprocess(img)
        start = time.time()
        _ = model(inp)
        times.append(time.time() - start)
    return times

def log_results(model_type, model_path, avg_fps, times):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"Model: {model_path}\nType: {model_type}\nAvg FPS: {avg_fps:.2f}\nTimes: {','.join([f'{t:.4f}' for t in times])}\n{'-'*40}\n")

def main():
    parser = argparse.ArgumentParser(description="Benchmark model inference FPS.")
    parser.add_argument("--model_type", choices=["onnx", "pt"], required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    images = load_images(args.image_dir)
    if not images:
        print("No images found.")
        return
    if args.model_type == "onnx":
        times = benchmark_onnx(args.model_path, images)
    else:
        times = benchmark_pt(args.model_path, images, args.device)
    avg_fps = len(times) / sum(times)
    print(f"Avg FPS: {avg_fps:.2f}")
    log_results(args.model_type, args.model_path, avg_fps, times)
    print(f"Results logged to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
