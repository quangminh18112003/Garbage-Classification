# Training Guide — Garbage-Classification (YOLOv8)

This document explains the training method we used for this project, step-by-step instructions to train, how training is organized, and practical advice to make inference real-time (webcam) if needed.

---

## 1. Overview — method and setup

- Model family: YOLOv8 (Ultralytics implementation)
- Model variant used for initial experiments: `yolov8n.pt` (Nano) — small and fast, good for CPU and edge devices
- Task: Object detection (3 classes: `glass`, `biological`, `plastic`)
- Dataset structure: `dataset_split/` with `images/train`, `images/val`, `images/test` and corresponding `labels/*` (YOLO format)
- Training script: `training/train.py` (uses ultralytics.YOLO API)
- data config: `training/data.yaml` (points to dataset relative path and names/nc)

Why YOLOv8 Nano?
- Low parameter count → faster inference and less memory pressure
- Easier to run on CPU for development and demo


## 2. Training settings used in this repo (current configuration)

Main choices (matching `training/train.py`):
- `model`: `yolov8n.pt` (pretrained seed weights)
- `epochs`: 50 (reduced for CPU; can be increased on GPU)
- `imgsz`: 416 (reduces compute & memory vs 640)
- `batch`: 4
- `workers`: 0 (safer for CPU systems)
- `device`: `cpu` (explicit for machines without GPU)
- Data augmentation: mild — hsv, small rotate/translate/scale, mosaic reduced, mixup off (for CPU runs)
- Seed: 666 (reproducibility)
- Early stopping/patience: 20

These settings are conservative for CPU training. On GPU you should increase image size, batch size and epochs for better accuracy.


## 3. How training works (high level)

1. The training script loads `yolov8n.pt` base model and overrides the number of classes using `nc` in `data.yaml`.
2. The dataset is read in YOLO format (images + `.txt` labels per image). The trainer automatically builds dataloaders for train/val.
3. Pretrained backbone weights are transferred to the detection head; some layers can be frozen depending on configuration.
4. The training loop computes losses (box, class, dfl) and performs optimizer steps to update weights.
5. Validation runs each epoch and training outputs metrics like mAP@50, mAP@50-95, precision/recall.
6. Best weights (and checkpoint per epoch) are saved in `training/runs_train/<exp>/weights/`.


## 4. Step-by-step: Training locally (CPU) — runnable commands

Open a terminal and use your Python environment (use a venv if possible):

```powershell
cd c:\python\Phanloairac
# Install requirements if not already
pip install -r requirements.txt

# Run training (already configured in training/train.py)
cd training
python train.py
```

Training output and artifacts appear in:
- `training/runs_train/<exp_name>/` — contains `weights/` (`best.pt`) and `results.csv` and plots


## 5. Recording & versioning training results

We include a small helper `commit_results.py` that creates `TRAINING_RESULTS.md` summarizing the latest `runs_train/exp*` results. Workflow we suggest:

```powershell
# after training completes
python commit_results.py
# review changes
git add TRAINING_RESULTS.md
git commit -m "train: completed <exp_name> — add metrics summary"
git push origin main
```

If your weights are < 100MB you can include them in the repo. For larger weights use Git LFS.


## 6. Inference / Realtime detection with camera

Short answer: Yes — inference (detection) is real-time capable, but training itself is offline. The model once trained can be used for real-time webcam detection.

Key considerations for realtime inference:
- Use a lightweight model (YOLOv8n or yolov8s) to reduce latency
- Run inference on a GPU where available (preferred) — device `0` in `ultralytics` (replace `device='cpu'` with `device='0'`)
- Export and optimize model for deployment (ONNX, TensorRT, OpenVINO) for lower latency on dedicated hardware
- Reduce input resolution (e.g., 416 or 320) for faster per-frame inference at the cost of some accuracy
- Run inference in a loop with batch size 1 and use asynchronous IO for webcam capture + model inference

Example inference with ultralytics (simple loop):

```python
from ultralytics import YOLO
model = YOLO('runs_train/exp/weights/best.pt')
# For webcam (device 0):
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # inference per frame
    res = model.predict(frame, device='cpu', conf=0.35, imgsz=416)
    # render & display
    out = res[0].plot()
    cv2.imshow('webcam', out[:, :, ::-1])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

Important: run inference with `device='0'` to use GPU if available.


## 7. Ways to get faster / lower latency inference (practical checklist)

If you need real-time (> 15–30 FPS):
- Run on GPU (CUDA) with optimized drivers
- Export model to ONNX and run with ONNXRuntime with GPU/GPU-TensorRT
- Use TensorRT (NVIDIA) or OpenVINO (Intel) for inference acceleration
- Reduce model size or use quantization (8-bit) to speed up
- Lower `imgsz` and avoid heavy post-processing

Example of export + ONNXRuntime:
```powershell
# In Python
from ultralytics import YOLO
model = YOLO('runs_train/exp/weights/best.pt')
model.export(format='onnx', imgsz=416)
# Run ONNXRuntime inference (separate script)
```


## 8. Streamlit integration — realtime experience

The current `GiaoDien/app.py` supports:
- Upload image inference
- Webcam live detection via helper method `helper.play_webcam()`

If you need low-latency webcam detection on Streamlit:
- Run the model inference loop in `helper.play_webcam()` using a lightweight model and GPU
- Use `st.camera_input` or a streaming approach where you send single frames for inference
- Consider moving inference outside Streamlit (serve model via a fast API: FastAPI + GPU) and have Streamlit display outputs — this separates UI and heavy compute and makes streaming smoother.


## 9. Recommendations for production and mobile/edge

- If running on embedded/edge (Jetson, Coral): choose compatible model and convert to appropriate format
- For CPU server: use OpenVINO (Intel) or ONNX + OpenVINO for speedups
- For NVIDIA GPU server: use TensorRT or ONNX with TensorRT provider
- Consider model pruning / quantization to trade some accuracy for speed


## 10. Next steps we can do for you (pick one):
1. Add a small example script `inference/webcam_realtime.py` that uses ONNXRuntime for inference and shows FPS.
2. Add a FastAPI wrapper and Streamlit demo that queries the model server for extremely responsive webcam inference.
3. Add TensorRT/ONNX export & sample scripts for your machine (if you have GPU access).

---

If you'd like, I'll add this file (`TRAINING_GUIDE.md`) to the repository and update `README.md` to link to it. Also I can create an example realtime inference script next — which option do you prefer from the next steps above?