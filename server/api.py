"""
FastAPI server to serve an ONNX model for inference.
- Endpoint: POST /predict (multipart/form-data) with field 'file' containing the image
- Returns: JSON {detections: [ {x1,y1,x2,y2,conf,class} ], image: base64 PNG with drawn boxes }

Usage (run server):
    uvicorn server.api:app --host 0.0.0.0 --port 8000 --workers 1

Client example (curl):
    curl -X POST -F "file=@test.jpg" "http://localhost:8000/predict" -v
"""
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path
import io
import base64

import numpy as np
from PIL import Image
import cv2

try:
    import onnxruntime as ort
    HAS_ONNX = True
except Exception:
    HAS_ONNX = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False

app = FastAPI(title='Garbage Classification ONNX Server')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

MODEL_SESSION = None
ULTRALYTICS_MODEL = None

# load ONNX session if path provided via env or default
DEFAULT_ONNX = Path('models/best.onnx')
DEFAULT_PT = Path('training/runs_train/exp_cpu/weights/best.pt')


@app.on_event('startup')
async def load_model():
    global MODEL_SESSION, ULTRALYTICS_MODEL
    # Prefer ONNX model if available
    if DEFAULT_ONNX.exists() and HAS_ONNX:
        MODEL_SESSION = ort.InferenceSession(str(DEFAULT_ONNX), providers=['CPUExecutionProvider'])
        print('Loaded ONNX model:', DEFAULT_ONNX)
    elif DEFAULT_PT.exists() and HAS_ULTRALYTICS:
        ULTRALYTICS_MODEL = YOLO(str(DEFAULT_PT))
        print('Loaded ultralytics weights:', DEFAULT_PT)
    else:
        print('No model loaded. Place ONNX or PyTorch weights at', DEFAULT_ONNX, 'or', DEFAULT_PT)


def preprocess_pil(pil_img, imgsz=416):
    img = np.array(pil_img.convert('RGB'))[:, :, ::-1]  # RGB->BGR
    h0, w0 = img.shape[:2]
    r = imgsz / max(h0, w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    resized = cv2.resize(img, (nw, nh))
    top = (imgsz - nh) // 2
    left = (imgsz - nw) // 2
    padded = cv2.copyMakeBorder(resized, top, imgsz-nh-top, left, imgsz-nw-left, cv2.BORDER_CONSTANT, value=(114,114,114))
    img_input = padded.astype(np.float32)/255.0
    img_input = np.transpose(img_input, (2,0,1))[None, ...]
    return img_input, (r, left, top), (h0, w0)


def draw_boxes_numpy(img_bgr, detections):
    for d in detections:
        x1,y1,x2,y2,conf,c = d
        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
        cv2.rectangle(img_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img_bgr, f'{int(c)} {conf:.2f}', (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return img_bgr


def postprocess_onnx(outputs, conf_thresh=0.35):
    dets = []
    # try common output shapes
    if isinstance(outputs, (list,tuple)) and len(outputs) == 1:
        arr = outputs[0]
        if len(arr.shape) == 2 and arr.shape[1] >= 6:
            for row in arr:
                conf = float(row[4])
                if conf < conf_thresh: continue
                dets.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), conf, float(row[5])])
    return dets


@app.post('/predict')
async def predict(file: UploadFile = File(...), conf: float = 0.35):
    if MODEL_SESSION is None and ULTRALYTICS_MODEL is None:
        return JSONResponse({'error': 'No model loaded on server'}, status_code=500)

    data = await file.read()
    pil_img = Image.open(io.BytesIO(data))

    if MODEL_SESSION is not None:
        try:
            imgsz = 416
            x, resize_info, orig_shape = preprocess_pil(pil_img, imgsz)
            input_name = MODEL_SESSION.get_inputs()[0].name
            out_names = [o.name for o in MODEL_SESSION.get_outputs()]
            res = MODEL_SESSION.run(out_names, {input_name: x.astype(np.float32)})
            dets = postprocess_onnx(res, conf_thresh=conf)
            # map back from padded/resized to original coordinates is model-specific — here we assume outputs are in original coords
        except Exception as e:
            return JSONResponse({'error': f'ONNX inference failed: {e}'}, status_code=500)

    else:
        # use ultralytics
        results = ULTRALYTICS_MODEL.predict(pil_img, device='cpu', imgsz=416, conf=conf)
        dets = []
        for box in results[0].boxes:
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            dets.append([x1,y1,x2,y2,float(box.conf),int(box.cls)])

    # create return image with boxes
    img_bgr = cv2.cvtColor(np.array(pil_img.convert('RGB'))[:,:, ::-1], cv2.COLOR_BGR2RGB) # ensure BGR -> convert back
    img_bgr = draw_boxes_numpy(img_bgr, dets)

    # encode image as base64 png
    _, buf = cv2.imencode('.png', img_bgr)
    b64 = base64.b64encode(buf).decode('utf-8')

    return JSONResponse({'detections': dets, 'image_b64': b64})
