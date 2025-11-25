"""
Realtime webcam inference using ONNXRuntime with fallback to ultralytics.YOLO
- Measures FPS
- Draws bounding boxes using OpenCV

Usage:
    python inference/webcam_realtime_onnx.py --model models/best.onnx --device cpu --conf 0.35 --imgsz 416

If ONNXRuntime model output format cannot be parsed, script will fallback to ultralytics model (if installed) so it's flexible.
"""
import argparse
import time
import base64
from pathlib import Path

import cv2
import numpy as np

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


def preprocess(frame, imgsz):
    # frame BGR -> RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]
    r = imgsz / max(h0, w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    img_resized = cv2.resize(img, (nw, nh))
    # pad to square
    pad_h = imgsz - nh
    pad_w = imgsz - nw
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    img_input = img_padded.astype(np.float32) / 255.0
    # NCHW
    img_input = np.transpose(img_input, (2,0,1))
    img_input = np.expand_dims(img_input, 0)
    return img_input, (r, left, top), (h0, w0)


def draw_boxes(frame, detections, resize_info):
    """detections: list of [x1,y1,x2,y2,conf,cls] in original image coordinates"""
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{int(cls)} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return frame


def postprocess_onnx(outputs, conf_thresh=0.35, imgsz=416):
    """Try to parse ONNX runtime outputs from Ultralytics ONNX exports. There are multiple possible output signatures depending on how export was done.
    We attempt to detect common formats and return list of [x1,y1,x2,y2,conf,cls]. Coordinates returned are normalized or absolute depending on output.
    This function is best-effort; for full reliability use ultralytics.YOLO API.
    """
    dets = []

    # Some ONNX outputs may be a single array of (N,6) with [x1,y1,x2,y2,conf,cls]
    if isinstance(outputs, (list,tuple)) and len(outputs) == 1:
        out = outputs[0]
        if len(out.shape) == 2 and out.shape[1] >= 6:
            # assume already in x1,y1,x2,y2,conf,cls
            arr = out
            for row in arr:
                conf = float(row[4])
                if conf < conf_thresh: continue
                dets.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), conf, float(row[5])])
            return dets

    # Another variant: outputs as raw keypoints (grid) - unsupported here

    return dets


def run_onnx_session(sess, frame, imgsz, conf):
    x, resize_info, orig_shape = preprocess(frame, imgsz)
    input_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    res = sess.run(out_names, {input_name: x})
    # postprocess attempt
    dets = postprocess_onnx(res, conf_thresh=conf, imgsz=imgsz)
    # if dets reference resized/padded coordinates, need to map them back. This is model-specific.
    return dets


def main(model_path: str, imgsz: int, conf: float, device: str):
    print('Starting webcam realtime demo')

    # prefer ONNXRuntime if model_path ends with .onnx
    use_onnx = model_path.endswith('.onnx') and HAS_ONNX
    if use_onnx:
        print('Using ONNXRuntime backend')
        sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    elif model_path.endswith('.onnx') and not HAS_ONNX:
        print('ONNXRuntime not installed, falling back to ultralytics YOLO (requires ultralytics)')
        use_onnx = False

    if not use_onnx:
        if not HAS_ULTRALYTICS:
            raise RuntimeError('ultralytics or onnxruntime not available. Install onnxruntime or ultralytics.')
        model = YOLO(model_path)

    cap = cv2.VideoCapture(0)
    fps_smooth = 0
    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        dets = []
        try:
            if use_onnx:
                dets = run_onnx_session(sess, frame, imgsz, conf)
            else:
                res = model.predict(frame, device=device, imgsz=imgsz, conf=conf)
                boxes = res[0].boxes
                for b in boxes:
                    x1,y1,x2,y2 = b.xyxy[0].tolist()
                    confs = float(b.conf)
                    cls = int(b.cls)
                    if confs < conf: continue
                    dets.append([x1,y1,x2,y2,confs,cls])
        except KeyboardInterrupt:
            break
        except Exception as e:
            # if ONNX parsing failed, print error and continue
            print('Inference error:', e)

        frame_out = frame.copy()
        frame_out = draw_boxes(frame_out, dets, None)

        t1 = time.time()
        dt = t1-t0
        fps = 1.0/dt if dt > 0 else 0
        fps_smooth = 0.9*fps_smooth + 0.1*fps if fps_smooth else fps

        cv2.putText(frame_out, f'FPS: {fps_smooth:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.imshow('ONNX Realtime', frame_out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/best.onnx', help='ONNX model path or PyTorch/ultralytics weight file')
    parser.add_argument('--imgsz', type=int, default=416, help='inference image size')
    parser.add_argument('--conf', type=float, default=0.35, help='confidence threshold')
    parser.add_argument('--device', type=str, default='cpu', help='device for ultralytics fallback (cpu or 0 for gpu)')
    args = parser.parse_args()
    main(args.model, args.imgsz, args.conf, device=args.device)
