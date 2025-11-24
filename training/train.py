import os
from ultralytics import YOLO

def main():
    # Đặt seed mặc định
    seed = 666
    
    # Khởi tạo model, có thể thay 'yolov8n.pt' bằng kiến trúc khác, ví dụ: 'yolov8s.pt', 'yolov8m.pt', ...
    model = YOLO('yolov8n.pt')
    
    # Thực hiện train - Tối ưu cho CPU
    model.train(
        data='data.yaml',      # đường dẫn file data.yaml
        epochs=50,              # giảm epoch để test
        imgsz=416,             # giảm kích thước ảnh từ 640
        seed=seed,             # gán seed để tái lặp kết quả
        batch=4,               # giảm batch size còn 4
        project='runs_train',  # thư mục lưu kết quả
        name='exp_cpu',        # tên folder riêng cho CPU
        device='cpu',          # sử dụng CPU
        workers=0,             # không sử dụng workers (CPU bottleneck)
        amp=False,             # tắt Automatic Mixed Precision cho CPU
        cache=False,           # không cache images
        patience=20,           # early stopping
        # Một số hyperparameter augment thường dùng, có thể tinh chỉnh:
        hsv_h=0.015,           # biến thiên Hue
        hsv_s=0.7,             # biến thiên Saturation
        hsv_v=0.4,             # biến thiên Value
        degrees=30,            # giảm độ xoay
        translate=0.1,         # giảm tịnh tiến
        scale=0.3,             # giảm scale
        shear=5,               # giảm shear
        flipud=0.05,           # giảm lật
        fliplr=0.3,            # giảm flip
        mosaic=0.8,            # giảm mosaic
        mixup=0.0              # tắt mixup
    )

if __name__ == "__main__":
    main()
