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
        epochs=150,             # số epoch
        imgsz=640,             # kích thước ảnh
        seed=seed,             # gán seed để tái lặp kết quả
        batch=8,               # giảm batch size cho CPU
        project='runs_train',  # thư mục lưu kết quả
        name='exp',            # tên folder con trong runs_train
        device='cpu',          # sử dụng CPU
        workers=2,             # số workers (giảm cho CPU)
        amp=False,             # tắt Automatic Mixed Precision cho CPU
        # Một số hyperparameter augment thường dùng, có thể tinh chỉnh:
        hsv_h=0.015,           # biến thiên Hue
        hsv_s=0.7,             # biến thiên Saturation
        hsv_v=0.4,             # biến thiên Value
        degrees=60,           # xoay ảnh ngẫu nhiên
        translate=0.2,         # tịnh tiến ảnh ngẫu nhiên
        scale=0.5,             # scale ảnh ngẫu nhiên
        shear=10,             # shear ảnh
        flipud=0.1,            # lật trên-dưới
        fliplr=0.5,            # lật trái-phải
        mosaic=1.0,            # xác suất mosaic
        mixup=0.1              # giảm mixup cho CPU
    )

if __name__ == "__main__":
    main()
