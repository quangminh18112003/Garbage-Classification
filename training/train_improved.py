"""
Script training cải thiện để phát hiện tốt hơn với ảnh rác trên nước
- Tăng epochs
- Tăng image size
- Cải thiện augmentation
"""
import os
from ultralytics import YOLO

def main():
    # Đặt seed mặc định
    seed = 666
    
    # Khởi tạo model từ best.pt nếu có, nếu không thì từ pretrained
    pretrained_model = 'yolov8n.pt'
    best_model_path = 'runs_train/exp_cpu/weights/best.pt'
    
    if os.path.exists(best_model_path):
        print(f"📦 Tiếp tục training từ: {best_model_path}")
        model = YOLO(best_model_path)
    else:
        print(f"🆕 Khởi tạo từ pretrained: {pretrained_model}")
        model = YOLO(pretrained_model)
    
    # Thực hiện train - Cải thiện cho phát hiện tốt hơn
    model.train(
        data='data.yaml',      # đường dẫn file data.yaml
        epochs=100,            # Tăng epochs để học tốt hơn
        imgsz=640,             # Tăng kích thước ảnh để phát hiện tốt hơn
        seed=seed,             # gán seed để tái lặp kết quả
        batch=8,               # Tăng batch size nếu có GPU, giữ 4 nếu CPU
        project='runs_train',  # thư mục lưu kết quả
        name='exp_improved',   # tên folder riêng cho training cải thiện
        device='cpu',          # sử dụng CPU (đổi thành '0' nếu có GPU)
        workers=0,            # không sử dụng workers (CPU bottleneck)
        amp=False,            # tắt Automatic Mixed Precision cho CPU
        cache=False,          # không cache images
        patience=30,          # early stopping - tăng patience
        # Augmentation tốt hơn để học được nhiều trường hợp khác nhau
        hsv_h=0.02,           # biến thiên Hue
        hsv_s=0.7,            # biến thiên Saturation
        hsv_v=0.4,            # biến thiên Value
        degrees=45,            # tăng độ xoay để học được góc độ khác nhau
        translate=0.2,        # tăng tịnh tiến
        scale=0.5,            # tăng scale để học được kích thước khác nhau
        shear=10,             # tăng shear
        flipud=0.1,           # tăng lật dọc
        fliplr=0.5,           # tăng flip ngang
        mosaic=1.0,           # bật mosaic đầy đủ
        mixup=0.1,            # bật mixup nhẹ
        copy_paste=0.1        # copy-paste augmentation
    )

if __name__ == "__main__":
    main()

