"""
Script training nhanh với ít epochs hơn - Phù hợp cho demo hoặc test nhanh
Giảm epochs xuống để training nhanh hơn
"""
import os
from pathlib import Path
from ultralytics import YOLO

def main():
    """
    Hàm main - Training với ít epochs hơn
    """
    
    # Đảm bảo chạy từ đúng thư mục
    os.chdir(Path(__file__).parent)
    
    print("🚀 Bắt đầu training nhanh (ít epochs)...")
    print("=" * 60)
    
    # =============================
    # THIẾT LẬP THAM SỐ
    # =============================
    
    seed = 666
    pretrained_model = 'yolov8n.pt'
    best_model_path = 'runs_train/exp_cpu/weights/best.pt'
    
    # Load model
    if os.path.exists(best_model_path):
        print(f"📦 Tiếp tục từ: {best_model_path}")
        model = YOLO(best_model_path)
    else:
        print(f"🆕 Khởi tạo từ: {pretrained_model}")
        model = YOLO(pretrained_model)
    
    # =============================
    # THAM SỐ TRAINING (GIẢM EPOCHS)
    # =============================
    
    # ⚠️ GIẢM EPOCHS XUỐNG ĐÂY:
    epochs = 20  # Giảm từ 100 xuống 20 (có thể điều chỉnh: 10, 15, 20, 30...)
    
    # Các tham số khác giữ nguyên hoặc có thể giảm
    imgsz = 640      # Có thể giảm xuống 416 để nhanh hơn
    batch = 8        # Giữ nguyên hoặc giảm xuống 4
    device = 'cpu'
    workers = 0
    amp = False
    cache = False
    patience = 10    # Giảm patience vì epochs ít hơn
    
    print(f"\n⚙️ Cấu hình training:")
    print(f"  - Epochs: {epochs} (GIẢM để training nhanh hơn)")
    print(f"  - Image size: {imgsz}")
    print(f"  - Batch size: {batch}")
    print(f"  - Device: {device}")
    print("=" * 60)
    
    # =============================
    # BẮT ĐẦU TRAINING
    # =============================
    
    print("\n🎯 Bắt đầu training...")
    
    results = model.train(
        data='data.yaml',
        epochs=epochs,           # ⚠️ SỐ EPOCHS GIẢM XUỐNG
        imgsz=imgsz,
        batch=batch,
        seed=seed,
        project='runs_train',
        name='exp_quick',        # Tên experiment riêng cho training nhanh
        device=device,
        workers=workers,
        amp=amp,
        cache=cache,
        patience=patience,
        # Augmentation giữ nguyên
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=45,
        translate=0.2,
        scale=0.5,
        shear=10,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
    )
    
    print("\n✅ Training hoàn tất!")
    print(f"📁 Kết quả tại: runs_train/exp_quick/")
    print(f"📦 Model tốt nhất: runs_train/exp_quick/weights/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()


