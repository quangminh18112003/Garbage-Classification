"""
Script training chỉ 10 epochs - Training nhanh cho test/demo
"""
import os
from pathlib import Path
from ultralytics import YOLO

def main():
    """
    Hàm main - Training chỉ 10 epochs
    """
    
    # Đảm bảo chạy từ đúng thư mục
    os.chdir(Path(__file__).parent)
    
    print("🚀 Bắt đầu training với 10 epochs...")
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
    # THAM SỐ TRAINING - CHỈ 10 EPOCHS
    # =============================
    
    epochs = 10  # ⚡ CHỈ 10 EPOCHS
    
    # Các tham số khác
    imgsz = 640
    batch = 8
    device = 'cpu'
    workers = 0
    amp = False
    cache = False
    patience = 5  # Giảm patience vì chỉ 10 epochs
    
    print(f"\n⚙️ Cấu hình training:")
    print(f"  - Epochs: {epochs} ⚡ (CHỈ 10 EPOCHS - Training nhanh)")
    print(f"  - Image size: {imgsz}")
    print(f"  - Batch size: {batch}")
    print(f"  - Device: {device}")
    print(f"  - Patience: {patience} (early stopping)")
    print("=" * 60)
    print("\n⏳ Training sẽ mất khoảng 30-60 phút tùy máy...")
    print("=" * 60)
    
    # =============================
    # BẮT ĐẦU TRAINING
    # =============================
    
    print("\n🎯 Bắt đầu training...\n")
    
    results = model.train(
        data='data.yaml',
        epochs=epochs,           # ⚡ CHỈ 10 EPOCHS
        imgsz=imgsz,
        batch=batch,
        seed=seed,
        project='runs_train',
        name='exp_10epochs',     # Tên experiment riêng
        device=device,
        workers=workers,
        amp=amp,
        cache=cache,
        patience=patience,
        # Augmentation
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
    
    print("\n" + "=" * 60)
    print("✅ Training hoàn tất!")
    print("=" * 60)
    print(f"\n📁 Kết quả tại: runs_train/exp_10epochs/")
    print(f"📦 Model tốt nhất: runs_train/exp_10epochs/weights/best.pt")
    print(f"📊 Metrics: runs_train/exp_10epochs/results.csv")
    print("\n💡 Để sử dụng model:")
    print(f"   copy runs_train\\exp_10epochs\\weights\\best.pt ..\\..\\GiaoDien\\weights\\best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()

