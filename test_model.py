"""
Script test nhanh để kiểm tra model có hoạt động không
Chạy: python test_model.py
"""
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from PIL import Image

def test_model():
    model_path = Path('GiaoDien/weights/best.pt')
    
    if not model_path.exists():
        print(f"❌ Không tìm thấy model tại: {model_path}")
        print("💡 Hãy đảm bảo file best.pt có trong GiaoDien/weights/")
        return
    
    print(f"📦 Đang tải model từ: {model_path}")
    try:
        model = YOLO(str(model_path))
        print(f"✅ Model đã tải thành công!")
        print(f"📊 Số classes: {len(model.names)}")
        print(f"📋 Classes: {list(model.names.values())}")
    except Exception as e:
        print(f"❌ Lỗi khi tải model: {e}")
        return
    
    # Tạo ảnh test đơn giản (màu xanh lá)
    print("\n🧪 Tạo ảnh test...")
    test_img = Image.new('RGB', (640, 480), color='green')
    
    # Test với các cấu hình khác nhau
    configs = [
        {"conf": 0.1, "imgsz": 320, "name": "Conf=10%, Size=320px"},
        {"conf": 0.15, "imgsz": 416, "name": "Conf=15%, Size=416px"},
        {"conf": 0.2, "imgsz": 640, "name": "Conf=20%, Size=640px"},
        {"conf": 0.2, "imgsz": 800, "name": "Conf=20%, Size=800px"},
        {"conf": 0.15, "imgsz": 1024, "name": "Conf=15%, Size=1024px"},
    ]
    
    print("\n🔍 Test với các cấu hình khác nhau:\n")
    for config in configs:
        try:
            results = model.predict(
                test_img,
                conf=config["conf"],
                imgsz=config["imgsz"],
                max_det=100,
                verbose=False
            )
            num_det = len(results[0].boxes)
            print(f"  {config['name']}: {num_det} detections")
        except Exception as e:
            print(f"  {config['name']}: ❌ Lỗi - {e}")
    
    print("\n✅ Test hoàn thành!")
    print("\n💡 Nếu model không phát hiện được gì với ảnh test, có thể:")
    print("   1. Model chưa được train tốt")
    print("   2. Cần thêm ảnh rác trên nước vào dataset và retrain")
    print("   3. Thử với ảnh thực tế trong app Streamlit")

if __name__ == '__main__':
    test_model()

