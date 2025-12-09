"""
Script kiểm tra trạng thái training và model đã được lưu
Giúp bạn biết model đã được lưu ở đâu và có thể dùng ngay
"""
from pathlib import Path
import pandas as pd

def check_training_status(exp_name='exp_modular'):
    """
    Kiểm tra trạng thái training và model đã được lưu
    
    Args:
        exp_name: Tên experiment (exp_modular, exp_cpu, exp_detailed, v.v.)
    """
    print("=" * 60)
    print("🔍 KIỂM TRA TRẠNG THÁI TRAINING")
    print("=" * 60)
    
    # Đường dẫn thư mục training results
    exp_dir = Path('runs_train') / exp_name
    
    if not exp_dir.exists():
        print(f"\n❌ Không tìm thấy thư mục: {exp_dir}")
        print("\n💡 Các thư mục có sẵn:")
        runs_dir = Path('runs_train')
        if runs_dir.exists():
            for subdir in runs_dir.iterdir():
                if subdir.is_dir():
                    print(f"   - {subdir.name}")
        return
    
    print(f"\n✅ Tìm thấy thư mục: {exp_dir}")
    
    # Kiểm tra weights
    weights_dir = exp_dir / 'weights'
    print(f"\n📦 Kiểm tra weights:")
    
    if weights_dir.exists():
        best_pt = weights_dir / 'best.pt'
        last_pt = weights_dir / 'last.pt'
        
        if best_pt.exists():
            size_mb = best_pt.stat().st_size / (1024 * 1024)
            print(f"   ✅ best.pt: {size_mb:.2f} MB ⭐ (MODEL TỐT NHẤT - DÙNG CÁI NÀY!)")
        else:
            print(f"   ⚠️  best.pt: Chưa có (chưa có epoch nào tốt)")
        
        if last_pt.exists():
            size_mb = last_pt.stat().st_size / (1024 * 1024)
            print(f"   ✅ last.pt: {size_mb:.2f} MB (Model cuối cùng)")
        else:
            print(f"   ⚠️  last.pt: Chưa có")
    else:
        print(f"   ❌ Thư mục weights chưa tồn tại")
    
    # Kiểm tra results.csv
    results_csv = exp_dir / 'results.csv'
    print(f"\n📊 Kiểm tra metrics:")
    
    if results_csv.exists():
        try:
            df = pd.read_csv(results_csv)
            num_epochs = len(df)
            print(f"   ✅ results.csv: {num_epochs} epochs đã train")
            
            if num_epochs > 0:
                # Tìm epoch tốt nhất
                if 'metrics/mAP50(B)' in df.columns:
                    best_idx = df['metrics/mAP50(B)'].idxmax()
                    best_epoch = df.loc[best_idx]
                    print(f"\n   🏆 EPOCH TỐT NHẤT:")
                    print(f"      - Epoch: {int(best_epoch['epoch'])}")
                    print(f"      - mAP50: {best_epoch['metrics/mAP50(B)']:.4f}")
                    if 'metrics/mAP50-95(B)' in best_epoch:
                        print(f"      - mAP50-95: {best_epoch['metrics/mAP50-95(B)']:.4f}")
                
                # Epoch cuối cùng
                last_epoch = df.iloc[-1]
                print(f"\n   📌 EPOCH CUỐI CÙNG:")
                print(f"      - Epoch: {int(last_epoch['epoch'])}")
                if 'metrics/mAP50(B)' in last_epoch:
                    print(f"      - mAP50: {last_epoch['metrics/mAP50(B)']:.4f}")
        except Exception as e:
            print(f"   ⚠️  Không thể đọc results.csv: {e}")
    else:
        print(f"   ⚠️  results.csv: Chưa có (training chưa bắt đầu hoặc đang chạy)")
    
    # Hướng dẫn sử dụng
    print(f"\n" + "=" * 60)
    print("💡 HƯỚNG DẪN SỬ DỤNG MODEL")
    print("=" * 60)
    
    if weights_dir.exists() and (weights_dir / 'best.pt').exists():
        best_path = weights_dir / 'best.pt'
        print(f"\n✅ Model đã sẵn sàng! Bạn có thể:")
        print(f"\n1. Load model trong Python:")
        print(f"   from ultralytics import YOLO")
        print(f"   model = YOLO('{best_path}')")
        print(f"   results = model.predict('image.jpg')")
        
        print(f"\n2. Copy vào GiaoDien:")
        print(f"   copy {best_path} ..\\..\\GiaoDien\\weights\\best.pt")
        
        print(f"\n3. Sử dụng trong Streamlit app:")
        print(f"   App sẽ tự động load từ GiaoDien/weights/best.pt")
    else:
        print(f"\n⏳ Training đang chạy hoặc chưa hoàn thành...")
        print(f"   Đợi training xong, model sẽ tự động được lưu tại:")
        print(f"   {weights_dir / 'best.pt'}")
    
    print(f"\n" + "=" * 60)
    print("📝 LƯU Ý: Model được lưu TỰ ĐỘNG sau mỗi epoch!")
    print("   KHÔNG CẦN TRAIN LẠI - chỉ cần load file .pt")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    # Lấy tên experiment từ command line hoặc dùng mặc định
    exp_name = sys.argv[1] if len(sys.argv) > 1 else 'exp_modular'
    
    # Đảm bảo chạy từ thư mục training
    import os
    os.chdir(Path(__file__).parent)
    
    check_training_status(exp_name)


