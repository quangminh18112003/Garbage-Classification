"""
Script training modular - Training với cấu trúc module, theo dõi từng epoch
"""
import os
import sys
from pathlib import Path

# Thêm thư mục training vào path để import modules
sys.path.insert(0, str(Path(__file__).parent))

from config import TrainingConfig
from callbacks import EpochTracker
from model_utils import ModelManager

def main():
    """Hàm main để chạy training"""
    
    # =============================
    # 📋 Khởi tạo cấu hình
    # =============================
    print("🚀 Bắt đầu training với cấu trúc modular...")
    config = TrainingConfig()
    config.print_config()
    
    # =============================
    # 📊 Khởi tạo epoch tracker
    # =============================
    epoch_tracker = EpochTracker(config.log_dir)
    print(f"📝 Log directory: {config.log_dir}")
    
    # =============================
    # 🤖 Load model
    # =============================
    model_manager = ModelManager(
        pretrained_model=config.pretrained_model,
        best_model_path=config.best_model_path
    )
    model = model_manager.load_model()
    model_manager.print_model_info()
    
    # =============================
    # 🎯 Bắt đầu training
    # =============================
    print("\n🎯 Bắt đầu training...")
    print("=" * 60)
    
    # Lấy các tham số training
    train_kwargs = config.get_train_kwargs()
    
    # Thực hiện training
    results = model.train(**train_kwargs)
    
    # =============================
    # 📊 Thu thập metrics sau training
    # =============================
    print("\n📊 Thu thập metrics từ kết quả training...")
    
    # Tìm file results.csv từ YOLO
    results_csv = Path(config.project) / config.name / 'results.csv'
    if results_csv.exists():
        print(f"📖 Đang đọc kết quả từ: {results_csv}")
        epoch_tracker.results_csv_path = results_csv
        epoch_tracker.update_from_csv(results_csv)
    else:
        print(f"⚠️ Không tìm thấy file results.csv tại: {results_csv}")
    
    # Lấy metrics từ epoch cuối cùng
    all_epochs = epoch_tracker.get_all_epochs()
    if all_epochs:
        final_metrics = all_epochs[-1].get('metrics', {})
    else:
        final_metrics = {}
    
    # =============================
    # 💾 Lưu thông tin training
    # =============================
    print("\n💾 Lưu thông tin training...")
    
    # Lưu summary
    summary_file = config.log_dir / 'training_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("📊 TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total Epochs: {config.epochs}\n")
        f.write(f"Image Size: {config.imgsz}\n")
        f.write(f"Batch Size: {config.batch}\n")
        f.write(f"Device: {config.device}\n")
        f.write("\nFinal Metrics:\n")
        for key, value in final_metrics.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write("=" * 60 + "\n")
    
    print(f"✅ Đã lưu summary tại: {summary_file}")
    
    # =============================
    # 📈 In thống kê epochs
    # =============================
    all_epochs = epoch_tracker.get_all_epochs()
    if all_epochs:
        print(f"\n📈 Đã hoàn thành {len(all_epochs)} epochs")
        print(f"📁 Log file: {epoch_tracker.log_file}")
        print(f"📝 Training log: {config.log_file}")
        print(f"\n💡 Để xem chi tiết epochs, chạy:")
        print(f"   python read_epochs.py --log_dir {config.log_dir}")
        print(f"   python read_epochs.py --summary --log_dir {config.log_dir}")
    else:
        print(f"\n💡 Để đọc epochs từ CSV, chạy:")
        print(f"   python read_epochs.py --csv {results_csv} --log_dir {config.log_dir}")
    
    print("\n✅ Training hoàn tất!")
    print("=" * 60)


if __name__ == "__main__":
    # Đảm bảo chạy từ thư mục training
    os.chdir(Path(__file__).parent)
    main()

