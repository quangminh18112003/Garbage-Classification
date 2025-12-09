"""
Script đọc và hiển thị thông tin từng epoch từ kết quả training
"""
import sys
from pathlib import Path
import argparse

# Thêm thư mục training vào path
sys.path.insert(0, str(Path(__file__).parent))

from callbacks import EpochTracker

def main():
    parser = argparse.ArgumentParser(description='Đọc và hiển thị thông tin epochs từ training')
    parser.add_argument('--log_dir', type=str, default='runs_train/exp_modular/logs',
                       help='Đường dẫn thư mục log')
    parser.add_argument('--csv', type=str, default=None,
                       help='Đường dẫn file results.csv (nếu muốn đọc từ CSV)')
    parser.add_argument('--epoch', type=int, default=None,
                       help='Hiển thị thông tin một epoch cụ thể')
    parser.add_argument('--summary', action='store_true',
                       help='Hiển thị tóm tắt tất cả epochs')
    
    args = parser.parse_args()
    
    # Khởi tạo tracker
    log_dir = Path(args.log_dir)
    csv_path = Path(args.csv) if args.csv else None
    
    tracker = EpochTracker(log_dir, csv_path)
    
    # Nếu có CSV, cập nhật từ CSV
    if csv_path and csv_path.exists():
        print(f"📖 Đang đọc từ CSV: {csv_path}")
        tracker.update_from_csv(csv_path)
    
    # Hiển thị thông tin
    if args.epoch is not None:
        epoch_info = tracker.get_epoch(args.epoch)
        if epoch_info:
            tracker.print_epoch_summary(epoch_info['epoch'], epoch_info['metrics'])
        else:
            print(f"⚠️ Không tìm thấy epoch {args.epoch}")
    elif args.summary:
        tracker.print_all_epochs_summary()
    else:
        # Hiển thị epoch mới nhất
        latest = tracker.get_latest_epoch()
        if latest:
            print("📊 EPOCH MỚI NHẤT:")
            tracker.print_epoch_summary(latest['epoch'], latest['metrics'])
        else:
            print("⚠️ Chưa có dữ liệu epochs")
            print(f"💡 Sử dụng --csv để đọc từ file results.csv")
            print(f"💡 Sử dụng --summary để xem tất cả epochs")

if __name__ == "__main__":
    main()


