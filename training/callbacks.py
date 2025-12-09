"""
Module callbacks - Theo dõi và log từng epoch trong quá trình training
Đọc từ file results.csv mà YOLO tự động tạo
"""
import json
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class EpochTracker:
    """Class theo dõi và lưu thông tin từng epoch từ YOLO results.csv"""
    
    def __init__(self, log_dir: Path, results_csv_path: Optional[Path] = None):
        self.log_dir = log_dir
        self.log_file = log_dir / 'epochs_log.json'
        self.results_csv_path = results_csv_path
        self.epochs_data: List[Dict] = []
        self.current_epoch = 0
        
        # Tạo file log nếu chưa có
        if self.log_file.exists():
            self.load_epochs()
    
    def load_epochs(self):
        """Load dữ liệu epochs từ file JSON"""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                self.epochs_data = json.load(f)
            if self.epochs_data:
                self.current_epoch = len(self.epochs_data)
            print(f"✅ Đã load {len(self.epochs_data)} epochs từ file log")
        except Exception as e:
            print(f"⚠️ Không thể load epochs log: {e}")
            self.epochs_data = []
    
    def read_results_csv(self, csv_path: Path) -> List[Dict]:
        """Đọc file results.csv từ YOLO training"""
        epochs = []
        try:
            if not csv_path.exists():
                return epochs
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epoch_info = {
                        'epoch': int(row.get('epoch', 0)),
                        'timestamp': datetime.now().isoformat(),
                        'metrics': {
                            'train/box_loss': float(row.get('train/box_loss', 0)),
                            'train/cls_loss': float(row.get('train/cls_loss', 0)),
                            'train/dfl_loss': float(row.get('train/dfl_loss', 0)),
                            'metrics/mAP50(B)': float(row.get('metrics/mAP50(B)', 0)),
                            'metrics/mAP50-95(B)': float(row.get('metrics/mAP50-95(B)', 0)),
                            'metrics/precision(B)': float(row.get('metrics/precision(B)', 0)),
                            'metrics/recall(B)': float(row.get('metrics/recall(B)', 0)),
                        }
                    }
                    epochs.append(epoch_info)
        except Exception as e:
            print(f"⚠️ Lỗi khi đọc results.csv: {e}")
        
        return epochs
    
    def update_from_csv(self, csv_path: Optional[Path] = None):
        """Cập nhật epochs từ file CSV"""
        if csv_path is None:
            csv_path = self.results_csv_path
        
        if csv_path is None or not csv_path.exists():
            return
        
        new_epochs = self.read_results_csv(csv_path)
        if new_epochs:
            # Cập nhật hoặc thêm epochs mới
            for epoch_info in new_epochs:
                epoch_num = epoch_info['epoch']
                if epoch_num < len(self.epochs_data):
                    self.epochs_data[epoch_num] = epoch_info
                else:
                    self.epochs_data.append(epoch_info)
            
            # Lưu vào file JSON
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.epochs_data, f, indent=2, ensure_ascii=False)
            
            self.current_epoch = len(self.epochs_data)
            print(f"✅ Đã cập nhật {len(new_epochs)} epochs từ CSV")
    
    def save_epoch(self, epoch: int, metrics: Dict):
        """Lưu thông tin một epoch"""
        epoch_info = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        # Cập nhật hoặc thêm epoch mới
        if epoch < len(self.epochs_data):
            self.epochs_data[epoch] = epoch_info
        else:
            self.epochs_data.append(epoch_info)
        
        # Lưu vào file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.epochs_data, f, indent=2, ensure_ascii=False)
        
        self.current_epoch = epoch + 1
        print(f"💾 Đã lưu epoch {epoch} vào log")
    
    def get_epoch(self, epoch: int) -> Optional[Dict]:
        """Lấy thông tin một epoch cụ thể"""
        if 0 <= epoch < len(self.epochs_data):
            return self.epochs_data[epoch]
        return None
    
    def get_all_epochs(self) -> List[Dict]:
        """Lấy tất cả epochs"""
        return self.epochs_data
    
    def get_latest_epoch(self) -> Optional[Dict]:
        """Lấy epoch mới nhất"""
        if self.epochs_data:
            return self.epochs_data[-1]
        return None
    
    def print_epoch_summary(self, epoch: int, metrics: Dict):
        """In tóm tắt thông tin epoch"""
        print("\n" + "=" * 60)
        print(f"📊 EPOCH {epoch}")
        print("=" * 60)
        print(f"⏰ Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n📈 Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 60 + "\n")
    
    def print_all_epochs_summary(self):
        """In tóm tắt tất cả epochs"""
        if not self.epochs_data:
            print("⚠️ Chưa có dữ liệu epochs")
            return
        
        print("\n" + "=" * 60)
        print(f"📊 TỔNG KẾT {len(self.epochs_data)} EPOCHS")
        print("=" * 60)
        
        # In header
        print(f"{'Epoch':<8} {'Box Loss':<12} {'Cls Loss':<12} {'mAP50':<10} {'mAP50-95':<12}")
        print("-" * 60)
        
        # In từng epoch
        for epoch_info in self.epochs_data:
            epoch = epoch_info['epoch']
            metrics = epoch_info['metrics']
            box_loss = metrics.get('train/box_loss', 0)
            cls_loss = metrics.get('train/cls_loss', 0)
            map50 = metrics.get('metrics/mAP50(B)', 0)
            map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
            
            print(f"{epoch:<8} {box_loss:<12.4f} {cls_loss:<12.4f} {map50:<10.4f} {map50_95:<12.4f}")
        
        print("=" * 60 + "\n")

