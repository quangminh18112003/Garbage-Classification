# 📚 Hướng Dẫn Training Modular

Hệ thống training được tổ chức theo module để dễ quản lý và theo dõi từng epoch.

## 📁 Cấu Trúc Modules

```
training/
├── config.py          # Module quản lý cấu hình training
├── callbacks.py       # Module theo dõi và log từng epoch
├── model_utils.py     # Module quản lý model (load, save)
├── train_modular.py   # Script training chính sử dụng các modules
└── read_epochs.py     # Script đọc và hiển thị thông tin epochs
```

## 🚀 Cách Sử Dụng

### 1. Training với Modular System

```bash
cd training
python train_modular.py
```

Script sẽ:
- ✅ Load cấu hình từ `config.py`
- ✅ Khởi tạo epoch tracker để theo dõi từng epoch
- ✅ Load model từ pretrained hoặc best model
- ✅ Bắt đầu training và tự động lưu thông tin từng epoch
- ✅ Đọc kết quả từ `results.csv` sau khi training xong

### 2. Đọc Thông Tin Epochs

#### Xem epoch mới nhất:
```bash
python read_epochs.py --log_dir runs_train/exp_modular/logs
```

#### Xem tất cả epochs:
```bash
python read_epochs.py --summary --log_dir runs_train/exp_modular/logs
```

#### Xem một epoch cụ thể:
```bash
python read_epochs.py --epoch 10 --log_dir runs_train/exp_modular/logs
```

#### Đọc từ file CSV:
```bash
python read_epochs.py --csv runs_train/exp_modular/results.csv --log_dir runs_train/exp_modular/logs
```

## ⚙️ Tùy Chỉnh Cấu Hình

Chỉnh sửa file `config.py` để thay đổi các tham số training:

```python
class TrainingConfig:
    def __init__(self):
        self.epochs = 100        # Số epochs
        self.imgsz = 640         # Kích thước ảnh
        self.batch = 8           # Batch size
        self.device = 'cpu'      # 'cpu' hoặc '0' (GPU)
        # ... các tham số khác
```

## 📊 Cấu Trúc Log Files

Sau khi training, các file log được lưu tại:
```
runs_train/exp_modular/
├── logs/
│   ├── epochs_log.json      # JSON chứa thông tin tất cả epochs
│   ├── training.log          # Log text của training
│   └── training_summary.txt  # Tóm tắt training
└── results.csv               # File CSV từ YOLO (tự động tạo)
```

## 📈 Thông Tin Mỗi Epoch

Mỗi epoch được lưu với các thông tin:
- **epoch**: Số thứ tự epoch
- **timestamp**: Thời gian
- **metrics**: 
  - `train/box_loss`: Box loss
  - `train/cls_loss`: Classification loss
  - `train/dfl_loss`: DFL loss
  - `metrics/mAP50(B)`: mAP@0.5
  - `metrics/mAP50-95(B)`: mAP@0.5:0.95
  - `metrics/precision(B)`: Precision
  - `metrics/recall(B)`: Recall

## 🔧 Tích Hợp Modules

Bạn có thể import và sử dụng các modules trong code của mình:

```python
from config import TrainingConfig
from callbacks import EpochTracker
from model_utils import ModelManager

# Sử dụng config
config = TrainingConfig()
config.epochs = 150  # Tùy chỉnh

# Sử dụng epoch tracker
tracker = EpochTracker(config.log_dir)
epochs = tracker.get_all_epochs()

# Sử dụng model manager
manager = ModelManager('yolov8n.pt')
model = manager.load_model()
```

## 💡 Tips

1. **Theo dõi training real-time**: Mở terminal khác và chạy `read_epochs.py --summary` để xem tiến trình
2. **Tiếp tục training**: Script tự động tìm `best.pt` để tiếp tục training
3. **Đọc từ CSV**: Nếu training đã chạy xong, dùng `--csv` để đọc lại kết quả

## 📝 Ví Dụ Sử Dụng

### Training với GPU:
```python
# Trong config.py
self.device = '0'  # GPU
self.batch = 16    # Tăng batch size
```

### Training với nhiều epochs:
```python
# Trong config.py
self.epochs = 200
self.patience = 50  # Tăng patience
```

### Đọc và phân tích epochs:
```python
from callbacks import EpochTracker
from pathlib import Path

tracker = EpochTracker(Path('runs_train/exp_modular/logs'))
tracker.update_from_csv(Path('runs_train/exp_modular/results.csv'))

# Lấy epoch tốt nhất (mAP cao nhất)
all_epochs = tracker.get_all_epochs()
best_epoch = max(all_epochs, key=lambda x: x['metrics'].get('metrics/mAP50(B)', 0))
print(f"Best epoch: {best_epoch['epoch']}")
```


