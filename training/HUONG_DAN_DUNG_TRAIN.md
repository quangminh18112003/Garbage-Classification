# ⏹️ Hướng Dẫn Dừng Training

## 🛑 Cách Dừng Training Đang Chạy

### Cách 1: Dừng Bằng Phím Tắt (Nhanh Nhất) ⚡

Khi training đang chạy trong terminal/command prompt:

**Windows (PowerShell/CMD):**
```
Nhấn: Ctrl + C
```

**Linux/Mac:**
```
Nhấn: Ctrl + C
```

- Nhấn **1 lần**: Dừng nhẹ nhàng (graceful stop)
- Nhấn **2 lần liên tiếp**: Dừng ngay lập tức (force stop)

### Cách 2: Đóng Terminal/Command Prompt

- Đóng cửa sổ terminal đang chạy training
- Training sẽ dừng ngay lập tức

### Cách 3: Dừng Process Trong Task Manager (Windows)

1. Mở **Task Manager** (Ctrl + Shift + Esc)
2. Tìm process `python.exe` hoặc `pythonw.exe`
3. Click chuột phải → **End Task**

## ✅ Sau Khi Dừng Training

### Model Đã Được Lưu Tự Động!

Khi bạn dừng training, YOLO đã tự động lưu:

1. **`best.pt`** - Model tốt nhất từ các epochs đã train
2. **`last.pt`** - Model của epoch cuối cùng trước khi dừng
3. **`results.csv`** - Metrics của tất cả epochs đã train

### Vị Trí File:

```
training/runs_train/exp_modular/
├── weights/
│   ├── best.pt    ⭐ Model tốt nhất - DÙNG CÁI NÀY!
│   └── last.pt   📌 Model cuối cùng
└── results.csv   📊 Metrics đã train
```

## 🔄 Tiếp Tục Training Từ Điểm Dừng

### Cách 1: Tiếp Tục Từ Best Model

```python
from ultralytics import YOLO

# Load model tốt nhất đã train
model = YOLO('training/runs_train/exp_modular/weights/best.pt')

# Tiếp tục training thêm epochs
model.train(
    data='data.yaml',
    epochs=50,  # Train thêm 50 epochs
    resume=True  # ⚠️ QUAN TRỌNG: Tiếp tục từ checkpoint
)
```

### Cách 2: Load Best Model và Train Tiếp

```python
from ultralytics import YOLO

# Load model đã train
model = YOLO('training/runs_train/exp_modular/weights/best.pt')

# Train tiếp với epochs mới
model.train(
    data='data.yaml',
    epochs=100,  # Tổng số epochs mong muốn
    project='runs_train',
    name='exp_continued',  # Tên experiment mới
    device='cpu',
)
```

## 📊 Kiểm Tra Kết Quả Sau Khi Dừng

### Xem Số Epochs Đã Train:

```bash
# Kiểm tra file results.csv
python check_training_status.py exp_modular
```

Hoặc:

```python
import pandas as pd

df = pd.read_csv('training/runs_train/exp_modular/results.csv')
print(f"Đã train: {len(df)} epochs")
print(f"Epoch tốt nhất: {df['metrics/mAP50(B)'].idxmax()}")
```

## ⚠️ Lưu Ý Quan Trọng

### 1. Model Vẫn Có Thể Dùng
- ✅ Ngay cả khi dừng giữa chừng, `best.pt` vẫn là model tốt nhất
- ✅ Có thể dùng ngay để inference
- ✅ Không cần train lại từ đầu

### 2. Early Stopping
- Nếu training dừng do early stopping (patience), đó là bình thường
- Model đã đạt được mức tốt nhất có thể

### 3. Dừng Giữa Epoch
- Nếu dừng giữa epoch, epoch đó sẽ không được tính
- Chỉ các epochs hoàn chỉnh mới được lưu vào CSV

## 🎯 Ví Dụ Thực Tế

### Scenario 1: Dừng Training Đang Chạy

```
Training đang chạy:
Epoch 1/100: 50% ──────────── 500/1000

Bạn nhấn Ctrl + C
→ Training dừng
→ best.pt đã được lưu (nếu có epoch tốt)
→ last.pt = epoch 1
```

### Scenario 2: Tiếp Tục Training

```python
# Load model đã train (ví dụ: đã train 20 epochs)
model = YOLO('runs_train/exp_modular/weights/best.pt')

# Tiếp tục train thêm 30 epochs
model.train(
    data='data.yaml',
    epochs=50,  # Tổng 50 epochs (20 đã train + 30 mới)
    resume=True,
    project='runs_train',
    name='exp_modular',  # Cùng tên để tiếp tục
)
```

## 💡 Tips

1. **Nếu training quá lâu**: Dừng và dùng `best.pt` hiện tại
2. **Nếu muốn train thêm**: Load `best.pt` và train tiếp
3. **Luôn kiểm tra**: Xem `results.csv` để biết đã train bao nhiêu epochs

## ✅ Tóm Tắt

**Cách dừng:**
- ⚡ **Ctrl + C** (nhanh nhất)
- 🪟 Đóng terminal
- 🔧 Task Manager

**Sau khi dừng:**
- ✅ Model đã được lưu tự động (`best.pt`)
- ✅ Có thể dùng ngay
- ✅ Có thể tiếp tục training sau

**Không cần lo lắng - Model vẫn tốt!** 🚀

