# 💾 Hướng Dẫn: Model Được Lưu Tự Động - KHÔNG CẦN TRAIN LẠI!

## ✅ YOLO Tự Động Lưu Model Sau Mỗi Epoch

Khi bạn đang training (như bạn thấy epoch 1/100), YOLO **TỰ ĐỘNG** lưu:

### 📁 Cấu Trúc Thư Mục Sau Training

```
training/runs_train/exp_modular/  (hoặc exp_cpu, exp_detailed tùy script)
├── weights/
│   ├── best.pt          ⭐ MODEL TỐT NHẤT (mAP cao nhất)
│   ├── last.pt          📌 MODEL CUỐI CÙNG (epoch cuối)
│   └── epoch1.pt        📦 Model từng epoch (nếu có)
│
├── results.csv          📊 File CSV chứa metrics tất cả epochs
├── args.yaml            ⚙️ File cấu hình training
├── confusion_matrix.png 📈 Biểu đồ confusion matrix
├── F1_curve.png         📈 F1 score curve
├── P_curve.png          📈 Precision curve
├── R_curve.png          📈 Recall curve
├── PR_curve.png         📈 Precision-Recall curve
└── train_batch*.jpg     🖼️ Ảnh mẫu training
```

## 🎯 Các File Quan Trọng

### 1. **best.pt** - Model Tốt Nhất ⭐
- Model có **mAP cao nhất** trong quá trình training
- Đây là model bạn nên dùng để inference
- Được cập nhật tự động khi có epoch tốt hơn

### 2. **last.pt** - Model Cuối Cùng
- Model sau epoch cuối cùng
- Có thể không phải tốt nhất, nhưng là model mới nhất

### 3. **results.csv** - Metrics Tất Cả Epochs
- Chứa tất cả metrics: loss, mAP, precision, recall...
- Có thể đọc bằng Excel hoặc Python

## 🚀 Cách Sử Dụng Model Đã Train (KHÔNG CẦN TRAIN LẠI!)

### Cách 1: Load Model Để Inference (Phát Hiện)

```python
from ultralytics import YOLO

# Load model đã train
model = YOLO('training/runs_train/exp_modular/weights/best.pt')

# Sử dụng để phát hiện
results = model.predict('path/to/image.jpg')
```

### Cách 2: Copy Model Vào GiaoDien

```bash
# Copy model tốt nhất vào thư mục GiaoDien
copy training\runs_train\exp_modular\weights\best.pt GiaoDien\weights\best.pt
```

Sau đó app Streamlit sẽ tự động dùng model này!

### Cách 3: Tiếp Tục Training Từ Model Đã Train

```python
from ultralytics import YOLO

# Load model đã train (thay vì pretrained)
model = YOLO('training/runs_train/exp_modular/weights/best.pt')

# Tiếp tục training thêm epochs
model.train(
    data='data.yaml',
    epochs=50,  # Train thêm 50 epochs
    resume=True  # Tiếp tục từ checkpoint
)
```

## 📊 Đọc Kết Quả Training

### Xem Metrics Trong results.csv:

```python
import pandas as pd

# Đọc file CSV
df = pd.read_csv('training/runs_train/exp_modular/results.csv')

# Xem tất cả metrics
print(df.head())

# Tìm epoch tốt nhất (mAP cao nhất)
best_epoch = df.loc[df['metrics/mAP50(B)'].idxmax()]
print(f"Best epoch: {best_epoch['epoch']}")
print(f"Best mAP50: {best_epoch['metrics/mAP50(B)']}")
```

### Hoặc dùng script có sẵn:

```bash
python read_epochs.py --summary --log_dir runs_train/exp_modular/logs
```

## ⚠️ Lưu Ý Quan Trọng

### ✅ Model Được Lưu Tự Động
- **best.pt** được cập nhật mỗi khi có epoch tốt hơn
- **last.pt** được cập nhật sau mỗi epoch
- Bạn **KHÔNG CẦN** làm gì cả, YOLO tự động lưu!

### ✅ Không Cần Train Lại
- Sau khi training xong, bạn có thể dùng model ngay
- Chỉ cần load file `.pt` là xong
- Training 1 lần, dùng mãi mãi!

### ✅ Có Thể Tiếp Tục Training
- Nếu muốn train thêm, load `best.pt` hoặc `last.pt`
- Dùng `resume=True` để tiếp tục từ checkpoint
- Hoặc train từ đầu với model đã train làm pretrained

## 🎯 Workflow Đề Xuất

### Sau Khi Training Xong:

1. **Kiểm tra kết quả:**
   ```bash
   cd training/runs_train/exp_modular
   # Xem results.csv, các file .png
   ```

2. **Copy model tốt nhất:**
   ```bash
   copy weights\best.pt ..\..\..\GiaoDien\weights\best.pt
   ```

3. **Test với Streamlit:**
   ```bash
   cd ..\..\..\GiaoDien
   streamlit run app.py
   ```

4. **Nếu cần train thêm:**
   - Load `best.pt` và train tiếp
   - Hoặc điều chỉnh hyperparameters và train lại

## 💡 Ví Dụ Thực Tế

### Scenario 1: Training Đang Chạy (Như Bạn Bây Giờ)
```
Epoch 1/100: best.pt đã được tạo (nếu đây là epoch đầu)
Epoch 2/100: best.pt được cập nhật nếu epoch 2 tốt hơn
...
Epoch 50/100: best.pt = model tốt nhất từ epoch 1-50
```

### Scenario 2: Training Xong
```
✅ best.pt = Model tốt nhất (ví dụ: epoch 45)
✅ last.pt = Model epoch 100
✅ results.csv = Tất cả metrics từ epoch 1-100
```

### Scenario 3: Muốn Dùng Model
```python
# Chỉ cần load, KHÔNG CẦN TRAIN LẠI!
model = YOLO('training/runs_train/exp_modular/weights/best.pt')
results = model.predict('image.jpg')
```

## 🎉 Kết Luận

**YOLO TỰ ĐỘNG LƯU MODEL - BẠN KHÔNG CẦN LÀM GÌ!**

- ✅ Model được lưu tự động sau mỗi epoch
- ✅ `best.pt` = model tốt nhất, dùng để inference
- ✅ `last.pt` = model cuối cùng
- ✅ `results.csv` = tất cả metrics
- ✅ **KHÔNG CẦN TRAIN LẠI** - chỉ cần load file `.pt`
- ✅ Có thể tiếp tục training từ model đã train

**Chỉ cần đợi training xong, sau đó copy `best.pt` vào `GiaoDien/weights/` là xong!** 🚀


