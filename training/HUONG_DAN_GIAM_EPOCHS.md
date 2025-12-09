# ⚡ Hướng Dẫn Giảm Epochs - Training Nhanh Hơn

## 🎯 Tại Sao Giảm Epochs?

- **Training nhanh hơn** - Tiết kiệm thời gian
- **Phù hợp cho demo/test** - Không cần train quá lâu
- **Vẫn đạt kết quả tốt** - Với early stopping, model vẫn học được

## 📝 Cách Giảm Epochs

### Cách 1: Sửa Trực Tiếp Trong Code (Nhanh Nhất)

#### File `config.py`:
```python
self.epochs = 50  # Giảm từ 100 xuống 50 (hoặc 20, 30...)
```

#### File `train_with_comments.py`:
```python
epochs = 50  # Giảm từ 100 xuống 50
```

### Cách 2: Dùng Script Training Nhanh

```bash
python train_quick.py
```

Script này đã được cấu hình sẵn với **20 epochs** (có thể sửa trong file)

### Cách 3: Chỉnh Sửa Khi Chạy

Trong code, thay đổi trước khi gọi `model.train()`:

```python
# Giảm epochs
epochs = 20  # Thay vì 100

model.train(
    data='data.yaml',
    epochs=epochs,  # Dùng số epochs đã giảm
    # ... các tham số khác
)
```

## ⚙️ Gợi Ý Số Epochs

| Mục đích | Số Epochs | Thời gian (ước tính) |
|----------|-----------|---------------------|
| **Test nhanh** | 10-20 | ~30 phút - 1 giờ |
| **Demo** | 20-30 | ~1-2 giờ |
| **Training đầy đủ** | 50-100 | ~3-6 giờ |
| **Training tốt nhất** | 100-200 | ~6-12 giờ |

## 💡 Lưu Ý

### 1. Early Stopping
- Với `patience=30`, nếu model không cải thiện trong 30 epochs, training sẽ dừng
- Nếu giảm epochs xuống 20, nên giảm patience xuống 10-15

### 2. Model Vẫn Tốt
- YOLO tự động lưu `best.pt` (model tốt nhất)
- Ngay cả khi train ít epochs, model vẫn có thể tốt nếu early stopping hoạt động

### 3. Có Thể Tiếp Tục Training
- Sau khi train xong, có thể load `best.pt` và train thêm
- Không cần train lại từ đầu

## 🚀 Ví Dụ: Training Nhanh 20 Epochs

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='data.yaml',
    epochs=20,        # ⚡ GIẢM XUỐNG 20
    imgsz=640,
    batch=8,
    patience=10,      # ⚡ GIẢM PATIENCE
    project='runs_train',
    name='exp_quick',
    device='cpu',
)
```

## 📊 So Sánh

| Epochs | Thời gian | Chất lượng model |
|--------|-----------|------------------|
| 10 | ~30 phút | Cơ bản, đủ để test |
| 20 | ~1 giờ | Tốt cho demo |
| 50 | ~3 giờ | Tốt, phù hợp đồ án |
| 100 | ~6 giờ | Rất tốt, đầy đủ |

## ✅ Kết Luận

**Có thể giảm epochs xuống 20-50 để training nhanh hơn!**

- ✅ Vẫn đạt kết quả tốt với early stopping
- ✅ Tiết kiệm thời gian
- ✅ Có thể tiếp tục training sau nếu cần
- ✅ Model tự động lưu `best.pt`

**Khuyến nghị cho đồ án: 30-50 epochs là đủ!**


