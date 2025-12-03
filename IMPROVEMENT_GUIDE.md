# 🚀 Hướng Dẫn Cải Thiện Model Phát Hiện Rác Trên Nước

## 🔍 Vấn Đề Hiện Tại
Model chỉ phát hiện được một vài chai nhựa trong ảnh rác trên nước, không phát hiện được hết các đối tượng.

## ✅ Giải Pháp Nhanh (Áp Dụng Ngay)

### 1. Giảm Confidence Threshold
Trong giao diện Streamlit, bạn đã có thể điều chỉnh:
- **Confidence mặc định**: Giảm từ 40% xuống **25%** (đã cập nhật)
- **Kích thước ảnh**: Tăng từ 416 lên **640px** (đã cập nhật)
- **IOU threshold**: Đã thêm 0.45 để giảm overlap

**Cách test:**
1. Chạy app: `streamlit run GiaoDien/app.py`
2. Upload ảnh rác trên nước
3. Giảm confidence xuống 20-25%
4. Tăng image size lên 640 hoặc 800
5. Click "Phát hiện đối tượng"

### 2. Test với Confidence Thấp Hơn
Nếu vẫn không đủ, thử:
- Confidence: **15-20%** (có thể có false positives nhưng sẽ bắt được nhiều hơn)
- Image size: **800px** hoặc **1024px** (nếu máy đủ mạnh)

## 🔧 Giải Pháp Lâu Dài (Cải Thiện Model)

### Bước 1: Thêm Ảnh Rác Trên Nước Vào Dataset

#### Cách 1: Sử dụng LabelImg để tạo annotations
```bash
# Cài đặt LabelImg
pip install labelimg
labelimg
```

1. Mở LabelImg
2. Chọn thư mục: `dataset_split/images/train/`
3. Format: YOLO
4. Thêm ảnh rác trên nước vào
5. Label các đối tượng: glass, biological, plastic
6. Save annotations vào `dataset_split/labels/train/`

#### Cách 2: Sử dụng Roboflow (Online)
1. Upload ảnh lên Roboflow
2. Label annotations
3. Export về YOLO format
4. Copy vào `dataset_split/`

### Bước 2: Retrain Model với Dataset Mới

#### Option A: Fine-tuning từ model hiện tại
```bash
cd training
python train_improved.py
```

Script này sẽ:
- Load model `best.pt` hiện tại (nếu có)
- Training thêm 100 epochs với dataset mới
- Image size 640px (tốt hơn 416px)
- Augmentation tốt hơn

#### Option B: Training từ đầu với dataset đầy đủ
```bash
cd training
python train.py
# Hoặc chỉnh sửa train.py để tăng epochs và imgsz
```

### Bước 3: Kiểm Tra và Đánh Giá

Sau khi training xong:
1. Test với ảnh rác trên nước
2. Kiểm tra metrics trong `training/runs_train/exp_improved/results.csv`
3. Mục tiêu:
   - **mAP50**: > 80%
   - **Precision**: > 0.75
   - **Recall**: > 0.70

## 📊 Cấu Hình Training Đề Xuất

### Cho CPU:
```python
epochs=100
imgsz=640
batch=4
device='cpu'
```

### Cho GPU (nếu có):
```python
epochs=150
imgsz=640
batch=16
device='0'  # GPU index
workers=4
amp=True  # Mixed precision
```

## 🎯 Checklist Cải Thiện

- [x] Giảm confidence threshold xuống 25%
- [x] Tăng image size lên 640px
- [x] Thêm IOU threshold
- [ ] Thu thập ảnh rác trên nước (ít nhất 50-100 ảnh)
- [ ] Label annotations cho ảnh mới
- [ ] Thêm vào dataset train/val
- [ ] Retrain với `train_improved.py`
- [ ] Test và đánh giá kết quả
- [ ] Copy `best.pt` mới vào `GiaoDien/weights/`

## 💡 Tips Quan Trọng

1. **Dataset đa dạng**: Đảm bảo dataset có:
   - Rác trên nước
   - Rác trên cạn
   - Rác trong túi
   - Rác rải rác
   - Nhiều góc độ khác nhau
   - Điều kiện ánh sáng khác nhau

2. **Class Balance**: Đảm bảo mỗi class có đủ số lượng:
   - Glass: ~30% dataset
   - Biological: ~30% dataset  
   - Plastic: ~40% dataset

3. **Validation Set**: Giữ 20-30% dataset cho validation để đánh giá đúng

4. **Test Set**: Giữ 10-20% dataset cho test cuối cùng

## 🔄 Workflow Đề Xuất

```
1. Thu thập ảnh rác trên nước (50-100 ảnh)
   ↓
2. Label annotations bằng LabelImg
   ↓
3. Thêm vào dataset_split/images/train và labels/train
   ↓
4. Chia lại train/val/test (80/15/5)
   ↓
5. Retrain với train_improved.py
   ↓
6. Test với ảnh thực tế
   ↓
7. Nếu tốt → Deploy, nếu chưa → Lặp lại từ bước 1
```

## 📝 Script Hỗ Trợ

Đã tạo `training/train_improved.py` với:
- Epochs cao hơn (100)
- Image size lớn hơn (640)
- Augmentation tốt hơn
- Tự động load từ best.pt nếu có

## 🆘 Troubleshooting

### Model vẫn không phát hiện đủ?
- Giảm confidence xuống 15%
- Tăng image size lên 800-1024px
- Kiểm tra xem dataset có đủ ảnh tương tự không

### Training quá chậm?
- Giảm epochs xuống 50-70
- Giảm image size xuống 512px
- Giảm batch size

### Overfitting?
- Tăng augmentation
- Thêm dropout
- Tăng validation set

---

**Lưu ý**: Cải thiện model cần thời gian và dữ liệu. Giải pháp nhanh (giảm confidence) có thể giúp ngay, nhưng để model tốt lâu dài cần retrain với dataset tốt hơn.


