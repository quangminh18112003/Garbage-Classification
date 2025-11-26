# 🚀 Hướng Dẫn Chạy Chương Trình Phân Loại Rác Thải

## 📋 Yêu Cầu Hệ Thống

- Python 3.8 trở lên
- Windows/Linux/Mac
- Webcam (nếu muốn dùng tính năng webcam)

## 🔧 Bước 1: Cài Đặt Dependencies

### Mở Terminal/PowerShell tại thư mục dự án:

```powershell
# Di chuyển đến thư mục dự án
cd c:\python\Phanloairac

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

**Lưu ý**: Nếu bạn đang dùng virtual environment, hãy kích hoạt nó trước:
```powershell
# Nếu có virtual environment
.\GiaoDien\quan\Scripts\activate  # Windows
# hoặc
source venv/bin/activate  # Linux/Mac
```

## 📦 Bước 2: Kiểm Tra Model Weights

Chương trình cần file model `best.pt` trong thư mục `GiaoDien/weights/`

### Nếu chưa có model:

**Option 1: Copy từ training results (nếu đã training)**
```powershell
# Copy model từ kết quả training
copy training\runs_train\exp_cpu\weights\best.pt GiaoDien\weights\best.pt
```

**Option 2: Sử dụng pretrained model tạm thời**
```powershell
# Copy pretrained model làm best.pt (chỉ để test, không tốt bằng model đã train)
copy yolov8n.pt GiaoDien\weights\best.pt
```

**Option 3: Training model mới**
```powershell
cd training
python train.py
# Sau khi training xong, copy best.pt như Option 1
```

## 🎯 Bước 3: Chạy Chương Trình

### Cách 1: Chạy từ thư mục gốc (Khuyến nghị)

```powershell
# Từ thư mục c:\python\Phanloairac
streamlit run GiaoDien/app.py
```

### Cách 2: Chạy từ thư mục GiaoDien

```powershell
cd GiaoDien
streamlit run app.py
```

### Sau khi chạy lệnh:

1. Terminal sẽ hiển thị:
   ```
   You can now view your Streamlit app in your browser.
   Local URL: http://localhost:8501
   Network URL: http://192.168.x.x:8501
   ```

2. Trình duyệt sẽ tự động mở, hoặc bạn mở trình duyệt và truy cập: **http://localhost:8501**

## 🖼️ Bước 4: Sử Dụng Chương Trình

### Phát Hiện Từ Ảnh:

1. **Chọn nguồn**: Trong sidebar, chọn "Image"
2. **Upload ảnh**: Click "📂 Chọn ảnh..." và chọn file ảnh (.jpg, .png, .bmp, .webp)
3. **Cấu hình**:
   - **Độ tin cậy (%)**: Điều chỉnh từ 15-100% (mặc định 25%)
     - Thấp hơn = phát hiện nhiều hơn (có thể có false positives)
     - Cao hơn = chính xác hơn (có thể bỏ sót)
   - **Kích thước ảnh (px)**: 320-1280px (mặc định 640px)
     - Lớn hơn = phát hiện tốt hơn nhưng chậm hơn
4. **Phát hiện**: Click nút "🚀 Phát hiện đối tượng"
5. **Xem kết quả**:
   - Ảnh với bounding boxes
   - Bảng chi tiết các đối tượng phát hiện được
   - Biểu đồ số lượng từng class

### Phát Hiện Từ Webcam:

1. **Chọn nguồn**: Trong sidebar, chọn "Webcam"
2. **Cấu hình**: Điều chỉnh confidence và image size như trên
3. **Bắt đầu**: Click "Start Webcam"
4. **Dừng**: Click "Stop Webcam"
5. **Xem kết quả**: 
   - Video real-time với bounding boxes
   - Bảng và biểu đồ cập nhật theo thời gian thực

## ⚙️ Cấu Hình Nâng Cao

### Thay Đổi Port (nếu port 8501 bị chiếm):

```powershell
streamlit run GiaoDien/app.py --server.port 8502
```

### Chạy trên Network (để truy cập từ máy khác):

```powershell
streamlit run GiaoDien/app.py --server.address 0.0.0.0
```

## 🐛 Xử Lý Lỗi Thường Gặp

### Lỗi 1: "File mô hình không tồn tại"

**Nguyên nhân**: Thiếu file `best.pt` trong `GiaoDien/weights/`

**Giải pháp**:
```powershell
# Kiểm tra file có tồn tại không
dir GiaoDien\weights\best.pt

# Nếu không có, copy từ training hoặc pretrained
copy yolov8n.pt GiaoDien\weights\best.pt
```

### Lỗi 2: "ModuleNotFoundError"

**Nguyên nhân**: Thiếu thư viện

**Giải pháp**:
```powershell
pip install -r requirements.txt
```

### Lỗi 3: "Webcam không mở được"

**Nguyên nhân**: 
- Webcam đang được sử dụng bởi ứng dụng khác
- Webcam không được kết nối
- Quyền truy cập bị từ chối

**Giải pháp**:
- Đóng các ứng dụng khác đang dùng webcam
- Kiểm tra kết nối webcam
- Chạy với quyền Administrator (Windows)

### Lỗi 4: "CUDA out of memory" (nếu dùng GPU)

**Giải pháp**:
- Giảm image size xuống 416 hoặc 320
- Giảm batch size trong training
- Hoặc chuyển sang CPU: `device='cpu'` trong code

## 📊 Tips Để Phát Hiện Tốt Hơn

1. **Với ảnh rác trên nước**:
   - Giảm confidence xuống 20-25%
   - Tăng image size lên 640-800px
   - Đảm bảo ảnh có độ phân giải tốt

2. **Với ảnh rõ ràng**:
   - Confidence: 30-40%
   - Image size: 640px là đủ

3. **Nếu phát hiện quá nhiều (false positives)**:
   - Tăng confidence lên 40-50%

4. **Nếu bỏ sót nhiều đối tượng**:
   - Giảm confidence xuống 15-20%
   - Tăng image size lên 800-1024px

## 🔄 Dừng Chương Trình

- Nhấn `Ctrl + C` trong terminal
- Hoặc đóng tab trình duyệt và dừng terminal

## 📝 Lưu Ý Quan Trọng

1. **Lần đầu chạy**: Model sẽ được tải vào memory, có thể mất 10-30 giây
2. **Inference speed**: 
   - CPU: ~1-3 giây/ảnh
   - GPU: ~0.1-0.5 giây/ảnh
3. **Model tốt nhất**: Sử dụng model đã được training với dataset của bạn, không phải pretrained model

## 🎓 Training Model Mới (Tùy chọn)

Nếu muốn cải thiện độ chính xác:

```powershell
cd training
python train.py
# Hoặc với cấu hình cải thiện
python train_improved.py

# Sau khi training xong
copy training\runs_train\exp_cpu\weights\best.pt GiaoDien\weights\best.pt
```

---

**Chúc bạn sử dụng thành công!** 🎉

Nếu gặp vấn đề, hãy kiểm tra:
- File `best.pt` có tồn tại không
- Đã cài đặt đủ dependencies chưa
- Python version >= 3.8

