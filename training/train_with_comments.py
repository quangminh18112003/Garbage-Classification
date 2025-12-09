"""
Script training YOLO với comment chi tiết từng dòng
Giải thích rõ ràng mỗi dòng code làm gì
"""
import os  # Module để làm việc với hệ điều hành (tạo thư mục, kiểm tra file, v.v.)
from pathlib import Path  # Module để làm việc với đường dẫn file/thư mục một cách dễ dàng
from ultralytics import YOLO  # Import YOLO từ thư viện Ultralytics để train model

def main():
    """
    Hàm main - hàm chính để chạy toàn bộ quá trình training
    """
    
    # =============================
    # BƯỚC 1: THIẾT LẬP MÔI TRƯỜNG
    # =============================
    
    # Đảm bảo script chạy từ đúng thư mục training
    # Path(__file__) lấy đường dẫn file hiện tại
    # .parent lấy thư mục cha (thư mục training)
    # os.chdir() đổi thư mục làm việc hiện tại
    os.chdir(Path(__file__).parent)
    
    # In thông báo bắt đầu training
    print("🚀 Bắt đầu training model YOLO...")
    print("=" * 60)  # In dòng phân cách gồm 60 dấu =
    
    # =============================
    # BƯỚC 2: THIẾT LẬP SEED
    # =============================
    
    # Seed là số ngẫu nhiên cố định để đảm bảo kết quả có thể tái lặp
    # Nếu dùng cùng seed, training sẽ cho kết quả giống nhau
    seed = 666  # Gán seed = 666
    
    # =============================
    # BƯỚC 3: CHỌN MODEL PRETRAINED
    # =============================
    
    # YOLOv8 có nhiều phiên bản: n (nano), s (small), m (medium), l (large), x (xlarge)
    # yolov8n.pt là phiên bản nhỏ nhất, nhanh nhất, phù hợp cho CPU
    pretrained_model = 'yolov8n.pt'  # Đường dẫn file model pretrained
    
    # Kiểm tra xem file model có tồn tại không
    # Path() tạo đối tượng đường dẫn
    # .exists() kiểm tra file có tồn tại không
    if not Path(pretrained_model).exists():
        # Nếu không tồn tại, in thông báo lỗi
        print(f"❌ Không tìm thấy file model: {pretrained_model}")
        print("💡 Hãy đảm bảo file yolov8n.pt có trong thư mục training")
        return  # Dừng chương trình nếu không có model
    
    # =============================
    # BƯỚC 4: KIỂM TRA MODEL ĐÃ TRAIN (ĐỂ TIẾP TỤC TRAINING)
    # =============================
    
    # Đường dẫn đến model tốt nhất từ lần training trước (nếu có)
    # runs_train/exp_cpu/weights/best.pt là nơi YOLO lưu model tốt nhất
    best_model_path = 'runs_train/exp_cpu/weights/best.pt'
    
    # Kiểm tra xem có model đã train chưa
    if os.path.exists(best_model_path):  # os.path.exists() kiểm tra file có tồn tại
        # Nếu có, in thông báo và load model đó để tiếp tục training
        print(f"📦 Tìm thấy model đã train, tiếp tục từ: {best_model_path}")
        model = YOLO(best_model_path)  # YOLO() load model từ file .pt
    else:
        # Nếu không có, load model pretrained mới
        print(f"🆕 Khởi tạo model mới từ: {pretrained_model}")
        model = YOLO(pretrained_model)  # Load model pretrained
    
    # =============================
    # BƯỚC 5: HIỂN THỊ THÔNG TIN MODEL
    # =============================
    
    # In thông tin về model đã load
    print("\n🤖 Thông tin Model:")
    print(f"  - Số classes: {len(model.names)}")  # model.names là dict chứa tên các class
    print(f"  - Tên classes: {', '.join(model.names.values())}")  # In tên các class
    print("=" * 60)
    
    # =============================
    # BƯỚC 6: THIẾT LẬP CÁC THAM SỐ TRAINING
    # =============================
    
    print("\n⚙️ Thiết lập tham số training...")
    
    # data='data.yaml' - File cấu hình dataset
    # File này chứa đường dẫn đến train/val/test và tên các class
    data_file = 'data.yaml'
    
    # epochs=10 - Số lần model sẽ học toàn bộ dataset
    # Mỗi epoch = 1 lần duyệt qua toàn bộ dữ liệu training
    # ⚠️ CÓ THỂ TĂNG LÊN (20, 30, 50, 100) nếu muốn training lâu hơn
    epochs = 10  # 10 epochs cho training nhanh
    
    # imgsz=640 - Kích thước ảnh đầu vào (640x640 pixels)
    # Ảnh sẽ được resize về kích thước này trước khi đưa vào model
    # Số lớn hơn = chất lượng tốt hơn nhưng chậm hơn
    imgsz = 640
    
    # batch=8 - Số ảnh xử lý cùng lúc trong 1 lần
    # Batch lớn = nhanh hơn nhưng cần nhiều RAM/VRAM hơn
    # Với CPU nên dùng batch nhỏ (4-8), với GPU có thể dùng lớn hơn (16-32)
    batch = 8
    
    # device='cpu' - Thiết bị để train
    # 'cpu' = dùng CPU (chậm nhưng không cần GPU)
    # '0' = dùng GPU đầu tiên (nhanh hơn nhiều nếu có GPU)
    device = 'cpu'
    
    # workers=0 - Số luồng để load dữ liệu
    # 0 = không dùng đa luồng (an toàn cho CPU)
    # Số lớn hơn = load nhanh hơn nhưng tốn RAM
    workers = 0
    
    # amp=False - Automatic Mixed Precision
    # False = tắt (cần cho CPU)
    # True = bật (tăng tốc trên GPU, giảm bộ nhớ)
    amp = False
    
    # cache=False - Có cache ảnh vào RAM không
    # False = không cache (tiết kiệm RAM)
    # True = cache (nhanh hơn nhưng tốn RAM)
    cache = False
    
    # patience=5 - Early stopping patience
    # Nếu mAP không cải thiện trong 5 epochs liên tiếp, dừng training
    # Giúp tránh overfitting và tiết kiệm thời gian
    # Giảm xuống 5 vì chỉ train 10 epochs
    patience = 5
    
    # project='runs_train' - Tên thư mục chứa kết quả training
    project = 'runs_train'
    
    # name='exp_detailed' - Tên thư mục con chứa experiment này
    # Kết quả sẽ lưu tại: runs_train/exp_detailed/
    name = 'exp_detailed'
    
    # =============================
    # BƯỚC 7: THIẾT LẬP DATA AUGMENTATION
    # =============================
    # Data augmentation = tạo thêm dữ liệu bằng cách biến đổi ảnh
    # Giúp model học tốt hơn, tránh overfitting
    
    print("\n🎨 Thiết lập Data Augmentation...")
    
    # hsv_h=0.02 - Thay đổi màu sắc (Hue) trong khoảng ±2%
    # Giúp model học được ảnh với ánh sáng/ màu sắc khác nhau
    hsv_h = 0.02
    
    # hsv_s=0.7 - Thay đổi độ bão hòa màu (Saturation) ±70%
    # Giúp model học được ảnh có màu đậm/nhạt khác nhau
    hsv_s = 0.7
    
    # hsv_v=0.4 - Thay đổi độ sáng (Value) ±40%
    # Giúp model học được ảnh sáng/tối khác nhau
    hsv_v = 0.4
    
    # degrees=45 - Xoay ảnh ngẫu nhiên trong khoảng ±45 độ
    # Giúp model học được đối tượng ở các góc độ khác nhau
    degrees = 45
    
    # translate=0.2 - Dịch chuyển ảnh ngẫu nhiên ±20%
    # Giúp model học được đối tượng ở các vị trí khác nhau trong ảnh
    translate = 0.2
    
    # scale=0.5 - Phóng to/thu nhỏ ảnh ngẫu nhiên ±50%
    # Giúp model học được đối tượng có kích thước khác nhau
    scale = 0.5
    
    # shear=10 - Làm méo ảnh (shear) ±10 độ
    # Giúp model học được đối tượng bị biến dạng
    shear = 10
    
    # flipud=0.1 - Xác suất lật ảnh theo chiều dọc (10%)
    # Giúp model học được đối tượng bị lật
    flipud = 0.1
    
    # fliplr=0.5 - Xác suất lật ảnh theo chiều ngang (50%)
    # Giúp model học được đối tượng ở cả 2 bên trái/phải
    fliplr = 0.5
    
    # mosaic=1.0 - Xác suất dùng mosaic augmentation (100%)
    # Mosaic = ghép 4 ảnh thành 1 ảnh lớn
    # Giúp model học được nhiều đối tượng cùng lúc
    mosaic = 1.0
    
    # mixup=0.1 - Xác suất dùng mixup augmentation (10%)
    # Mixup = trộn 2 ảnh với nhau
    # Giúp model học được các trường hợp trung gian
    mixup = 0.1
    
    # copy_paste=0.1 - Xác suất copy-paste augmentation (10%)
    # Copy một phần ảnh này dán vào ảnh khác
    # Giúp model học được nhiều đối tượng trong 1 ảnh
    copy_paste = 0.1
    
    # =============================
    # BƯỚC 8: BẮT ĐẦU TRAINING
    # =============================
    
    print("\n🎯 Bắt đầu training...")
    print("=" * 60)
    print("📝 Thông tin training:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Image size: {imgsz}x{imgsz}")
    print(f"  - Batch size: {batch}")
    print(f"  - Device: {device}")
    print(f"  - Data file: {data_file}")
    print("=" * 60)
    print("\n⏳ Training đang chạy, vui lòng đợi...\n")
    
    # model.train() - Hàm chính để bắt đầu training
    # Tất cả các tham số được truyền vào bằng keyword arguments
    results = model.train(
        data=data_file,          # File cấu hình dataset
        epochs=epochs,           # Số epochs
        imgsz=imgsz,             # Kích thước ảnh
        batch=batch,             # Batch size
        seed=seed,               # Seed để tái lặp kết quả
        project=project,         # Thư mục project
        name=name,               # Tên experiment
        device=device,           # Thiết bị (CPU/GPU)
        workers=workers,         # Số workers
        amp=amp,                 # Mixed precision
        cache=cache,             # Cache images
        patience=patience,       # Early stopping patience
        # Các tham số augmentation
        hsv_h=hsv_h,            # Hue augmentation
        hsv_s=hsv_s,            # Saturation augmentation
        hsv_v=hsv_v,            # Value augmentation
        degrees=degrees,        # Rotation augmentation
        translate=translate,    # Translation augmentation
        scale=scale,            # Scale augmentation
        shear=shear,            # Shear augmentation
        flipud=flipud,          # Vertical flip
        fliplr=fliplr,          # Horizontal flip
        mosaic=mosaic,          # Mosaic augmentation
        mixup=mixup,            # Mixup augmentation
        copy_paste=copy_paste,  # Copy-paste augmentation
    )
    
    # results = kết quả training (chứa metrics, model path, v.v.)
    # Sau khi training xong, YOLO tự động:
    # - Lưu model tốt nhất tại: runs_train/exp_detailed/weights/best.pt
    # - Lưu model cuối cùng tại: runs_train/exp_detailed/weights/last.pt
    # - Lưu metrics tại: runs_train/exp_detailed/results.csv
    # - Lưu plots tại: runs_train/exp_detailed/
    
    # =============================
    # BƯỚC 9: HIỂN THỊ KẾT QUẢ
    # =============================
    
    print("\n" + "=" * 60)
    print("✅ Training hoàn tất!")
    print("=" * 60)
    
    # Kiểm tra xem có file results.csv không
    results_csv = Path(project) / name / 'results.csv'
    if results_csv.exists():
        print(f"\n📊 Kết quả training đã được lưu tại:")
        print(f"  - CSV: {results_csv}")
        print(f"  - Best model: {Path(project) / name / 'weights' / 'best.pt'}")
        print(f"  - Last model: {Path(project) / name / 'weights' / 'last.pt'}")
    
    # In thông báo hướng dẫn
    print("\n💡 Để xem kết quả chi tiết:")
    print(f"   - Xem plots: {Path(project) / name}")
    print(f"   - Xem metrics: {results_csv}")
    print(f"   - Sử dụng model: {Path(project) / name / 'weights' / 'best.pt'}")
    
    print("\n" + "=" * 60)


# =============================
# ĐIỂM VÀO CHƯƠNG TRÌNH
# =============================

# if __name__ == "__main__": 
# - Kiểm tra xem file này có được chạy trực tiếp không
# - Nếu có (không phải import), thì chạy hàm main()
# - Đây là cách Python chuẩn để tạo script có thể chạy độc lập
if __name__ == "__main__":
    main()  # Gọi hàm main() để bắt đầu training

