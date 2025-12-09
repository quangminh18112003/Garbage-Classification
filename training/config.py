"""
Module cấu hình training - Quản lý tất cả các tham số training
"""
import os
from pathlib import Path

class TrainingConfig:
    """Class quản lý cấu hình training"""
    
    def __init__(self):
        # =============================
        # 📦 Model Configuration
        # =============================
        self.pretrained_model = 'yolov8n.pt'  # Model pretrained
        self.best_model_path = 'runs_train/exp_cpu/weights/best.pt'  # Đường dẫn model tốt nhất
        
        # =============================
        # 📊 Training Parameters
        # =============================
        self.data_yaml = 'data.yaml'  # File cấu hình dataset
        self.epochs = 10  # Số epoch (10 epochs cho training nhanh)
        self.imgsz = 640  # Kích thước ảnhcd training
        python train_10_epochs.py
        self.batch = 8  # Batch size
        self.seed = 666  # Seed để tái lặp kết quả
        
        # =============================
        # 🖥️ Device Configuration
        # =============================
        self.device = 'cpu'  # 'cpu' hoặc '0' (GPU)
        self.workers = 0  # Số workers (0 cho CPU)
        self.amp = False  # Automatic Mixed Precision (False cho CPU)
        self.cache = False  # Cache images
        
        # =============================
        # ⏹️ Early Stopping
        # =============================
        self.patience = 30  # Early stopping patience
        
        # =============================
        # 🎨 Data Augmentation
        # =============================
        self.hsv_h = 0.02  # Biến thiên Hue
        self.hsv_s = 0.7  # Biến thiên Saturation
        self.hsv_v = 0.4  # Biến thiên Value
        self.degrees = 45  # Độ xoay
        self.translate = 0.2  # Tịnh tiến
        self.scale = 0.5  # Scale
        self.shear = 10  # Shear
        self.flipud = 0.1  # Lật dọc
        self.fliplr = 0.5  # Flip ngang
        self.mosaic = 1.0  # Mosaic augmentation
        self.mixup = 0.1  # Mixup augmentation
        self.copy_paste = 0.1  # Copy-paste augmentation
        
        # =============================
        # 📁 Project Configuration
        # =============================
        self.project = 'runs_train'  # Thư mục project
        self.name = 'exp_modular'  # Tên experiment
        
        # =============================
        # 📝 Logging Configuration
        # =============================
        self.log_dir = Path(self.project) / self.name / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / 'training.log'
    
    def get_train_kwargs(self):
        """Trả về dictionary các tham số để truyền vào model.train()"""
        return {
            'data': self.data_yaml,
            'epochs': self.epochs,
            'imgsz': self.imgsz,
            'batch': self.batch,
            'seed': self.seed,
            'project': self.project,
            'name': self.name,
            'device': self.device,
            'workers': self.workers,
            'amp': self.amp,
            'cache': self.cache,
            'patience': self.patience,
            'hsv_h': self.hsv_h,
            'hsv_s': self.hsv_s,
            'hsv_v': self.hsv_v,
            'degrees': self.degrees,
            'translate': self.translate,
            'scale': self.scale,
            'shear': self.shear,
            'flipud': self.flipud,
            'fliplr': self.fliplr,
            'mosaic': self.mosaic,
            'mixup': self.mixup,
            'copy_paste': self.copy_paste,
        }
    
    def print_config(self):
        """In ra cấu hình training"""
        print("=" * 60)
        print("📋 CẤU HÌNH TRAINING")
        print("=" * 60)
        print(f"Model: {self.pretrained_model}")
        print(f"Epochs: {self.epochs}")
        print(f"Image Size: {self.imgsz}")
        print(f"Batch Size: {self.batch}")
        print(f"Device: {self.device}")
        print(f"Project: {self.project}/{self.name}")
        print("=" * 60)

