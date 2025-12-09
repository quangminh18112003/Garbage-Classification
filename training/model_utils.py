"""
Module quản lý model - Load, save và quản lý model
"""
import os
from pathlib import Path
from ultralytics import YOLO
from typing import Optional

class ModelManager:
    """Class quản lý model loading và saving"""
    
    def __init__(self, pretrained_model: str, best_model_path: Optional[str] = None):
        self.pretrained_model = pretrained_model
        self.best_model_path = best_model_path
        self.model = None
    
    def load_model(self) -> YOLO:
        """Load model từ pretrained hoặc best model"""
        if self.best_model_path and os.path.exists(self.best_model_path):
            print(f"📦 Tiếp tục training từ: {self.best_model_path}")
            self.model = YOLO(self.best_model_path)
        else:
            print(f"🆕 Khởi tạo từ pretrained: {self.pretrained_model}")
            self.model = YOLO(self.pretrained_model)
        
        return self.model
    
    def get_model_info(self) -> dict:
        """Lấy thông tin về model"""
        if self.model is None:
            return {}
        
        info = {
            'model_name': self.model.model_name if hasattr(self.model, 'model_name') else 'Unknown',
            'num_classes': len(self.model.names) if hasattr(self.model, 'names') else 0,
            'class_names': list(self.model.names.values()) if hasattr(self.model, 'names') else [],
        }
        
        # Đếm số parameters nếu có
        if hasattr(self.model, 'model'):
            try:
                total_params = sum(p.numel() for p in self.model.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
                info['total_parameters'] = total_params
                info['trainable_parameters'] = trainable_params
            except:
                pass
        
        return info
    
    def print_model_info(self):
        """In thông tin model"""
        info = self.get_model_info()
        print("\n" + "=" * 60)
        print("🤖 THÔNG TIN MODEL")
        print("=" * 60)
        for key, value in info.items():
            if isinstance(value, list):
                print(f"{key}: {', '.join(map(str, value))}")
            else:
                print(f"{key}: {value}")
        print("=" * 60 + "\n")
    
    def save_checkpoint(self, path: str):
        """Lưu checkpoint của model"""
        if self.model is None:
            print("⚠️ Model chưa được load")
            return
        
        try:
            # Tạo thư mục nếu chưa có
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(path)
            print(f"✅ Đã lưu checkpoint tại: {path}")
        except Exception as e:
            print(f"❌ Lỗi khi lưu checkpoint: {e}")


