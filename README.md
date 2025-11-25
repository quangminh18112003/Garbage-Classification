# 🌍 Phân Loại Rác Thải bằng YOLO

Dự án phân loại rác thải thành 3 loại: **Glass (Kính)**, **Biological (Hữu cơ)**, **Plastic (Nhựa)** sử dụng YOLOv8.

## 📊 Dataset

- **Training images**: 8918 ảnh
- **Classes**: 3 (glass, biological, plastic)
- **Format**: YOLO format (ảnh + annotations)

## 🚀 Cài đặt

```bash
# Clone repository
git clone <repository-url>
cd Phanloairac

# Cài đặt dependencies
pip install -r requirements.txt
```

## 📦 Requirements

```
ultralytics>=8.0.0
streamlit>=1.28.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
pillow>=10.0.0
matplotlib>=3.8.0
pandas>=2.0.0
```

## 🎯 Training
See the full training methodology, how we train, and realtime inference tips in the dedicated guide: `TRAINING_GUIDE.md`.

```bash
cd training
python train.py
```

## 🖼️ Inference

```bash
streamlit run GiaoDien/app.py
```

## 📁 Cấu trúc Dự Án

```
Phanloairac/
├── dataset_split/          # Dataset được chia thành train/val/test
├── GiaoDien/              # Streamlit Web Interface
├── training/              # Training script
├── weights/               # Model weights
└── runs/                  # Training results
```

## 👤 Author

[Your Name]

## 📝 License

MIT License
