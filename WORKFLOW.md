# ✅ Training & GitHub Workflow Checklist

## Setup Hoàn Thành

### ✅ Project Structure

### ✅ Git Repository

### ✅ Training Setup

### ✅ Documentation

## Hướng Dẫn Tiếp Theo

#### Hướng dẫn benchmark FPS:
```powershell
python benchmark_inference.py --model_type onnx --model_path training/best.onnx --image_dir dataset_split/images/test
python benchmark_inference.py --model_type pt --model_path training/best.pt --image_dir dataset_split/images/test --device cpu
```
Kết quả sẽ được ghi vào `benchmark_results.txt` để so sánh tốc độ ONNX vs PyTorch.
```bash
# 1. Truy cập https://github.com/new
# 2. Đặt tên: Garbage-Classification
# 3. Copy URL SSH hoặc HTTPS
# 4. Chạy lệnh:
git remote add origin YOUR_REPO_URL
git branch -M main
git push -u origin main
```

### 📌 Bước 2: Theo dõi Training
```bash
# Monitor training progress
cd c:\python\Phanloairac\training
# Training đang chạy ở background
# Kết quả lưu ở: training/runs_train/exp2/
```

### 📌 Bước 3: Sau khi Training Hoàn Thành

#### a) Kiểm tra Results:
```bash
# Xem best model metrics
cat training/runs_train/exp2/results.csv | tail -5

# Xem weights
ls -la training/runs_train/exp2/weights/
```

#### b) Commit Results:
```bash
# 1. Run script để tạo summary
python commit_results.py

# 2. Hoặc manual commit
git add training/runs_train/exp2/
git commit -m "train: Complete epoch 150, mAP50: XX.X%"
git push origin main
```

#### c) Copy Best Weights:
```bash
# Copy best model to GiaoDien
cp training/runs_train/exp2/weights/best.pt GiaoDien/weights/best.pt

# Commit weight update
git add GiaoDien/weights/best.pt
git commit -m "Update: Best model after training"
git push origin main
```

### 📌 Bước 4: Test Model

```bash
# Start Streamlit app
cd GiaoDien
streamlit run app.py

# Test with image/webcam
```

### Optional: Export ONNX + run model server + Streamlit client

```powershell
# Export ONNX (example)
python training/export_to_onnx.py --weights training/runs_train/exp2/weights/best.pt --output models/best.onnx --imgsz 416

# Run server (loads ONNX if available)
uvicorn server.api:app --host 0.0.0.0 --port 8000

# Run streamlit client
streamlit run GiaoDien/app_server.py
```

This setup moves heavy compute (model inference) to a separate process/server and keeps Streamlit lightweight for UI. It helps achieve smoother realtime UX and lets you move the model server to a GPU-enabled machine later.

### 📌 Bước 5: Update Documentation

Cập nhật README.md với:
- Final metrics
- Training time
- Dataset statistics
- Performance on test set

```bash
# Example commit
git add README.md
git commit -m "docs: Add final training metrics and results"
git push origin main
```

## Daily Training Workflow

```bash
# Mỗi ngày training:
cd c:\python\Phanloairac

# 1. Check status
git status

# 2. Xem tiến độ training (nếu vẫn chạy)
# tail -20 training/runs_train/exp2/train_output.log

# 3. Khi training xong:
python commit_results.py
git add .
git commit -m "train: Session X - $(date +%Y-%m-%d)"
git push origin main
```

## Git Commands Quick Reference

```bash
# Status
git status

# View logs
git log --oneline -10

# View specific commit
git show <commit-id>

# Undo last commit
git reset --soft HEAD~1

# Push changes
git push origin main

# Pull latest
git pull origin main

# Create backup branch
git branch backup-$(date +%Y-%m-%d)
```

## Important Files Locations

```
Phanloairac/
├── training/
│   ├── train.py              # Training script
│   ├── data.yaml             # Dataset config
│   └── runs_train/
│       └── exp2/             # Latest results
│           ├── weights/best.pt
│           └── results.csv
├── GiaoDien/
│   ├── app.py                # Streamlit app
│   └── weights/              # Model weights
├── TRAINING_LOG.md           # Logs
├── GITHUB_SETUP.md           # This file
├── README.md                 # Project info
└── .git/                     # Git repo
```

## Troubleshooting

### Training bị timeout?
```bash
# Resume từ last checkpoint
# Thêm vào train.py: resume=True
```

### Git push bị từ chối?
```bash
git pull origin main --rebase
git push origin main
```

### Model weights quá lớn (>100MB)?
```bash
# Sử dụng Git LFS
git lfs install
git lfs track "*.pt"
git add .gitattributes
```

---

## 🎯 Training Metrics Target

- **mAP50**: > 75%
- **mAP50-95**: > 50%
- **Box Loss**: < 1.5
- **Cls Loss**: < 0.5

---

**Last Updated**: November 24, 2025
**Status**: ✅ Setup Complete - Training Running
