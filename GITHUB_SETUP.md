# 📚 Hướng Dẫn GitHub Setup

## 1. Khởi tạo Remote Repository

### Bước 1: Tạo repository trên GitHub
1. Đăng nhập vào https://github.com
2. Click "New Repository"
3. Đặt tên: `Garbage-Classification`
4. Chọn "Private" (tùy chọn)
5. Click "Create Repository"
6. Copy URL repository (HTTPS hoặc SSH)

### Bước 2: Kết nối local với remote

```bash
cd c:\python\Phanloairac

# Thêm remote repository
git remote add origin https://github.com/YOUR_USERNAME/Garbage-Classification.git

# Kiểm tra remote
git remote -v

# Push code lên GitHub (first time)
git branch -M main
git push -u origin main
```

## 2. Git Workflow cho Training

### Sau mỗi training, thực hiện:

```bash
# 1. Cập nhật training log
python commit_results.py

# 2. Kiểm tra changes
git status

# 3. Add files
git add .

# 4. Commit với message rõ ràng
git commit -m "Training round X: epoch Y completed, mAP: Z%"

# 5. Push lên GitHub
git push origin main
```

## 3. Commit Messages Best Practices

### Format:
```
Type: Mô tả ngắn

- Chi tiết 1
- Chi tiết 2
```

### Types:
- `feat`: Tính năng mới
- `fix`: Sửa lỗi
- `train`: Training updates
- `docs`: Documentation
- `chore`: Maintenance

### Ví dụ:
```
train: Complete epoch 50 with improved metrics

- Box loss: 1.234
- Cls loss: 0.567
- mAP50: 85.2%
- Dataset: 8918 training images
```

## 4. Kiểm Tra Training Progress

### View logs:
```bash
# Xem git log
git log --oneline

# Xem từng commit
git show <commit-hash>
```

## 5. Colaboration (Nếu làm nhóm)

### Clone repository:
```bash
git clone https://github.com/USERNAME/Garbage-Classification.git
cd Garbage-Classification
```

### Pull latest changes:
```bash
git pull origin main
```

### Tạo branch cho feature:
```bash
git checkout -b feature/your-feature-name
git add .
git commit -m "Add your feature"
git push origin feature/your-feature-name

# Tạo Pull Request trên GitHub
```

## 6. .gitignore Reminder

File `.gitignore` đã include:
- ✅ `*.pt` - Model weights
- ✅ `weights/` - Weight folder
- ✅ `runs/`, `runs_train/` - Training results
- ✅ `__pycache__/` - Cache files
- ✅ `.venv/` - Virtual environment

## 7. Troubleshooting

### Push bị từ chối?
```bash
git pull origin main --rebase
git push origin main
```

### Xóa file khỏi git (đã commit):
```bash
git rm --cached file_name
git commit -m "Remove file from tracking"
```

### Reset lại commit:
```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1
```

---

**Có thắc mắc?** Xem tài liệu GitHub: https://docs.github.com/en
