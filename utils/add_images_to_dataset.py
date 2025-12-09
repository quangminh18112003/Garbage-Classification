"""
Script hỗ trợ thêm ảnh mới vào dataset
Sử dụng: python utils/add_images_to_dataset.py --source_dir path/to/new/images --target train
"""
import argparse
import shutil
from pathlib import Path

def add_images(source_dir, target_split='train'):
    """
    Copy ảnh từ thư mục nguồn vào dataset_split
    
    Args:
        source_dir: Đường dẫn thư mục chứa ảnh mới
        target_split: 'train', 'val', hoặc 'test'
    """
    source_path = Path(source_dir)
    dataset_root = Path('dataset_split')
    
    if not source_path.exists():
        print(f"❌ Thư mục không tồn tại: {source_path}")
        return
    
    if target_split not in ['train', 'val', 'test']:
        print(f"❌ Target split không hợp lệ: {target_split}. Chọn 'train', 'val', hoặc 'test'")
        return
    
    # Đường dẫn đích
    images_dest = dataset_root / 'images' / target_split
    images_dest.mkdir(parents=True, exist_ok=True)
    
    # Tìm tất cả ảnh
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(source_path.glob(f'*{ext}')))
        image_files.extend(list(source_path.glob(f'*{ext.upper()}')))
    
    if not image_files:
        print(f"⚠️ Không tìm thấy ảnh nào trong {source_path}")
        return
    
    print(f"📸 Tìm thấy {len(image_files)} ảnh")
    print(f"📁 Đích: {images_dest}")
    
    # Copy ảnh
    copied = 0
    for img_file in image_files:
        dest_file = images_dest / img_file.name
        
        # Tránh ghi đè
        if dest_file.exists():
            print(f"⚠️ File đã tồn tại, bỏ qua: {img_file.name}")
            continue
        
        shutil.copy2(img_file, dest_file)
        copied += 1
        print(f"✅ Đã copy: {img_file.name}")
    
    print(f"\n✨ Hoàn thành! Đã thêm {copied}/{len(image_files)} ảnh vào {target_split}")
    print(f"💡 Bước tiếp theo: Sử dụng LabelImg để tạo annotations (.txt files)")
    print(f"   LabelImg: pip install labelimg && labelimg")
    print(f"   Mở thư mục: {images_dest}")
    print(f"   Format: YOLO")
    print(f"   Lưu labels vào: {dataset_root / 'labels' / target_split}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Thêm ảnh mới vào dataset')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Đường dẫn thư mục chứa ảnh mới')
    parser.add_argument('--target', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Thêm vào train/val/test (mặc định: train)')
    
    args = parser.parse_args()
    add_images(args.source_dir, args.target)





