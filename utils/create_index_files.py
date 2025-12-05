import os

# ========== 配置：修改为你自己的路径 ==========
# 你的数据根目录
DATA_ROOT = r"D:\2025_PROJECT\Dataset_WinnerStyle"
# 根据你的文件夹结构，清晰图（real）和模糊图（blur）所在的子目录
CLEAR_IMAGES_DIR = "train/target"   # 清晰图像 (ground truth)
BLURRY_IMAGES_DIR = "train/input"   # 模糊/退化图像 (input)
# ===========================================

def create_index_txt(data_root, image_subdir, output_filename):
    """
    扫描指定子目录下的所有图片，生成索引文件。
    """
    image_dir = os.path.join(data_root, image_subdir)
    if not os.path.exists(image_dir):
        print(f"[错误] 目录不存在: {image_dir}")
        return False
    
    # 获取所有支持的图片文件
    image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]
    image_files.sort()  # 按文件名排序以保证一致性
    
    if not image_files:
        print(f"[警告] 目录中没有找到图片: {image_dir}")
        return False
    
    # 写入文本文件，每行一个文件名（不含路径）
    output_path = os.path.join(data_root, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        for img_file in image_files:
            f.write(f"{img_file}\n")
    
    print(f"[成功] 已创建 {output_filename}，包含 {len(image_files)} 个图像。")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("生成数据加载器所需的索引文件 (.txt)")
    print(f"数据根目录: {DATA_ROOT}")
    print("=" * 60)
    
    # 创建训练集索引文件（根据你的具体需求，可能需要创建多个）
    # 示例：为清晰图像（target）创建索引
    success_clear = create_index_txt(DATA_ROOT, CLEAR_IMAGES_DIR, "RSBlur_real_train.txt")
    # 示例：为模糊图像（input）创建索引
    success_blur = create_index_txt(DATA_ROOT, BLURRY_IMAGES_DIR, "RSBlur_blur_train.txt")
    
    if success_clear or success_blur:
        print("\n[操作完成] 请重新运行训练命令。")
    else:
        print("\n[操作失败] 请检查上述错误信息。")