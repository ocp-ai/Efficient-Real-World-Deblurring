import os

# ========== 配置：修改为你自己的路径 ==========
DATA_ROOT = r"D:\2025_PROJECT\Dataset_WinnerStyle"
# ===========================================

def create_index_for_subdir(subdir_name, output_filename):
    """
    为指定的子目录创建索引文件。
    例如: subdir_name="train/target", output_filename="RSBlur_real_train.txt"
    """
    image_dir = os.path.join(DATA_ROOT, subdir_name)
    
    if not os.path.exists(image_dir):
        print(f"[跳过] 目录不存在: {image_dir}")
        return False
    
    # 获取所有图片文件
    image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(image_extensions)]
    image_files.sort()
    
    if not image_files:
        print(f"[警告] 目录中没有图片: {image_dir}")
        return False
    
    # 写入文件
    output_path = os.path.join(DATA_ROOT, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        for img_file in image_files:
            f.write(f"{img_file}\n")
    
    print(f"  ✅ 已创建: {output_filename} ({len(image_files)}个图像)")
    return True

def main():
    print("=" * 60)
    print("正在生成数据加载器所需的所有索引文件")
    print(f"数据根目录: {DATA_ROOT}")
    print("=" * 60)
    
    # 首先，检查你的数据目录结构
    print("\n[1] 检查数据目录结构...")
    if not os.path.exists(DATA_ROOT):
        print(f"❌ 错误: 数据根目录不存在: {DATA_ROOT}")
        return
    
    # 列出所有子目录，了解数据结构
    for item in os.listdir(DATA_ROOT):
        item_path = os.path.join(DATA_ROOT, item)
        if os.path.isdir(item_path):
            print(f"  发现目录: {item}/")
            # 列出子目录的内容
            for sub_item in os.listdir(item_path):
                sub_path = os.path.join(item_path, sub_item)
                if os.path.isdir(sub_path):
                    print(f"    ├── {sub_item}/")
    
    # 基于常见的RSBlur数据集结构，创建可能的索引文件
    print("\n[2] 创建索引文件...")
    
    # 可能的组合（根据错误信息推断）
    # 训练集
    create_index_for_subdir("train/target", "RSBlur_real_train.txt")
    create_index_for_subdir("train/input", "RSBlur_blur_train.txt")
    
    # 验证集/测试集
    # 注意：如果你的数据集没有单独的验证集，可以用训练集的一部分代替
    # 这里我们先检查是否存在验证集目录
    if os.path.exists(os.path.join(DATA_ROOT, "valid")):
        create_index_for_subdir("valid/target", "RSBlur_real_valid.txt")
        create_index_for_subdir("valid/input", "RSBlur_blur_valid.txt")
        create_index_for_subdir("valid/target", "RSBlur_real_test.txt")
        create_index_for_subdir("valid/input", "RSBlur_blur_test.txt")
    elif os.path.exists(os.path.join(DATA_ROOT, "test")):
        create_index_for_subdir("test/target", "RSBlur_real_test.txt")
        create_index_for_subdir("test/input", "RSBlur_blur_test.txt")
    else:
        # 如果没有单独的验证集，使用训练集的前N个作为验证（常见做法）
        print("\n[提示] 未找到 'valid' 或 'test' 目录。")
        print("      将使用训练集的前100个图像创建验证集索引...")
        
        # 创建验证集索引（使用训练集的前100个文件）
        train_target_dir = os.path.join(DATA_ROOT, "train/target")
        if os.path.exists(train_target_dir):
            image_files = [f for f in os.listdir(train_target_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_files.sort()
            
            if len(image_files) > 100:
                # 取前100个作为验证集
                valid_files = image_files[:100]
                output_path = os.path.join(DATA_ROOT, "RSBlur_real_test.txt")
                with open(output_path, 'w', encoding='utf-8') as f:
                    for img_file in valid_files:
                        f.write(f"{img_file}\n")
                print(f"  ✅ 已创建: RSBlur_real_test.txt ({len(valid_files)}个图像)")
                
                # 同样为模糊图像创建
                train_input_dir = os.path.join(DATA_ROOT, "train/input")
                if os.path.exists(train_input_dir):
                    blur_files = [f for f in os.listdir(train_input_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    blur_files.sort()
                    if len(blur_files) > 100:
                        valid_blur_files = blur_files[:100]
                        output_path = os.path.join(DATA_ROOT, "RSBlur_blur_test.txt")
                        with open(output_path, 'w', encoding='utf-8') as f:
                            for img_file in valid_blur_files:
                                f.write(f"{img_file}\n")
                        print(f"  ✅ 已创建: RSBlur_blur_test.txt ({len(valid_blur_files)}个图像)")
    
    print("\n" + "=" * 60)
    print("✅ 索引文件创建完成！")
    print("=" * 60)
    
    # 显示生成的文件
    print("\n生成的索引文件:")
    for file in os.listdir(DATA_ROOT):
        if file.endswith('.txt'):
            file_path = os.path.join(DATA_ROOT, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = len(f.readlines())
            print(f"  📄 {file} ({line_count}行)")

if __name__ == "__main__":
    main()