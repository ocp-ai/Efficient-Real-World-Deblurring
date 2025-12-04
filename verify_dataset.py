import os

def verify():
    base = r"D:\2025_PROJECT\Dataset_WinnerStyle"
    
    print("验证设置...")
    print("=" * 60)
    
    # Check index files
    for fname in ["RSBlur_real_train.txt", "RSBlur_real_test.txt"]:
        fpath = os.path.join(base, fname)
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                lines = f.readlines()
                if lines:
                    print(f"✓ {fname}: {len(lines)} 行")
                    print(f"  示例: {lines[0].strip()}")
        else:
            print(f"❌ 缺失: {fname}")
    
    # Check images
    input_count = len([f for f in os.listdir(os.path.join(base, "train", "input")) 
                       if f.endswith('.png')])
    target_count = len([f for f in os.listdir(os.path.join(base, "train", "target")) 
                        if f.endswith('.png')])
    
    print(f"\n图像统计:")
    print(f"  模糊图像 (input): {input_count}")
    print(f"  清晰图像 (target): {target_count}")
    
    if input_count == target_count:
        print("✓ 图像数量匹配!")
    else:
        print("❌ 图像数量不匹配!")

verify()