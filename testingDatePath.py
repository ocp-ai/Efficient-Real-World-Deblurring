
import os
path = r'D:\2025_PROJECT\Dataset_WinnerStyle'
if os.path.exists(path):
    print(f'✅ 路径存在: {path}')
    # 检查关键子文件夹
    required = ['train/input', 'train/target']
    for req in required:
        full = os.path.join(path, req)
        if os.path.exists(full):
            count = len([f for f in os.listdir(full) if f.endswith('.png')])
            print(f'  ✅ {req}: {count} 张图片')
        else:
            print(f'  ❌ {req}: 文件夹不存在')
else:
    print(f'❌ 路径不存在: {path}')
