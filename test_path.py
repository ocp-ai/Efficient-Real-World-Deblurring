import os
# 用你修改后的 name 和路径进行测试
test_name = "My_Microscope_NAFNet_C16_L14"
test_dir = f"./models/{test_name}"
os.makedirs(test_dir, exist_ok=True)
print(f"测试目录创建成功：{test_dir}")