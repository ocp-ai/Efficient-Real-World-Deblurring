# fix_distributed.py
import os

print("Fixing distributed training initialization...")
print("="*60)

train_py = "./train.py"
backup_py = "./train.py.backup_distributed"

# Create backup
if os.path.exists(train_py):
    import shutil
    shutil.copy2(train_py, backup_py)
    print(f"✅ Backed up to: {backup_py}")

# Read file
with open(train_py, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

# Find the distributed section (look for "分布式环境初始化")
new_lines = []
in_distributed_section = False
replaced = False

for i, line in enumerate(lines):
    # Look for start of distributed section
    if "分布式环境初始化" in line or "dist.init_process_group" in line:
        in_distributed_section = True
    
    # Replace the entire problematic section
    if in_distributed_section and not replaced:
        # Skip old distributed code
        if "torch.cuda.set_device" in line or "print(f'进程状态" in line:
            # We've reached the end of distributed section
            in_distributed_section = False
            # Add the fixed code
            fixed_code = '''# ========== SIMPLE FIX: Disable distributed entirely ==========
world_size = 1
global_rank = 0
local_rank = 0

print("[单卡模式] 分布式训练已完全禁用。")

# Set environment variables to prevent errors
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

# Skip all distributed initialization
import torch.distributed as dist
if dist.is_initialized():
    dist.destroy_process_group()
# =================================================

# 设置当前进程使用的GPU
torch.cuda.set_device(local_rank)
print(f'进程状态: 全局排名 {global_rank} / 总进程数 {world_size}, 本地GPU编号 {local_rank}')'''
            
            new_lines.append(fixed_code)
            replaced = True
            continue
    
    if not in_distributed_section or replaced:
        new_lines.append(line)

# Write back
with open(train_py, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✅ Fixed distributed initialization in train.py")
print("\n" + "="*60)
print("Now try running training again:")
print("python train.py -p options/train/RSBlur.yml")
print("="*60)