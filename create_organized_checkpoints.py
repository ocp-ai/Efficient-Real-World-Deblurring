# quick_fix.py
import os

print("Quick fix for checkpoint saving...")
print("="*60)

# 1. Create directories
os.makedirs("./checkpoints/latest", exist_ok=True)
os.makedirs("./checkpoints/by_epoch", exist_ok=True)
os.makedirs("./checkpoints/best", exist_ok=True)
print("✅ Created checkpoint directories")

# 2. Update YAML with SIMPLE path
yaml_path = "./options/train/RSBlur.yml"
if os.path.exists(yaml_path):
    with open(yaml_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('path_save:'):
            # Replace with simple file path
            new_lines.append('  path_save: ./checkpoints/latest/model.pt\n')
            print("✅ Updated YAML: path_save: ./checkpoints/latest/model.pt")
        else:
            new_lines.append(line)
    
    with open(yaml_path, 'w') as f:
        f.writelines(new_lines)
else:
    print(f"❌ YAML not found: {yaml_path}")

print("\n" + "="*60)
print("✅ FIX COMPLETE!")
print("\nCheckpoints will save to: ./checkpoints/latest/model.pt")
print("\nTo backup checkpoints periodically, run:")
print("  python copy_checkpoints.py")
print("="*60)

# 3. Create backup script
with open("copy_checkpoints.py", 'w') as f:
    f.write('''import os
import shutil
import datetime

# Simple script to backup checkpoints
source = "./checkpoints/latest/model.pt"
dest_dir = "./checkpoints/by_epoch"

if os.path.exists(source):
    # Create timestamp
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    # Create destination filename
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"model_{timestamp}.pt")
    
    # Copy file
    shutil.copy2(source, dest)
    print(f"✅ Backed up: {dest}")
else:
    print("⚠️  No checkpoint found at: {source}")
''')

print("\n✅ Created backup script: copy_checkpoints.py")