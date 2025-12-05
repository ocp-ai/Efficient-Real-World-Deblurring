# check_current_setup.py
import os
import yaml
import sys

print("="*60)
print("Checking Current Training Setup")
print("="*60)

# 1. Check YAML file
yaml_path = "./options/train/RSBlur.yml"
if os.path.exists(yaml_path):
    with open(yaml_path, 'r') as f:
        content = f.read()
        # Look for path_save
        if 'path_save:' in content:
            lines = content.split('\n')
            for line in lines:
                if 'path_save:' in line:
                    print(f"YAML path_save: {line.strip()}")
                    break
else:
    print(f"❌ YAML not found: {yaml_path}")

# 2. Check models directory
print("\n📁 Models directory contents:")
models_dir = "./models"
if os.path.exists(models_dir):
    for item in os.listdir(models_dir):
        full_path = os.path.join(models_dir, item)
        if os.path.isfile(full_path):
            size = os.path.getsize(full_path)
            print(f"  📄 {item} ({size:,} bytes)")
        else:
            print(f"  📁 {item}/")
else:
    print("  Models directory doesn't exist")

# 3. Check train.py for save_checkpoint call
print("\n🔍 Checking train.py line 102...")
train_py = "./train.py"
if os.path.exists(train_py):
    with open(train_py, 'r') as f:
        lines = f.readlines()
        if len(lines) >= 102:
            print(f"Line 102: {lines[101].strip()}")
            
            # Check a few lines before and after
            print("Context (lines 95-110):")
            for i in range(95, min(111, len(lines))):
                print(f"{i+1:3d}: {lines[i].rstrip()}")
else:
    print("❌ train.py not found")

print("\n" + "="*60)
print("RECOMMENDATION:")
print("1. Update YAML: path_save: ./models/My_Microscope_NAFNet_C16_L14.pt")
print("2. Add safe_save_checkpoint wrapper to train.py")
print("3. Stop and restart training")
print("="*60)