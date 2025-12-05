# verify_fix.py
import os
import yaml

print("="*60)
print("VERIFYING FIXES")
print("="*60)

# 1. Check YAML
yaml_path = "./options/train/RSBlur.yml"
print("1. Checking YAML...")
if os.path.exists(yaml_path):
    with open(yaml_path, 'r') as f:
        for line in f:
            if 'path_save:' in line:
                print(f"   ✅ Found: {line.strip()}")
                if '.pt' in line:
                    print("   ✅ Has .pt extension - GOOD!")
                else:
                    print("   ❌ Missing .pt extension - BAD!")
                break
else:
    print("   ❌ YAML not found")

# 2. Check directory is gone
print("\n2. Checking directory...")
dir_path = "./models/My_Microscope_NAFNet_C16_L14"
if os.path.exists(dir_path):
    print(f"   ❌ Directory still exists: {dir_path}")
    print("   Remove it with: os.rmdir('./models/My_Microscope_NAFNet_C16_L14')")
else:
    print("   ✅ Directory removed - GOOD!")

# 3. Check train.py
print("\n3. Checking train.py...")
train_py = "./train.py"
if os.path.exists(train_py):
    with open(train_py, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    if 'safe_save_checkpoint' in content:
        print("   ✅ safe_save_checkpoint function exists")
    else:
        print("   ❌ safe_save_checkpoint not found")
        
    if 'safe_save_checkpoint(' in content:
        print("   ✅ safe_save_checkpoint() is being called")
    elif 'save_checkpoint(' in content:
        print("   ❌ Still calling save_checkpoint() instead of safe_save_checkpoint()")
    else:
        print("   ⚠️  No checkpoint calls found")
else:
    print("   ❌ train.py not found")

# 4. Create test checkpoint
print("\n4. Testing checkpoint creation...")
test_path = "./models/My_Microscope_NAFNet_C16_L14.pt"
try:
    import torch
    test_checkpoint = {
        'test': True,
        'message': 'Test checkpoint created successfully',
        'epoch': 0
    }
    torch.save(test_checkpoint, test_path)
    print(f"   ✅ Test checkpoint created: {test_path}")
    
    # Verify it can be loaded
    loaded = torch.load(test_path, map_location='cpu')
    print(f"   ✅ Checkpoint loaded successfully")
    print(f"   ✅ Contains keys: {list(loaded.keys())}")
    
    # Clean up
    os.remove(test_path)
    print(f"   ✅ Test checkpoint cleaned up")
    
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*60)
print("FINAL INSTRUCTIONS:")
print("1. STOP current training (Ctrl+C in terminal)")
print("2. Run: python train.py -p options/train/RSBlur.yml")
print("3. Check that checkpoints are saved as .pt files")
print("="*60)