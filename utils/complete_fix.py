# complete_fix.py
import os
import shutil

print("="*60)
print("COMPLETE FIX FOR CHECKPOINT SAVING")
print("="*60)

# 1. Remove the empty directory
dir_path = "./models/My_Microscope_NAFNet_C16_L14"
if os.path.exists(dir_path) and os.path.isdir(dir_path):
    try:
        os.rmdir(dir_path)  # Only works if empty
        print(f"✅ Removed empty directory: {dir_path}")
    except:
        print(f"⚠️  Directory not empty, trying backup...")
        backup_dir = dir_path + "_backup"
        shutil.move(dir_path, backup_dir)
        print(f"   Moved to backup: {backup_dir}")

# 2. Update YAML file
yaml_path = "./options/train/RSBlur.yml"
if os.path.exists(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace ALL variations
    replacements = [
        ("path_save: ./models/My_Microscope_NAFNet_C16_L14", 
         "path_save: ./models/My_Microscope_NAFNet_C16_L14.pt"),
        ("path_save: ./models/My_Microscope_NAFNet_C16_L14/", 
         "path_save: ./models/My_Microscope_NAFNet_C16_L14.pt"),
        ("path_save: ./models/My_Microscope_NAFNet-C16-L14", 
         "path_save: ./models/My_Microscope_NAFNet_C16_L14.pt"),
        ("path_save: ./models/My_Microscope_NAFNet-C16-L14/", 
         "path_save: ./models/My_Microscope_NAFNet_C16_L14.pt"),
        ("path_save: ./models", 
         "path_save: ./models/My_Microscope_NAFNet_C16_L14.pt"),
        ("path_save: ./models/", 
         "path_save: ./models/My_Microscope_NAFNet_C16_L14.pt")
    ]
    
    changes_made = False
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"✅ Fixed YAML: {old} → {new}")
            changes_made = True
    
    if changes_made:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ YAML updated successfully")
    else:
        print(f"⚠️  No path_save lines found to fix")
        # Let's add it if missing
        if 'path_save:' not in content:
            # Find the save section
            lines = content.split('\n')
            new_lines = []
            in_save_section = False
            for line in lines:
                new_lines.append(line)
                if 'save:' in line:
                    in_save_section = True
                elif in_save_section and 'path_resume:' in line:
                    new_lines.append('  path_save: ./models/My_Microscope_NAFNet_C16_L14.pt')
                    print(f"✅ Added missing path_save line")
            
            with open(yaml_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines))
else:
    print(f"❌ YAML not found: {yaml_path}")

# 3. Create models directory
models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)
print(f"✅ Ensured models directory exists")

# 4. Test if we can write
test_file = "./models/test_write.pt"
try:
    import torch
    torch.save({'test': True}, test_file)
    os.remove(test_file)
    print(f"✅ Can write to models directory")
except Exception as e:
    print(f"❌ Cannot write to models directory: {e}")

print("\n" + "="*60)
print("STEP 1 COMPLETE!")
print("="*60)