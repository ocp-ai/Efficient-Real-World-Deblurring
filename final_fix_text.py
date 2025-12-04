import os
import random
import shutil

DATA_ROOT = r"D:\2025_PROJECT\Dataset_WinnerStyle"

def create_proper_rsblur_files():
    """
    Create CORRECT RSBlur format files based on official dataset_RSBlur.py
    """
    print("="*60)
    print("Fixing RSBlur dataset files based on official code")
    print("="*60)
    
    # 1. Backup old files
    print("\n1. Backing up old files...")
    backup_dir = os.path.join(DATA_ROOT, "backup_" + os.path.basename(DATA_ROOT))
    os.makedirs(backup_dir, exist_ok=True)
    
    old_files = ['RSBlur_blur_test.txt', 'RSBlur_blur_train.txt',
                 'RSBlur_real_test.txt', 'RSBlur_real_train.txt']
    
    for old_file in old_files:
        old_path = os.path.join(DATA_ROOT, old_file)
        if os.path.exists(old_path):
            backup_path = os.path.join(backup_dir, old_file)
            shutil.copy2(old_path, backup_path)
            print(f"   Backup: {old_file} -> {backup_path}")
    
    # 2. Get all image pairs from train/input and train/target
    print("\n2. Scanning image files...")
    input_dir = os.path.join(DATA_ROOT, "train", "input")    # Blurry
    target_dir = os.path.join(DATA_ROOT, "train", "target")  # Sharp
    
    # Get all PNG files
    input_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    target_files = [f for f in os.listdir(target_dir) if f.lower().endswith('.png')]
    
    input_files.sort()
    target_files.sort()
    
    print(f"   Blurry images (input/): {len(input_files)}")
    print(f"   Sharp images (target/): {len(target_files)}")
    
    # Verify they match
    if input_files != target_files:
        print("  WARNING:  File names don't match exactly!")
        # Find common files
        common = set(input_files) & set(target_files)
        common_files = sorted(list(common))
        print(f"   Common files: {len(common_files)}")
    else:
        common_files = input_files
    
    # 3. Shuffle and split 80/20
    print("\n3. Shuffling and splitting 80/20...")
    random.shuffle(common_files)
    split_idx = int(len(common_files) * 0.8)
    train_files = common_files[:split_idx]
    test_files = common_files[split_idx:]
    
    print(f"   Training pairs: {len(train_files)}")
    print(f"   Validation pairs: {len(test_files)}")
    
    # 4. Create CORRECT RSBlur format files (TWO COLUMNS!)
    print("\n4. Creating CORRECT RSBlur format files...")
    
    # IMPORTANT: Official code uses relative paths from train_path/test_path
    # So we need: "train/target/image.png train/input/image.png"
    
    # Training file
    train_path = os.path.join(DATA_ROOT, "RSBlur_real_train.txt")
    with open(train_path, 'w', encoding='utf-8') as f:
        for img_file in train_files:
            # TWO COLUMNS: sharp_path blur_path
            f.write(f"train/target/{img_file} train/input/{img_file}\n")
    
    # Testing/Validation file
    test_path = os.path.join(DATA_ROOT, "RSBlur_real_test.txt")
    with open(test_path, 'w', encoding='utf-8') as f:
        for img_file in test_files:
            f.write(f"train/target/{img_file} train/input/{img_file}\n")
    
    print(f"   Created: RSBlur_real_train.txt ({len(train_files)} pairs)")
    print(f"   Created: RSBlur_real_test.txt ({len(test_files)} pairs)")
    
    # 5. Verify format
    print("\n5. Verifying format...")
    with open(train_path, 'r') as f:
        lines = f.readlines()
        if lines:
            first_line = lines[0].strip()
            parts = first_line.split()
            print(f"   First line: {first_line}")
            print(f"   Columns: {len(parts)} (should be 2)")
            
            # Check if paths would exist
            sharp_test = os.path.join(DATA_ROOT, parts[0])
            blur_test = os.path.join(DATA_ROOT, parts[1])
            
            if os.path.exists(sharp_test) and os.path.exists(blur_test):
                print("   ✓ Files exist!")
            else:
                print(f"   ERROR: Files don't exist:")
                print(f"      {sharp_test}")
                print(f"      {blur_test}")
    
    return len(train_files), len(test_files)

def create_simple_test():
    """Create a simple test to verify the data loader works"""
    print("\n" + "="*60)
    print("Creating test script to verify dataset loading...")
    print("="*60)
    
    test_script = os.path.join(DATA_ROOT, "test_dataset.py")
    with open(test_script, 'w') as f:
        f.write('''import sys
import os

# Add the dataset_RSBlur.py directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_RSBlur import main_dataset_rsblur

# Test with your paths
try:
    train_loader, test_loaders, samplers = main_dataset_rsblur(
        train_path=r"D:/2025_PROJECT/Dataset_WinnerStyle",
        test_path=r"D:/2025_PROJECT/Dataset_WinnerStyle",
        verbose=True,
        num_workers=0,
        world_size=1,
        rank=0
    )
    
    print("\\nSUCCESS: Dataset loaded successfully!")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(test_loaders['RSBlur'])}")
    
    # Try to get one batch
    for batch_idx, (sharp, blur) in enumerate(train_loader):
        print(f"\\nFirst batch:")
        print(f"  Sharp shape: {sharp.shape}")
        print(f"  Blur shape: {blur.shape}")
        break
        
except Exception as e:
    print(f"\\nERROR: Error loading dataset: {e}")
    import traceback
    traceback.print_exc()
''')
    
    print(f"Test script created: {test_script}")
    print("Run it with: python test_dataset.py")

def main():
    print("FINAL FIX FOR RSBlur DATASET")
    print(f"Data root: {DATA_ROOT}")
    print("="*60)
    
    # Create proper files
    train_count, test_count = create_proper_rsblur_files()
    
    # Create test script
    create_simple_test()
    
    print("\n" + "="*60)
    print("FINAL FIX COMPLETE!")
    print("\nSummary:")
    print(f"  Training pairs: {train_count}")
    print(f"  Validation pairs: {test_count}")
    print(f"  Total: {train_count + test_count}")
    print("\nYour YAML should have:")
    print("  train_path: D:/2025_PROJECT/Dataset_WinnerStyle")
    print("  test_path: D:/2025_PROJECT/Dataset_WinnerStyle")
    print("\nThe data loader will:")
    print("  1. Read RSBlur_real_train.txt from test_path")
    print("  2. Read RSBlur_real_test.txt from test_path")
    print("  3. Join paths with train_path/test_path")
    print("="*60)

if __name__ == "__main__":
    main()