# find_actual_models.py
import os
import torch

print("Searching for actual model files...")
print("="*60)

# Look for model files in common locations
search_paths = [
    "./models/",
    "./checkpoints/",
    "./",
    "./outputs/",
    "./results/"
]

model_files = []
for path in search_paths:
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.pt') or file.endswith('.pth') or file.endswith('.ckpt'):
                    full_path = os.path.join(root, file)
                    # Check file size (models are usually > 1MB)
                    size = os.path.getsize(full_path)
                    if size > 1000000:  # > 1MB
                        model_files.append((full_path, size))

print(f"Found {len(model_files)} potential model files (>1MB)")

# Check each model file
print("\nChecking model files...")
for i, (model_path, size) in enumerate(model_files[:10]):  # Check first 10
    print(f"\n{i+1}. {os.path.relpath(model_path)}")
    print(f"   Size: {size/1024/1024:.2f} MB")
    
    try:
        data = torch.load(model_path, map_location='cpu')
        print(f"   Keys: {list(data.keys())}")
        
        # Check for common keys
        if 'model_state_dict' in data:
            print(f"   ✓ Has model_state_dict")
        if 'epoch' in data:
            print(f"   Epoch: {data['epoch']}")
        if 'valid_psnr' in data:
            print(f"   PSNR: {data['valid_psnr']:.4f}")
        elif 'metrics_eval' in data and isinstance(data['metrics_eval'], dict):
            if 'valid_psnr' in data['metrics_eval']:
                print(f"   PSNR: {data['metrics_eval']['valid_psnr']:.4f}")
        
    except Exception as e:
        print(f"   ❌ Error loading: {e}")