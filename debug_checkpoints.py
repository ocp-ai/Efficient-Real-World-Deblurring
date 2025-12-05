# debug_checkpoints.py
import torch
import os
import glob

print("Debugging checkpoint files...")
print("="*60)

# Find all checkpoint files
checkpoint_files = []
for root, dirs, files in os.walk("./checkpoints"):
    for file in files:
        if file.endswith('.pt') or file.endswith('.pth'):
            checkpoint_files.append(os.path.join(root, file))

print(f"Found {len(checkpoint_files)} checkpoint files")
print("\n" + "="*60)

# Check each file
for i, cp_path in enumerate(checkpoint_files[:10]):  # Check first 10
    try:
        data = torch.load(cp_path, map_location='cpu')
        print(f"\n{i+1}. {os.path.relpath(cp_path)}")
        print(f"   Keys in checkpoint: {list(data.keys())}")
        
        # Look for metrics
        if 'metrics_eval' in data:
            print(f"   metrics_eval keys: {list(data['metrics_eval'].keys())}")
            if 'valid_psnr' in data['metrics_eval']:
                print(f"   PSNR: {data['metrics_eval']['valid_psnr']:.4f}")
            else:
                print(f"   No valid_psnr in metrics_eval")
        elif 'metrics' in data:
            print(f"   metrics: {data['metrics']}")
        elif 'valid_psnr' in data:
            print(f"   PSNR: {data['valid_psnr']:.4f}")
        else:
            print(f"   No metrics found")
            
        if 'epoch' in data:
            print(f"   Epoch: {data['epoch']}")
            
    except Exception as e:
        print(f"\n{i+1}. {os.path.relpath(cp_path)} - ERROR: {e}")