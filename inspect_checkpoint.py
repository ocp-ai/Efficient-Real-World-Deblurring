# inspect_checkpoint.py
import torch
import os

print("Inspecting checkpoint structure...")
print("="*60)

# Check the latest checkpoint
latest_path = "./checkpoints/latest/latest/latest_checkpoint.pt"
if os.path.exists(latest_path):
    print(f"📁 Latest checkpoint: {latest_path}")
    
    data = torch.load(latest_path, map_location='cpu')
    
    print("\n🔍 All keys in checkpoint:")
    for key in data.keys():
        print(f"  - {key}: {type(data[key])}")
    
    print("\n📊 Detailed inspection:")
    
    # Check model_state_dict
    if 'model_state_dict' in data:
        print(f"  ✓ model_state_dict: {len(data['model_state_dict'])} items")
        # Show first few keys
        keys = list(data['model_state_dict'].keys())[:3]
        print(f"    Sample keys: {keys}")
    
    # Check optimizer
    if 'optimizer_state_dict' in data:
        print(f"  ✓ optimizer_state_dict: Present")
    
    # Check metrics_eval
    if 'metrics_eval' in data:
        print(f"  ✓ metrics_eval: {data['metrics_eval']}")
        if isinstance(data['metrics_eval'], dict):
            print(f"    Keys: {list(data['metrics_eval'].keys())}")
    else:
        print(f"  ❌ metrics_eval: NOT FOUND")
    
    # Check metrics_train
    if 'metrics_train' in data:
        print(f"  ✓ metrics_train: {data['metrics_train']}")
    else:
        print(f"  ❌ metrics_train: NOT FOUND")
    
    # Check for any metric data
    print("\n🔎 Searching for any metric data...")
    metric_keys = []
    for key in data.keys():
        if any(metric in str(key).lower() for metric in ['psnr', 'ssim', 'lpips', 'valid', 'metric']):
            metric_keys.append((key, type(data[key])))
    
    if metric_keys:
        print("Found potential metric keys:")
        for key, key_type in metric_keys:
            print(f"  - {key}: {key_type} = {data[key]}")
    else:
        print("No metric keys found")
    
    # Try to extract epoch
    epoch = data.get('epoch', 'Not found')
    print(f"\n📅 Epoch: {epoch}")
    
else:
    print(f"❌ Latest checkpoint not found: {latest_path}")