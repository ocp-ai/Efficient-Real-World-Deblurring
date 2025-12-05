# find_and_test_models.py
import torch
import os
import glob

print("Finding and testing all model files...")
print("="*60)

# Get all model files
model_files = []
for pattern in ["**/*.pt", "**/*.pth", "**/*.ckpt"]:
    for file in glob.glob(pattern, recursive=True):
        # Skip small files (likely not models)
        if os.path.getsize(file) > 100000:  # > 100KB
            model_files.append(file)

print(f"Found {len(model_files)} model files")
print("\n" + "="*60)

# Test each model
results = []
for model_file in model_files:
    try:
        print(f"\nTesting: {os.path.relpath(model_file)}")
        
        # Load checkpoint
        checkpoint = torch.load(model_file, map_location='cpu')
        
        # Extract information
        info = {
            'file': model_file,
            'size': os.path.getsize(model_file) / 1024 / 1024,  # MB
            'keys': list(checkpoint.keys())
        }
        
        # Look for PSNR
        psnr = None
        epoch = None
        
        if 'valid_psnr' in checkpoint:
            psnr = checkpoint['valid_psnr']
        elif 'metrics_eval' in checkpoint and isinstance(checkpoint['metrics_eval'], dict):
            if 'valid_psnr' in checkpoint['metrics_eval']:
                psnr = checkpoint['metrics_eval']['valid_psnr']
        elif 'metrics' in checkpoint and isinstance(checkpoint['metrics'], dict):
            if 'psnr' in checkpoint['metrics']:
                psnr = checkpoint['metrics']['psnr']
        
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
        elif 'metrics_train' in checkpoint and isinstance(checkpoint['metrics_train'], dict):
            if 'epoch' in checkpoint['metrics_train']:
                epoch = checkpoint['metrics_train']['epoch']
        
        info['psnr'] = psnr
        info['epoch'] = epoch
        
        print(f"  Size: {info['size']:.2f} MB")
        print(f"  Epoch: {epoch}")
        print(f"  PSNR: {psnr}")
        
        if psnr is not None and psnr > 0:
            results.append(info)
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

# Show results
if results:
    print("\n" + "="*60)
    print("MODELS WITH PSNR VALUES:")
    print("="*60)
    
    # Sort by PSNR (highest first)
    results.sort(key=lambda x: x['psnr'] if x['psnr'] is not None else 0, reverse=True)
    
    for i, result in enumerate(results):
        print(f"\n{i+1}. {os.path.relpath(result['file'])}")
        print(f"   PSNR: {result['psnr']:.4f}")
        print(f"   Epoch: {result['epoch']}")
        print(f"   Size: {result['size']:.2f} MB")
else:
    print("\n❌ No models found with valid PSNR values")

print("\n" + "="*60)
print("RECOMMENDATION:")
print("1. Run the directory check script to see what files you have")
print("2. Look for the largest .pt/.pth files (usually > 10MB)")
print("3. Check if training created any 'best_model.pt' or similar")
print("="*60)