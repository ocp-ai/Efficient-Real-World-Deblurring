import torch
import os
from pathlib import Path

def debug_all_checkpoints_in_detail():
    """Show ALL checkpoint files in detail"""
    
    checkpoint_dir = r"C:\Users\setup_vxkejr\Efficient-Real-World-Deblurring\checkpoints\latest\by_epoch"
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Directory not found: {checkpoint_dir}")
        return
    
    # Get ALL .pt files
    pt_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt') or file.endswith('.pth'):
            pt_files.append(os.path.join(checkpoint_dir, file))
    
    print(f"📁 Found {len(pt_files)} checkpoint files in: {checkpoint_dir}")
    print("=" * 80)
    
    if not pt_files:
        print("❌ No checkpoint files found!")
        return
    
    # Sort by epoch number in filename
    def extract_epoch(filename):
        import re
        match = re.search(r'epoch[_\s]*(\d+)', os.path.basename(filename), re.IGNORECASE)
        return int(match.group(1)) if match else 0
    
    pt_files.sort(key=extract_epoch)
    
    # Analyze EVERY checkpoint
    for i, filepath in enumerate(pt_files):
        filename = os.path.basename(filepath)
        print(f"\n{'='*80}")
        print(f"📄 Checkpoint {i+1}/{len(pt_files)}: {filename}")
        print(f"📂 Full path: {filepath}")
        
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # 1. Quick summary
            print("\n📋 QUICK SUMMARY:")
            print(f"  Keys: {list(checkpoint.keys())}")
            
            # 2. Look for PSNR in common locations
            print("\n🎯 PSNR SEARCH:")
            
            psnr_found = False
            # Check root level
            for key in ['psnr', 'valid_psnr', 'final_score']:
                if key in checkpoint:
                    value = checkpoint[key]
                    print(f"  ✓ Found at root['{key}']: {value}")
                    psnr_found = True
            
            # Check metrics_eval
            if 'metrics_eval' in checkpoint:
                metrics = checkpoint['metrics_eval']
                if isinstance(metrics, dict):
                    for key in ['valid_psnr', 'psnr']:
                        if key in metrics:
                            value = metrics[key]
                            print(f"  ✓ Found at metrics_eval['{key}']: {value}")
                            psnr_found = True
                    
                    # Check if nested (multiple datasets)
                    if len(metrics) > 0:
                        first_val = next(iter(metrics.values()))
                        if isinstance(first_val, dict) and 'valid_psnr' in first_val:
                            value = first_val['valid_psnr']
                            dataset_name = next(iter(metrics.keys()))
                            print(f"  ✓ Found at metrics_eval['{dataset_name}']['valid_psnr']: {value}")
                            psnr_found = True
            
            if not psnr_found:
                print("  ❌ No PSNR found in common locations")
                
                # Deep search
                print("  🔍 Deep searching...")
                def deep_find_psnr(obj, path=""):
                    results = []
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            current_path = f"{path}.{k}" if path else k
                            if isinstance(v, (int, float)) and 'psnr' in str(k).lower():
                                results.append((current_path, v))
                            elif isinstance(v, dict):
                                results.extend(deep_find_psnr(v, current_path))
                            elif isinstance(v, (list, tuple)):
                                for i, item in enumerate(v):
                                    results.extend(deep_find_psnr(item, f"{current_path}[{i}]"))
                    return results
                
                deep_results = deep_find_psnr(checkpoint)
                if deep_results:
                    for path, value in deep_results:
                        print(f"  ✓ Found at {path}: {value}")
                else:
                    print("  ❌ No PSNR found anywhere in checkpoint")
            
            # 3. Show epoch and timestamp
            print("\n📅 METADATA:")
            for key in ['epoch', 'timestamp', 'step', 'iteration']:
                if key in checkpoint:
                    print(f"  {key}: {checkpoint[key]}")
            
            # 4. File info
            file_size = os.path.getsize(filepath)
            print(f"\n📏 FILE INFO:")
            print(f"  Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
            
        except Exception as e:
            print(f"❌ ERROR loading file: {str(e)}")
            continue
    
    # Summary table at the end
    print(f"\n{'='*80}")
    print("📊 SUMMARY TABLE (All Checkpoints)")
    print("=" * 80)
    print(f"{'#':>3} | {'Filename':<30} | {'Epoch':>6} | {'PSNR':>8} | {'Size (MB)':>9}")
    print("-" * 80)
    
    summary_data = []
    for i, filepath in enumerate(pt_files):
        filename = os.path.basename(filepath)
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # Get epoch
            epoch = checkpoint.get('epoch', '?')
            
            # Get PSNR
            psnr = 'N/A'
            if 'psnr' in checkpoint:
                psnr = f"{checkpoint['psnr']:.4f}"
            elif 'metrics_eval' in checkpoint:
                metrics = checkpoint['metrics_eval']
                if isinstance(metrics, dict) and 'valid_psnr' in metrics:
                    psnr = f"{metrics['valid_psnr']:.4f}"
                elif isinstance(metrics, dict) and len(metrics) > 0:
                    first_val = next(iter(metrics.values()))
                    if isinstance(first_val, dict) and 'valid_psnr' in first_val:
                        psnr = f"{first_val['valid_psnr']:.4f}"
            
            # Get file size
            file_size = os.path.getsize(filepath)
            size_mb = file_size / 1024 / 1024
            
            summary_data.append((i+1, filename, epoch, psnr, size_mb))
            
            print(f"{i+1:3d} | {filename:<30} | {str(epoch):>6} | {psnr:>8} | {size_mb:>8.2f}")
            
        except Exception as e:
            print(f"{i+1:3d} | {filename:<30} | {'ERROR':>6} | {'ERROR':>8} | {'ERROR':>9}")
            continue
    
    # Find best PSNR
    print("\n" + "=" * 80)
    print("🏆 BEST CHECKPOINT ANALYSIS")
    
    # Extract numeric PSNR values
    numeric_data = []
    for idx, filename, epoch, psnr, size_mb in summary_data:
        try:
            if psnr != 'N/A' and psnr != 'ERROR':
                psnr_float = float(psnr)
                numeric_data.append((idx, filename, epoch, psnr_float, size_mb))
        except:
            pass
    
    if numeric_data:
        # Sort by PSNR
        numeric_data.sort(key=lambda x: x[3], reverse=True)
        
        best_idx, best_filename, best_epoch, best_psnr, best_size = numeric_data[0]
        print(f"Best PSNR: {best_psnr:.4f}")
        print(f"  File: {best_filename}")
        print(f"  Epoch: {best_epoch}")
        print(f"  Index: {best_idx}")
        
        # Show top 3
        print("\nTop 3 checkpoints by PSNR:")
        for i, (idx, filename, epoch, psnr, size_mb) in enumerate(numeric_data[:3]):
            print(f"  {i+1}. PSNR: {psnr:.4f} | Epoch: {epoch} | File: {filename}")
    else:
        print("❌ No valid PSNR values found in any checkpoints!")

# Run it to see ALL checkpoints
debug_all_checkpoints_in_detail()