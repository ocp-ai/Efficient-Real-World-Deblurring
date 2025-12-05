import torch
import os
from pathlib import Path

def read_checkpoint_psnr(checkpoint_path):
    """Read a checkpoint file and extract PSNR"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Different ways PSNR might be stored
        psnr = None
        
        if 'psnr' in checkpoint:
            psnr = checkpoint['psnr']
        elif 'metrics' in checkpoint and 'psnr' in checkpoint['metrics']:
            psnr = checkpoint['metrics']['psnr']
        elif 'metrics_eval' in checkpoint and 'valid_psnr' in checkpoint['metrics_eval']:
            psnr = checkpoint['metrics_eval']['valid_psnr']
        elif 'final_score' in checkpoint:
            psnr = checkpoint['final_score']
        
        epoch = checkpoint.get('epoch', 'Unknown')
        # Ensure epoch is string for printing
        if isinstance(epoch, (int, float)):
            epoch = str(int(epoch))
        
        return psnr, epoch, checkpoint_path
    except Exception as e:
        print(f"❌ Error reading {checkpoint_path}: {e}")
        return None, None, checkpoint_path

def scan_checkpoint_directory(base_path="./checkpoints"):
    """Scan a directory for all checkpoint files"""
    checkpoints = []
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(('.pt', '.pth')):
                full_path = os.path.join(root, file)
                checkpoints.append(full_path)
    
    return checkpoints

def analyze_all_checkpoints():
    """Analyze all checkpoints in organized structure"""
    
    print("📊 Checkpoint Analysis Report")
    print("=" * 60)
    
    base_paths = ["./checkpoints", "./experiments", "./results"]  # Common locations
    
    for base_path in base_paths:
        if not os.path.exists(base_path):
            continue
            
        print(f"\n📁 Scanning: {base_path}")
        
        # Check each organized folder
        folders = ['by_epoch', 'latest', 'best_models']
        
        for folder in folders:
            folder_path = os.path.join(base_path, folder)
            if os.path.exists(folder_path):
                print(f"\n  📂 Folder: {folder}/")
                print("  " + "-" * 40)
                
                checkpoints = []
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        if file.endswith(('.pt', '.pth')):
                            checkpoints.append(os.path.join(root, file))
                
                # Sort by filename (usually contains epoch)
                checkpoints.sort()
                
                for cp_path in checkpoints:
                    psnr, epoch, _ = read_checkpoint_psnr(cp_path)
                    if psnr is not None:
                        rel_path = os.path.relpath(cp_path, base_path)
                        # FIXED LINE: Use appropriate formatting
                        if epoch.isdigit():
                            print(f"    Epoch {int(epoch):3d} | PSNR: {psnr:.4f} | {rel_path}")
                        else:
                            print(f"    Epoch {epoch:>3s} | PSNR: {psnr:.4f} | {rel_path}")

# Run analysis
analyze_all_checkpoints()