import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def visualize_training_progress():
    """Visualize training progress from checkpoints"""
    
    # 1. Load training log
    log_path = Path("./checkpoints/logs/training_log.csv")
    if log_path.exists():
        df = pd.read_csv(log_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot PSNR
        axes[0, 0].plot(df['epoch'], df['psnr'], 'b-', label='PSNR')
        axes[0, 0].scatter(df[df['is_best_psnr']==1]['epoch'], 
                          df[df['is_best_psnr']==1]['psnr'], 
                          color='red', s=50, label='Best PSNR')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('PSNR')
        axes[0, 0].set_title('PSNR Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot SSIM
        axes[0, 1].plot(df['epoch'], df['ssim'], 'g-', label='SSIM')
        axes[0, 1].scatter(df[df['is_best_ssim']==1]['epoch'], 
                          df[df['is_best_ssim']==1]['ssim'], 
                          color='red', s=50, label='Best SSIM')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].set_title('SSIM Progress')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot LPIPS
        axes[1, 0].plot(df['epoch'], df['lpips'], 'r-', label='LPIPS')
        axes[1, 0].scatter(df[df['is_best_lpips']==1]['epoch'], 
                          df[df['is_best_lpips']==1]['lpips'], 
                          color='green', s=50, label='Best LPIPS')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LPIPS')
        axes[1, 0].set_title('LPIPS Progress (lower is better)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot Loss
        axes[1, 1].plot(df['epoch'], df['loss'], 'm-', label='Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('./checkpoints/visualizations/training_progress.png', dpi=150)
        plt.show()
        
        print(f"Best PSNR: {{df['psnr'].max():.4f}} at epoch {{df.loc[df['psnr'].idxmax(), 'epoch']}}")
        print(f"Best SSIM: {{df['ssim'].max():.4f}} at epoch {{df.loc[df['ssim'].idxmax(), 'epoch']}}")
        print(f"Best LPIPS: {{df['lpips'].min():.4f}} at epoch {{df.loc[df['lpips'].idxmin(), 'epoch']}}")
    
    # 2. List all checkpoints
    print("\n Available checkpoints:")
    checkpoints_dir = Path("./checkpoints/by_epoch")
    for subdir in checkpoints_dir.iterdir():
        if subdir.is_dir():
            print(f"\n  {{subdir.name}}/")
            for cp in subdir.glob("*.pt"):
                print(f"    - {{cp.name}}")
    
    # 3. Show best models
    print("\n Best models:")
    best_dir = Path("./checkpoints/best_models")
    for metric_dir in best_dir.iterdir():
        if metric_dir.is_dir():
            metric_name = metric_dir.name.replace('by_', '')
            best_files = list(metric_dir.glob("*.pt"))
            if best_files:
                latest_best = max(best_files, key=lambda x: x.stat().st_mtime)
                print(f"  {{metric_name.upper()}}: {{latest_best.name}}")

if __name__ == "__main__":
    visualize_training_progress()
