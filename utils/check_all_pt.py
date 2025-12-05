# test_all_checkpoints.py (run after training)
import torch
import glob

checkpoints = glob.glob("./checkpoints/latest/by_epoch/epoch_*.pt")
for cp in checkpoints:
    data = torch.load(cp, map_location='cpu')
    psnr = data['metrics_eval'].get('valid_psnr', 0)
    print(f"{cp}: PSNR = {psnr:.4f}")

# Pick the highest PSNR