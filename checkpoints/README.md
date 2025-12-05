# CHECKPOINTS ORGANIZATION

## Directory Structure
- `by_epoch/`          - Checkpoints saved at regular intervals
  - `every_10/`        - Every 10 epochs
  - `every_50/`        - Every 50 epochs  
  - `milestones/`      - Important milestones (epoch 1, 50, 100, etc.)
- `best_models/`       - Best performing models
  - `by_psnr/`         - Best PSNR models
  - `by_ssim/`         - Best SSIM models
  - `by_lpips/`        - Best LPIPS models
- `latest/`            - Always the latest model
- `configs/`           - Training configurations
- `logs/`              - Training logs and metrics
- `visualizations/`    - Sample outputs

## Naming Convention
- Epoch checkpoints: `epoch_XXX_psnr_YY.YY_ssim_0.ZZZ_lpips_0.AAA.pt`
- Best models: `best_psnr_YY.YY_epoch_XXX.pt`
- Latest: `latest_checkpoint.pt`

## Usage
- Resume training: Use any checkpoint from `by_epoch/` or `latest/`
- Final deployment: Use best model from `best_models/by_psnr/`
- Analysis: Check `logs/` for training history
