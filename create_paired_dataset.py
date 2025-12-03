"""
CREATE PAIRED DATASET FOR MICROSCOPY DEBLURRING
Purpose: Generate synthetic degraded images from clear microscopy images
         to create paired training data for supervised models (NAFNet, PromptIR,AdaIR, etc.)
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path
import argparse

# ============================================================================
# CONFIGURATION - ADJUST THESE TO MATCH YOUR REAL DEGRADATIONS
# ============================================================================

class DegradationConfig:
    """All degradation parameters - tune these to match your trainA images"""
    
    # 1. BLUR PARAMETERS (most important for defocus)
    BLUR_ENABLED = True
    BLUR_PROBABILITY = 0.8  # 80% of images get blur
    GAUSSIAN_KERNEL_RANGE = (3, 9)  # Kernel sizes: 3, 5, 7, 9
    GAUSSIAN_SIGMA_RANGE = (1.0, 3.0)  # Sigma values
    
    # 2. MOTION BLUR (if samples move during capture)
    MOTION_BLUR_ENABLED = False  # Usually not in microscopy
    MOTION_BLUR_PROBABILITY = 0.1
    MOTION_BLUR_LENGTH_RANGE = (5, 15)  # Length of motion blur
    
    # 3. LOW-LIGHT & NOISE PARAMETERS
    LOW_LIGHT_ENABLED = True
    LOW_LIGHT_PROBABILITY = 0.6  # 60% get low-light effect
    DARKEN_RANGE = (0.4, 0.8)  # 40% to 80% brightness
    
    # Noise types (choose what matches your microscope)
    GAUSSIAN_NOISE_PROBABILITY = 0.5
    GAUSSIAN_NOISE_STD_RANGE = (5, 25)  # Standard deviation
    
    POISSON_NOISE_PROBABILITY = 0.3  # Photon-counting noise (low light)
    
    # 4. CONTRAST & COLOR PARAMETERS
    CONTRAST_REDUCTION_PROBABILITY = 0.4
    CONTRAST_FACTOR_RANGE = (0.6, 0.9)  # Reduce contrast to 60-90%
    
    # 5. RESOLUTION/COMPRESSION (optional)
    DOWNSCALE_UPSCALE_PROBABILITY = 0.2  # Simulate resolution loss
    DOWNSCALE_FACTOR_RANGE = (0.5, 0.8)  # Downscale to 50-80% then upsample
    
    # 6. AUGMENTATION (to increase dataset diversity)
    RANDOM_CROP_ENABLED = True
    CROPS_PER_IMAGE = 4  # Number of random crops from each original
    CROP_SIZE = 256  # Output size

# ============================================================================
# DEGRADATION FUNCTIONS
# ============================================================================

def apply_gaussian_blur(image, config):
    """Apply Gaussian blur with random parameters"""
    if not config.BLUR_ENABLED or random.random() > config.BLUR_PROBABILITY:
        return image
    
    # Random odd kernel size
    ksize = random.choice(range(config.GAUSSIAN_KERNEL_RANGE[0], 
                                config.GAUSSIAN_KERNEL_RANGE[1] + 1, 2))
    sigma = random.uniform(config.GAUSSIAN_SIGMA_RANGE[0], 
                          config.GAUSSIAN_SIGMA_RANGE[1])
    
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)

def apply_motion_blur(image, config):
    """Apply linear motion blur (for sample movement)"""
    if not config.MOTION_BLUR_ENABLED or random.random() > config.MOTION_BLUR_PROBABILITY:
        return image
    
    length = random.randint(config.MOTION_BLUR_LENGTH_RANGE[0],
                           config.MOTION_BLUR_LENGTH_RANGE[1])
    angle = random.uniform(0, 180)
    
    # Create motion blur kernel
    kernel = np.zeros((length, length))
    kernel[(length - 1) // 2, :] = np.ones(length)
    kernel = cv2.warpAffine(kernel, 
                           cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0),
                           (length, length))
    kernel = kernel / np.sum(kernel)
    
    return cv2.filter2D(image, -1, kernel)

def apply_low_light(image, config):
    """Simulate low-light conditions with darkening and noise"""
    if not config.LOW_LIGHT_ENABLED or random.random() > config.LOW_LIGHT_PROBABILITY:
        return image
    
    result = image.copy().astype(np.float32)
    
    # 1. Darken the image
    darken_factor = random.uniform(config.DARKEN_RANGE[0], config.DARKEN_RANGE[1])
    result = result * darken_factor
    
    # 2. Add Gaussian noise
    if random.random() < config.GAUSSIAN_NOISE_PROBABILITY:
        noise_std = random.uniform(config.GAUSSIAN_NOISE_STD_RANGE[0],
                                  config.GAUSSIAN_NOISE_STD_RANGE[1])
        gaussian_noise = np.random.normal(0, noise_std, result.shape)
        result = result + gaussian_noise
    
    # 3. Add Poisson noise (characteristic of low-light microscopy)
    if random.random() < config.POISSON_NOISE_PROBABILITY:
        # Poisson noise is signal-dependent
        result = np.random.poisson(np.clip(result, 0, 255) / 10.0) * 10.0
    
    return np.clip(result, 0, 255).astype(np.uint8)

def reduce_contrast(image, config):
    """Reduce image contrast"""
    if random.random() > config.CONTRAST_REDUCTION_PROBABILITY:
        return image
    
    factor = random.uniform(config.CONTRAST_FACTOR_RANGE[0],
                           config.CONTRAST_FACTOR_RANGE[1])
    
    # Convert to float for calculation
    img_float = image.astype(np.float32)
    
    # Reduce contrast: bring pixels closer to mean
    mean_val = np.mean(img_float, axis=(0, 1), keepdims=True)
    result = mean_val + factor * (img_float - mean_val)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_resolution_degradation(image, config):
    """Simulate resolution loss by downscaling then upscaling"""
    if random.random() > config.DOWNSCALE_UPSCALE_PROBABILITY:
        return image
    
    factor = random.uniform(config.DOWNSCALE_FACTOR_RANGE[0],
                           config.DOWNSCALE_FACTOR_RANGE[1])
    
    h, w = image.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    
    # Downscale
    downscaled = cv2.resize(image, (new_w, new_h), 
                           interpolation=cv2.INTER_AREA)
    # Upscale back
    upscaled = cv2.resize(downscaled, (w, h), 
                         interpolation=cv2.INTER_LINEAR)
    
    return upscaled

def apply_degradation_pipeline(clean_image, config):
    """
    Apply all degradations in physically plausible order
    Returns: degraded version of the input
    """
    degraded = clean_image.copy()
    
    # Order matters! Apply in order of how they occur in real microscopy:
    
    # 1. First: Optical degradations (blur, resolution loss)
    degraded = apply_gaussian_blur(degraded, config)
    degraded = apply_motion_blur(degraded, config)
    degraded = apply_resolution_degradation(degraded, config)
    
    # 2. Then: Sensor/capture degradations (low light, noise)
    degraded = apply_low_light(degraded, config)
    degraded = reduce_contrast(degraded, config)
    
    return degraded

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def create_paired_dataset(source_dir, output_base, config):
    """
    Create paired dataset from clear microscopy images
    """
    # Create output directories
    input_dir = Path(output_base) / "train" / "input"
    target_dir = Path(output_base) / "train" / "target"
    
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all clear images
    source_path = Path(source_dir)
    image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    image_files = [f for f in source_path.iterdir() if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} clear images in {source_dir}")
    print(f"Generating {config.CROPS_PER_IMAGE} crops per image...")
    print(f"Total expected pairs: {len(image_files) * config.CROPS_PER_IMAGE}")
    print("-" * 50)
    
    pair_counter = 0
    
    for img_idx, img_path in enumerate(image_files):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not load {img_path}")
            continue
        
        # Convert to RGB if needed (most models expect RGB)
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        h, w = img.shape[:2]
        
        # Skip if image is too small
        if h < config.CROP_SIZE or w < config.CROP_SIZE:
            print(f"Skipping {img_path.name}: too small ({w}x{h})")
            continue
        
        # Create multiple random crops
        for crop_idx in range(config.CROPS_PER_IMAGE):
            if config.RANDOM_CROP_ENABLED:
                # Random crop
                y = random.randint(0, h - config.CROP_SIZE)
                x = random.randint(0, w - config.CROP_SIZE)
                clean_patch = img[y:y+config.CROP_SIZE, x:x+config.CROP_SIZE]
            else:
                # Center crop
                y = (h - config.CROP_SIZE) // 2
                x = (w - config.CROP_SIZE) // 2
                clean_patch = img[y:y+config.CROP_SIZE, x:x+config.CROP_SIZE]
            
            # Apply degradation pipeline to create input
            degraded_patch = apply_degradation_pipeline(clean_patch, config)
            
            # Save both patches
            save_name = f"pair_{pair_counter:06d}.png"
            
            # Save degraded as input
            cv2.imwrite(str(input_dir / save_name), degraded_patch)
            # Save clean as target/ground truth
            cv2.imwrite(str(target_dir / save_name), clean_patch)
            
            pair_counter += 1
            
            # Progress indicator
            if pair_counter % 100 == 0:
                print(f"Created {pair_counter} pairs...")
    
    print("-" * 50)
    print(f"Dataset creation complete!")
    print(f"Total pairs generated: {pair_counter}")
    print(f"Input images saved to: {input_dir}")
    print(f"Target images saved to: {target_dir}")
    
    # Create a summary file with configuration
    summary_path = Path(output_base) / "dataset_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("MICROSCOPY DEBLURRING PAIRED DATASET SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Source directory: {source_dir}\n")
        f.write(f"Total pairs: {pair_counter}\n")
        f.write(f"Image size: {config.CROP_SIZE}x{config.CROP_SIZE}\n")
        f.write(f"Created on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DEGRADATION CONFIGURATION:\n")
        f.write(f"  Blur enabled: {config.BLUR_ENABLED} (prob: {config.BLUR_PROBABILITY})\n")
        f.write(f"  Gaussian kernel range: {config.GAUSSIAN_KERNEL_RANGE}\n")
        f.write(f"  Gaussian sigma range: {config.GAUSSIAN_SIGMA_RANGE}\n")
        f.write(f"  Low light enabled: {config.LOW_LIGHT_ENABLED}\n")
        f.write(f"  Darken range: {config.DARKEN_RANGE}\n")
        f.write(f"  Gaussian noise probability: {config.GAUSSIAN_NOISE_PROBABILITY}\n")
        f.write(f"  Contrast reduction probability: {config.CONTRAST_REDUCTION_PROBABILITY}\n")
    
    print(f"Configuration summary saved to: {summary_path}")
    
    return pair_counter

# ============================================================================
# PARAMETER TUNING UTILITIES
# ============================================================================

def tune_parameters_by_comparison(real_blurry_dir, config):
    """
    Helper function to tune degradation parameters by comparing
    with real blurry images from trainA
    """
    print("\n" + "="*60)
    print("PARAMETER TUNING MODE")
    print("="*60)
    
    # Load a few real blurry images for comparison
    real_blurry_path = Path(real_blurry_dir)
    real_images = list(real_blurry_path.glob("*.*"))[:5]
    
    # Load a few clear images
    clear_image_path = Path(r"D:\2025_PROJECT\CycleGAN_Testing\dpdx_cysts\trainB_original")
    clear_images = list(clear_image_path.glob("*.*"))[:5]
    
    print("Compare these synthetic degradations with your real trainA images:")
    print(f"Real blurry images from: {real_blurry_dir}")
    print()
    
    for i, clear_img_path in enumerate(clear_images):
        if i >= 3:  # Just show 3 examples
            break
            
        clear_img = cv2.imread(str(clear_img_path))
        if clear_img is None:
            continue
            
        # Create degraded versions with different settings
        print(f"\nExample {i+1}: {clear_img_path.name}")
        print("  Original clear image -> Various degraded versions")
        
        # Test different blur levels
        test_config = DegradationConfig()
        
        # Mild degradation
        test_config.GAUSSIAN_SIGMA_RANGE = (0.5, 1.5)
        mild = apply_degradation_pipeline(clear_img, test_config)
        
        # Medium degradation
        test_config.GAUSSIAN_SIGMA_RANGE = (1.5, 3.0)
        medium = apply_degradation_pipeline(clear_img, test_config)
        
        # Strong degradation
        test_config.GAUSSIAN_SIGMA_RANGE = (3.0, 5.0)
        strong = apply_degradation_pipeline(clear_img, test_config)
        
        # Save for visual comparison
        output_dir = Path("parameter_tuning_samples")
        output_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(output_dir / f"sample{i}_original.png"), clear_img)
        cv2.imwrite(str(output_dir / f"sample{i}_mild.png"), mild)
        cv2.imwrite(str(output_dir / f"sample{i}_medium.png"), medium)
        cv2.imwrite(str(output_dir / f"sample{i}_strong.png"), strong)
        
        print(f"  Saved to: {output_dir}/")
        print("  Compare with your real blurry images in trainA/")
        print("  Adjust GAUSSIAN_SIGMA_RANGE in config to match")
    
    print("\n" + "="*60)
    print("ADJUSTMENT GUIDELINES:")
    print("1. Look at the real blurriness in your trainA images")
    print("2. Compare with mild/medium/strong synthetic versions")
    print("3. Adjust GAUSSIAN_SIGMA_RANGE in DegradationConfig")
    print("4. For noise: adjust GAUSSIAN_NOISE_STD_RANGE")
    print("5. For brightness: adjust DARKEN_RANGE")
    print("="*60)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import time
    
    parser = argparse.ArgumentParser(description='Create paired dataset for microscopy deblurring')
    parser.add_argument('--source', type=str, 
                       default=r"D:\2025_PROJECT\CycleGAN_Testing\dpdx_cysts\trainB_original",
                       help='Path to clear source images')
    parser.add_argument('--output', type=str, 
                       default=r"D:\2025_PROJECT\Dataset_WinnerStyle",  # yaml config path
                       help='Output directory for paired dataset')
    parser.add_argument('--tune', action='store_true',
                       help='Enter parameter tuning mode')
    parser.add_argument('--compare_with', type=str,
                       default=r"D:\2025_PROJECT\CycleGAN_Testing\dpdx_cysts\trainA", # compare with target rough images
                       help='Path to real blurry images for comparison')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = DegradationConfig()
    
    if args.tune:
        # Parameter tuning mode
        tune_parameters_by_comparison(args.compare_with, config)
    else:
        # Full dataset creation mode
        start_time = time.time()
        
        print("="*60)
        print("MICROSCOPY DEBLURRING - PAIRED DATASET CREATION")
        print("="*60)
        print(f"Source: {args.source}")
        print(f"Output: {args.output}")
        print(f"Crop size: {config.CROP_SIZE}x{config.CROP_SIZE}")
        print(f"Crops per image: {config.CROPS_PER_IMAGE}")
        print("-"*60)
        
        total_pairs = create_paired_dataset(args.source, args.output, config)
        
        elapsed = time.time() - start_time
        print(f"\nTotal time: {elapsed:.2f} seconds")
        print(f"Average: {elapsed/max(1, total_pairs):.3f} seconds per pair")
        
        print("\nNEXT STEP: Train NAFNet/AdaIR with:")
        print(f"  --train_dir {args.output}/train")



        # Terminal Command Lines: 

        # python create_paired_dataset.py --tune       # parameters fine tune
        # python create_paired_dataset.py              # create simulated dataset

        """
        print out after Terminal Command

        
        ============================================================
        PARAMETER TUNING MODE
        ============================================================
        Compare these synthetic degradations with your real trainA images:
        Real blurry images from: D:\2025_PROJECT\CycleGAN_Testing\dpdx_cysts\trainA


        Example 1: Answer to Case 226_unknown_2616_0283.jpg
        Original clear image -> Various degraded versions
        Saved to: parameter_tuning_samples/
        Compare with your real blurry images in trainA/
        Adjust GAUSSIAN_SIGMA_RANGE in config to match

        Example 2: Answer to Case 298_unknown_2304_0103.jpg
        Original clear image -> Various degraded versions
        Saved to: parameter_tuning_samples/
        Compare with your real blurry images in trainA/
        Adjust GAUSSIAN_SIGMA_RANGE in config to match

        Example 3: Answer to Case 298_unknown_2306_0185.jpg
        Original clear image -> Various degraded versions
        Saved to: parameter_tuning_samples/
        Compare with your real blurry images in trainA/
        Adjust GAUSSIAN_SIGMA_RANGE in config to match

        ============================================================
        ADJUSTMENT GUIDELINES:
        1. Look at the real blurriness in your trainA images
        2. Compare with mild/medium/strong synthetic versions
        3. Adjust GAUSSIAN_SIGMA_RANGE in DegradationConfig
        4. For noise: adjust GAUSSIAN_NOISE_STD_RANGE
        5. For brightness: adjust DARKEN_RANGE
        ============================================================


        ============================================================
        MICROSCOPY DEBLURRING - PAIRED DATASET CREATION
        ============================================================
        Source: D:\2025_PROJECT\CycleGAN_Testing\dpdx_cysts\trainB_original
        Output: D:\2025_PROJEECT\Dataset_WinnerStyle
        Crop size: 256x256
        Crops per image: 4
        ------------------------------------------------------------
        Found 411 clear images in D:\2025_PROJECT\CycleGAN_Testing\dpdx_cysts\trainB_original
        Generating 4 crops per image...
        Total expected pairs: 1644
        --------------------------------------------------
        Created 100 pairs...
        Created 200 pairs...
        Created 300 pairs...
        Created 400 pairs...
        Created 500 pairs...
        Created 600 pairs...
        Created 700 pairs...
        Created 800 pairs...
        Created 900 pairs...
        Created 1000 pairs...
        Created 1100 pairs...
        Created 1200 pairs...
        Created 1300 pairs...
        Created 1400 pairs...
        Created 1500 pairs...
        Created 1600 pairs...
        --------------------------------------------------
        Dataset creation complete!
        Total pairs generated: 1644
        Input images saved to: D:\2025_PROJEECT\Dataset_WinnerStyle\train\input
        Target images saved to: D:\2025_PROJEECT\Dataset_WinnerStyle\train\target
        Configuration summary saved to: D:\2025_PROJEECT\Dataset_WinnerStyle\dataset_summary.txt

        Total time: 28.89 seconds
        Average: 0.018 seconds per pair

        NEXT STEP: Train NAFNet/AdaIR with:
        --train_dir D:\2025_PROJEECT\Dataset_WinnerStyle/train
        
        """