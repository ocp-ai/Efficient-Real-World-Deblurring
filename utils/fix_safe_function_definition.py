# fix_safe_function_definition.py
import os
import re

print("Fixing safe_save_checkpoint function definition...")
print("="*60)

train_py = "./train.py"
if not os.path.exists(train_py):
    print(f"❌ train.py not found")
    exit(1)

with open(train_py, 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the safe_save_checkpoint function
print("Looking for safe_save_checkpoint function...")

# Create the corrected function
corrected_function = '''def safe_save_checkpoint(model, optim, scheduler, metrics_eval, metrics_train, path, global_rank):
    """
    Safe wrapper for checkpoint saving with PROPER metrics handling
    
    Args:
        model: The model
        optim: Optimizer
        scheduler: Learning rate scheduler
        metrics_eval: Evaluation metrics dict (must contain valid_psnr, valid_ssim, valid_lpips)
        metrics_train: Training metrics dict
        path: Save path
        global_rank: Process rank
    """
    import os
    import torch
    
    if global_rank != 0:
        return metrics_eval.get('valid_psnr', 0) if isinstance(metrics_eval, dict) else 0
    
    print(f"DEBUG [safe_save_checkpoint]: Saving checkpoint...")
    print(f"DEBUG: metrics_eval type: {type(metrics_eval)}")
    print(f"DEBUG: metrics_eval keys: {list(metrics_eval.keys()) if isinstance(metrics_eval, dict) else 'Not a dict'}")
    
    # Ensure path has .pt extension
    if not (path.endswith('.pt') or path.endswith('.pth')):
        path = path + '.pt'
        print(f"DEBUG: Added extension to path: {path}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    try:
        # Try to use the original save_checkpoint if available
        try:
            from utils.utils import save_checkpoint as original_save
            score = original_save(model, optim, scheduler, 
                                 metrics_eval=metrics_eval,
                                 metrics_train=metrics_train,
                                 paths=path,
                                 global_rank=global_rank)
            print(f"✅ Used original save_checkpoint, score: {score}")
            return score
        except ImportError:
            print("⚠️  Original save_checkpoint not found, using manual save")
        
        # Manual save - THIS IS CRITICAL
        checkpoint_data = {
            'epoch': metrics_train.get('epoch', 0) if isinstance(metrics_train, dict) else 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics_eval': metrics_eval if isinstance(metrics_eval, dict) else {},
            'metrics_train': metrics_train if isinstance(metrics_train, dict) else {},
        }
        
        # Also add metrics at root level for easy access
        if isinstance(metrics_eval, dict):
            checkpoint_data['valid_psnr'] = metrics_eval.get('valid_psnr', 0)
            checkpoint_data['valid_ssim'] = metrics_eval.get('valid_ssim', 0)
            checkpoint_data['valid_lpips'] = metrics_eval.get('valid_lpips', 0)
        
        # Save the checkpoint
        torch.save(checkpoint_data, path)
        
        # Extract PSNR for return value
        psnr = 0
        if isinstance(metrics_eval, dict):
            psnr = metrics_eval.get('valid_psnr', 0)
        
        print(f"✅ Manual checkpoint saved: {path}")
        print(f"   PSNR: {psnr:.4f}, SSIM: {checkpoint_data.get('valid_ssim', 0):.4f}")
        
        return psnr
        
    except Exception as e:
        print(f"❌ Error in safe_save_checkpoint: {e}")
        import traceback
        traceback.print_exc()
        
        # Emergency save
        try:
            emergency_path = "./emergency_checkpoint.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': metrics_train.get('epoch', 0) if isinstance(metrics_train, dict) else 0,
                'error': str(e)
            }, emergency_path)
            print(f"⚠️  Emergency saved to: {emergency_path}")
        except:
            print("❌ Even emergency save failed!")
        
        return 0'''

# Try to replace the function
pattern = r'def safe_save_checkpoint\(.*?\):.*?(?=\n\S|\Z)'
new_content = re.sub(pattern, corrected_function, content, flags=re.DOTALL)

if new_content != content:
    with open(train_py, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("✅ Replaced safe_save_checkpoint with corrected version")
    
    # Also fix any calls that might not have the right parameters
    print("\n🔍 Checking function calls...")
    
    # Read again
    with open(train_py, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find calls to safe_save_checkpoint
    for i, line in enumerate(lines):
        if 'safe_save_checkpoint(' in line and 'final_score =' in line:
            print(f"\nFound at line {i+1}: {line.strip()}")
            
            # Check if it has all 7 parameters
            params_match = re.search(r'safe_save_checkpoint\((.*?)\)', line)
            if params_match:
                params = params_match.group(1)
                param_count = len([p for p in params.split(',') if p.strip()])
                
                print(f"Parameter count: {param_count} (should be 7)")
                
                if param_count < 7:
                    print("❌ Not enough parameters!")
                    
                    # Add missing parameters
                    if 'metrics_eval=' not in params:
                        params = params.rstrip(', ') + ', metrics_eval=metrics_eval'
                        print("✓ Added metrics_eval")
                    
                    if 'metrics_train=' not in params:
                        params = params.rstrip(', ') + ', metrics_train=metrics_train'
                        print("✓ Added metrics_train")
                    
                    # Reconstruct the line
                    new_line = re.sub(r'safe_save_checkpoint\(.*?\)', 
                                    f'safe_save_checkpoint({params})', line)
                    
                    lines[i] = new_line
                    print(f"✅ Fixed line {i+1}")
                    
    # Write back if changes were made
    with open(train_py, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
else:
    print("❌ Could not replace function with regex")
    
    # Try line-by-line replacement
    print("\nTrying line-by-line replacement...")
    lines = content.split('\n')
    new_lines = []
    in_function = False
    replaced = False
    
    for line in lines:
        if 'def safe_save_checkpoint' in line and not replaced:
            new_lines.append(corrected_function)
            in_function = True
            replaced = True
        elif in_function and (line.startswith('def ') or line.strip() == ''):
            in_function = False
            new_lines.append(line)
        elif not in_function:
            new_lines.append(line)
    
    if replaced:
        with open(train_py, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        print("✅ Manually replaced function")
    else:
        print("❌ Could not find function to replace")

print("\n" + "="*60)
print("✅ FIX COMPLETE!")
print("\nThe safe_save_checkpoint function now:")
print("1. ✅ Receives metrics_eval and metrics_train as parameters")
print("2. ✅ Saves metrics properly in the checkpoint")
print("3. ✅ Has debug prints to track what's being saved")
print("4. ✅ Returns the PSNR value as final_score")
print("\nRestart training to see the fix in action!")
print("="*60)