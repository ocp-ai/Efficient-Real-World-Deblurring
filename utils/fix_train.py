# fix_train_py.py
import os

print("="*60)
print("FIXING train.py CHECKPOINT SAVING")
print("="*60)

train_py = "./train.py"
if not os.path.exists(train_py):
    print(f"❌ train.py not found at {train_py}")
    exit(1)

# Read the file
with open(train_py, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Check if safe_save_checkpoint already exists
if 'def safe_save_checkpoint' in content:
    print("✅ safe_save_checkpoint already exists in train.py")
else:
    # Add the safe wrapper function
    wrapper_code = '''
# ========== SAFE CHECKPOINT WRAPPER ==========
def safe_save_checkpoint(model, optim, scheduler, metrics_eval, metrics_train, path, global_rank):
    """Safe wrapper for checkpoint saving with error handling"""
    import os
    import torch
    
    # Ensure path has .pt extension
    if not (path.endswith('.pt') or path.endswith('.pth')):
        path = path + '.pt'
        if global_rank == 0:
            print(f"⚠️  Added .pt extension to checkpoint path: {{path}}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    try:
        # Try original save_checkpoint if available
        from utils.utils import save_checkpoint as original_save
        score = original_save(model, optim, scheduler, metrics_eval, 
                             metrics_train, path, global_rank)
        if global_rank == 0 and score != 0:
            print(f"✅ Checkpoint saved: {{path}} (score: {{score:.4f}})")
        return score
    except ImportError:
        # Fallback: manual save
        if global_rank == 0:
            checkpoint = {{
                'epoch': metrics_train.get('epoch', 0),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'metrics_eval': metrics_eval,
                'metrics_train': metrics_train,
                'final_score': metrics_eval.get('valid_psnr', 0)
            }}
            torch.save(checkpoint, path)
            print(f"⚠️  Manual checkpoint saved: {{path}}")
            return metrics_eval.get('valid_psnr', 0)
    except Exception as e:
        if global_rank == 0:
            print(f"❌ Error saving checkpoint: {{e}}")
            # Emergency save
            emergency_path = "./emergency_checkpoint.pt"
            torch.save({{
                'model': model.state_dict(),
                'epoch': metrics_train.get('epoch', 0),
                'error': str(e)
            }}, emergency_path)
            print(f"⚠️  Emergency saved to: {{emergency_path}}")
        return 0
# ============================================
'''

    # Insert after the last import or before run_model()
    import_lines = []
    lines = content.split('\n')
    insert_pos = -1
    
    for i, line in enumerate(lines):
        if 'import' in line and 'def ' not in line:
            import_lines.append(i)
    
    if import_lines:
        insert_pos = import_lines[-1] + 1
        lines.insert(insert_pos, wrapper_code)
        print(f"✅ Added safe_save_checkpoint function after imports (line {insert_pos+1})")
    else:
        # Insert near the top
        lines.insert(5, wrapper_code)
        print(f"✅ Added safe_save_checkpoint function near top")

# Now replace the save_checkpoint call
if 'save_checkpoint(' in content:
    # Simple replacement
    new_content = content.replace('save_checkpoint(', 'safe_save_checkpoint(')
    
    if new_content != content:
        with open(train_py, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ Replaced all save_checkpoint() calls with safe_save_checkpoint()")
    else:
        print("⚠️  Could not replace save_checkpoint calls")
else:
    print("❌ No save_checkpoint calls found in train.py")

print("\n" + "="*60)
print("STEP 2 COMPLETE!")
print("="*60)