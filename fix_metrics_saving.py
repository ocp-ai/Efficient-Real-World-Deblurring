# fix_safe_save_checkpoint_metrics.py
import os
import re

print("Fixing safe_save_checkpoint metrics saving...")
print("="*60)

train_py = "./train.py"
backup_py = "./train.py.backup_safe"

# Create backup
if os.path.exists(train_py):
    import shutil
    shutil.copy2(train_py, backup_py)
    print(f"✅ Backed up to: {backup_py}")

with open(train_py, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the safe_save_checkpoint function definition
print("\n🔍 Looking for safe_save_checkpoint function...")

if 'def safe_save_checkpoint' in content:
    print("✅ Found safe_save_checkpoint function")
    
    # Extract the function
    pattern = r'def safe_save_checkpoint\(.*?\):.*?(?=\n\S|\Z)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        old_function = match.group(0)
        print(f"Function length: {len(old_function)} characters")
        
        # Check if it already handles metrics properly
        if 'metrics_eval' in old_function and 'metrics_train' in old_function:
            print("✓ Function already receives metrics_eval and metrics_train")
        else:
            print("❌ Function doesn't receive metrics properly")
            
        # Check what it does with metrics
        if 'torch.save' in old_function:
            print("✓ Function saves with torch.save")
            # Check what it saves
            if 'metrics_eval' in old_function and 'metrics_train' in old_function:
                # Look for the torch.save call
                lines = old_function.split('\n')
                for i, line in enumerate(lines):
                    if 'torch.save' in line:
                        print(f"\nCurrent torch.save line: {line}")
                        # Show context
                        start = max(0, i-3)
                        end = min(len(lines), i+4)
                        for j in range(start, end):
                            print(f"  {j}: {lines[j]}")
                        break
        else:
            print("❌ Function doesn't use torch.save")
            
    else:
        print("❌ Could not extract function with regex")
        
else:
    print("❌ safe_save_checkpoint function not found")

# Now let's find where it's called
print("\n" + "="*60)
print("🔍 Looking for safe_save_checkpoint calls...")

lines = content.split('\n')
found_calls = []

for i, line in enumerate(lines):
    if 'safe_save_checkpoint(' in line:
        found_calls.append((i, line))
        print(f"\nLine {i+1}: {line.strip()}")
        
        # Show context
        start = max(0, i-2)
        end = min(len(lines), i+3)
        for j in range(start, end):
            print(f"{j+1:3d}: {lines[j].rstrip()}")

if found_calls:
    print(f"\n✅ Found {len(found_calls)} calls to safe_save_checkpoint")
    
    # Usually the first call is the one we want (around line 102)
    target_line, target_content = found_calls[0]
    
    print(f"\n🎯 Target line {target_line+1}:")
    print(f"   {target_content}")
    
    # Check what parameters are being passed
    if 'metrics_eval=' in target_content and 'metrics_train=' in target_content:
        print("✓ metrics_eval and metrics_train are being passed")
    else:
        print("❌ metrics_eval and metrics_train are NOT being passed!")
        
        # Fix this line
        print("\n🔧 Fixing the call...")
        
        # Extract the current parameters
        # Find the parameters between parentheses
        match = re.search(r'safe_save_checkpoint\((.*?)\)', target_content)
        if match:
            params = match.group(1)
            print(f"Current parameters: {params}")
            
            # Split by commas to see what's being passed
            param_list = [p.strip() for p in params.split(',')]
            print(f"Parameter list: {param_list}")
            
            # Check if metrics_eval and metrics_train are in there
            has_metrics_eval = any('metrics_eval' in p for p in param_list)
            has_metrics_train = any('metrics_train' in p for p in param_list)
            
            if not has_metrics_eval or not has_metrics_train:
                print("❌ Missing metrics parameters!")
                
                # We need to see what variables are available at this point
                print("\n🔍 Checking available variables before line", target_line+1)
                
                # Look for metrics_eval definition
                metrics_eval_line = -1
                for j in range(target_line-20, target_line):
                    if j >= 0 and 'metrics_eval' in lines[j] and '=' in lines[j]:
                        metrics_eval_line = j
                        print(f"Found metrics_eval at line {j+1}: {lines[j].strip()}")
                        break
                
                # Look for metrics_train definition  
                metrics_train_line = -1
                for j in range(target_line-20, target_line):
                    if j >= 0 and 'metrics_train' in lines[j] and '=' in lines[j]:
                        metrics_train_line = j
                        print(f"Found metrics_train at line {j+1}: {lines[j].strip()}")
                        break
                
                # Now fix the call
                if metrics_eval_line >= 0 and metrics_train_line >= 0:
                    print("\n🔧 Adding metrics parameters to the call...")
                    
                    # Create new line with metrics parameters
                    # Remove any existing path parameter
                    new_params = []
                    for param in param_list:
                        if 'paths=' in param or 'path=' in param:
                            # Keep this as is
                            new_params.append(param)
                        elif 'global_rank=' in param:
                            new_params.append(param)
                        elif 'model' in param or 'optim' in param or 'scheduler' in param:
                            new_params.append(param)
                    
                    # Add metrics parameters
                    new_params.append('metrics_eval=metrics_eval')
                    new_params.append('metrics_train=metrics_train')
                    
                    # Reconstruct the line
                    # Find the part before the parentheses
                    before_call = target_content.split('safe_save_checkpoint')[0]
                    new_line = f"{before_call}safe_save_checkpoint({', '.join(new_params)})"
                    
                    # Replace the line
                    lines[target_line] = new_line
                    
                    print(f"\n✅ Fixed line {target_line+1}:")
                    print(f"Old: {target_content}")
                    print(f"New: {new_line}")
                    
                    # Write back
                    with open(train_py, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(lines))
                    
                    print("\n✅ Updated train.py with proper metrics passing")
                    
                else:
                    print("❌ Could not find metrics_eval/metrics_train definitions")
                    
else:
    print("❌ No calls to safe_save_checkpoint found")

print("\n" + "="*60)
print("🎯 NEXT: Let's also fix the safe_save_checkpoint function itself")
print("="*60)