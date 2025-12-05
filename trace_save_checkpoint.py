# trace_save_checkpoint.py
import os
import re

print("Tracing save_checkpoint usage...")
print("="*60)

train_py = "./train.py"
if os.path.exists(train_py):
    with open(train_py, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all calls to save_checkpoint
    pattern = r'save_checkpoint\(.*?\)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    print(f"Found {len(matches)} save_checkpoint calls:")
    for i, match in enumerate(matches, 1):
        print(f"\n{i}. {match[:100]}...")
    
    # Also find the function definition
    if 'def safe_save_checkpoint' in content:
        print("\n🔍 Found save_checkpoint function definition")
        # Extract function
        start = content.find('def save_checkpoint')
        # Find next function definition
        next_func = content.find('\ndef ', start + 1)
        if next_func == -1:
            func_code = content[start:]
        else:
            func_code = content[start:next_func]
        
        print("\nFunction signature:")
        lines = func_code.split('\n')
        for line in lines[:10]:  # Show first 10 lines
            print(f"  {line}")
    
    # Check what parameters are being passed
    print("\n🔍 Checking line 102 in train.py...")
    lines = content.split('\n')
    if len(lines) > 101:
        line_102 = lines[101]
        print(f"Line 102: {line_102}")
        
        # Show context
        print("\nContext (lines 95-110):")
        for i in range(95, min(111, len(lines))):
            print(f"{i+1:3d}: {lines[i]}")
            
else:
    print(f"❌ train.py not found")