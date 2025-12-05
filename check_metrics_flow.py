# check_metrics_flow.py
import os

print("Tracing metrics flow...")
print("="*60)

# Look for where metrics_eval is created
train_py = "./train.py"
if os.path.exists(train_py):
    with open(train_py, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print("Looking for metrics_eval creation...")
    
    for i, line in enumerate(lines):
        if 'metrics_eval =' in line or 'metrics_eval:' in line:
            print(f"\nLine {i+1}: {line.strip()}")
            # Show context
            start = max(0, i-2)
            end = min(len(lines), i+3)
            for j in range(start, end):
                print(f"{j+1:3d}: {lines[j].rstrip()}")
    
    print("\n" + "="*60)
    print("Looking for eval_model call...")
    
    for i, line in enumerate(lines):
        if 'eval_model' in line:
            print(f"\nLine {i+1}: {line.strip()}")
            # Show context
            start = max(0, i-2)
            end = min(len(lines), i+3)
            for j in range(start, end):
                print(f"{j+1:3d}: {lines[j].rstrip()}")