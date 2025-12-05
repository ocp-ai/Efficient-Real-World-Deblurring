# find_save_checkpoint.py
import os
import re

def find_function_in_files():
    """Find where save_checkpoint is defined"""
    print("Searching for save_checkpoint function...")
    
    # Search all Python files
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    # Look for function definition
                    if 'def save_checkpoint' in content or 'save_checkpoint(' in content:
                        print(f"\n📁 Found in: {filepath}")
                        
                        # Show context
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if 'save_checkpoint' in line:
                                start = max(0, i-2)
                                end = min(len(lines), i+3)
                                print(f"   Lines {start+1}-{end}:")
                                for j in range(start, end):
                                    print(f"     {j+1}: {lines[j]}")
                                break
                                
                except Exception as e:
                    print(f"   Error reading {filepath}: {e}")
                    continue

if __name__ == "__main__":
    find_function_in_files()
    
    # Also check what's in models directory
    print("\n" + "="*60)
    print("Checking models directory...")
    models_path = "./models"
    if os.path.exists(models_path):
        for item in os.listdir(models_path):
            item_path = os.path.join(models_path, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                print(f"📄 {item} ({size} bytes)")
                # Check if it's a PyTorch file
                if size > 100:  # Likely a model file
                    try:
                        import torch
                        data = torch.load(item_path, map_location='cpu')
                        print(f"   Contains keys: {list(data.keys())}")
                    except:
                        print(f"   Not a PyTorch file or corrupted")
            else:
                print(f"📁 {item}/")