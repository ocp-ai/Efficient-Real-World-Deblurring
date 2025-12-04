import os
import sys

# Add your project to path
sys.path.append('.')

# Import the save_checkpoint function if possible
try:
    from utils.utils import save_checkpoint
    print("✅ Found save_checkpoint in utils.utils")
except ImportError:
    try:
        # Try another import path
        from tools.trainer import save_checkpoint
        print("✅ Found save_checkpoint in tools.trainer")
    except ImportError:
        print("❌ Cannot find save_checkpoint function")
        # Let's trace where it might be
        import subprocess
        result = subprocess.run(['findstr', '/s', '/i', 'def save_checkpoint', '*.py'], 
                               capture_output=True, text=True, shell=True)
        print(f"Found in files: {result.stdout}")

# Check the models directory
models_path = "./models"
print(f"\nChecking models directory: {models_path}")
print(f"Absolute path: {os.path.abspath(models_path)}")

if os.path.exists(models_path):
    print(f"✅ Directory exists")
    files = os.listdir(models_path)
    print(f"Files in directory: {files}")
else:
    print(f"❌ Directory does not exist")
    print("Creating directory...")
    os.makedirs(models_path, exist_ok=True)
    
# Test if we can write to it
test_file = os.path.join(models_path, "test.txt")
try:
    with open(test_file, 'w') as f:
        f.write("test")
    print(f"✅ Can write to directory")
    os.remove(test_file)
except Exception as e:
    print(f"❌ Cannot write to directory: {e}")