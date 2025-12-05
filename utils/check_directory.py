# check_directory.py
import os

dir_path = "./models/My_Microscope_NAFNet_C16_L14"
print(f"Checking directory: {dir_path}")

if os.path.exists(dir_path) and os.path.isdir(dir_path):
    print("✅ Is a directory")
    files = os.listdir(dir_path)
    print(f"Contents ({len(files)} items):")
    for file in files:
        full_path = os.path.join(dir_path, file)
        if os.path.isfile(full_path):
            size = os.path.getsize(full_path)
            print(f"  📄 {file} ({size:,} bytes)")
        else:
            print(f"  📁 {file}/")
            
    # If directory is empty, we can rename it
    if len(files) == 0:
        print("\nDirectory is empty - we can use it differently")
    else:
        print("\nDirectory has files - need to check if they're checkpoints")
else:
    print("❌ Directory doesn't exist or is not a directory")