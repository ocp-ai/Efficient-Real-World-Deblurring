# check_crop_dataset.py
import sys

print("Checking Crop_Dataset normalization...")
print("="*60)

try:
    # Try to import the dataset class
    from data.dataset_tools.datapipeline import Crop_Dataset
    
    # Check the source file
    import inspect
    source = inspect.getsource(Crop_Dataset)
    
    # Look for normalization
    if 'normalize' in source or 'Normalize' in source:
        print("✅ Crop_Dataset has normalization")
        
        # Extract relevant lines
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'normalize' in line.lower() or 'ToTensor' in line or 'Normalize' in line:
                print(f"  Line {i}: {line.strip()}")
    else:
        print("❌ No normalization found in Crop_Dataset")
        
except ImportError as e:
    print(f"❌ Could not import: {e}")
    
    # Try to find the file directly
    #datapipeline_path = "./data/dataset_tools/datapipeline.py"
    datapipeline_path = r"C:\Users\setup_vxkejr\pytorch-CycleGAN-and-pix2pix\CycleGAN.ipynb"
    if os.path.exists(datapipeline_path):
        print(f"\nReading {datapipeline_path} directly...")
        with open(datapipeline_path, 'r') as f:
            content = f.read()
            
            # Look for __init__ method of Crop_Dataset
            if 'class Crop_Dataset' in content:
                # Extract the class
                start = content.find('class Crop_Dataset')
                # Find next class or end of file
                next_class = content.find('\nclass ', start + 1)
                if next_class == -1:
                    class_content = content[start:]
                else:
                    class_content = content[start:next_class]
                
                print(f"\nCrop_Dataset __init__ parameters:")
                # Find __init__ method
                init_start = class_content.find('def __init__')
                if init_start != -1:
                    # Get the signature
                    init_end = class_content.find('):', init_start) + 2
                    init_sig = class_content[init_start:init_end]
                    print(f"  {init_sig}")
                    
                    # Check for tensor_transform
                    if 'tensor_transform' in class_content:
                        print("\n✅ Crop_Dataset uses tensor_transform")
                        # Find where it's used
                        lines = class_content.split('\n')
                        for line in lines:
                            if 'tensor_transform' in line:
                                print(f"  {line.strip()}")