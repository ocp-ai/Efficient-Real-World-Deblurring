# find_normalization.py
import os

print("Searching for normalization in code...")
print("="*60)

# Search common files
files_to_check = [
    "./data/dataset_tools/datapipeline.py",
    "./data/__init__.py", 
    "./data/dataset.py",
    "./tools/trainer.py",
    "./options/train/RSBlur.yml"
]

for filepath in files_to_check:
    if os.path.exists(filepath):
        print(f"\n🔍 Checking: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Look for normalization keywords
                keywords = ['normalize', 'norm_range', 'mean', 'std', 'ToTensor', 'Normalize']
                found = []
                
                for line in content.split('\n'):
                    if any(keyword in line.lower() for keyword in keywords):
                        # Skip comments and empty lines
                        if line.strip() and not line.strip().startswith('#'):
                            found.append(line.strip())
                
                if found:
                    print("Found normalization settings:")
                    for line in found[:10]:  # Show first 10
                        print(f"  {line}")
                else:
                    print("  No normalization found")
                    
        except Exception as e:
            print(f"  Error reading: {e}")