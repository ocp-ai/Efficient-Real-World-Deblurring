# check_training_logs.py
import os
import json

print("Checking training logs...")
print("="*60)

# Check if there are any log files
log_files = []
for root, dirs, files in os.walk("."):
    for file in files:
        if 'log' in file.lower() or file.endswith('.txt'):
            log_files.append(os.path.join(root, file))

print(f"Found {len(log_files)} potential log files")

# Look for training output
print("\nSearching for training output...")
training_output = []
for file in log_files[:5]:  # Check first 5
    try:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if 'PSNR:' in content and 'Epoch' in content:
                print(f"\n📄 Found in: {file}")
                # Extract PSNR values
                lines = content.split('\n')
                psnr_values = []
                for line in lines:
                    if 'PSNR:' in line and 'Epoch' in line:
                        # Extract PSNR value
                        import re
                        match = re.search(r'PSNR:\s*([\d\.]+)', line)
                        if match:
                            psnr = float(match.group(1))
                            # Extract epoch
                            epoch_match = re.search(r'Epoch\s*(\d+)', line)
                            epoch = int(epoch_match.group(1)) if epoch_match else 0
                            psnr_values.append((epoch, psnr))
                            print(f"   Epoch {epoch}: PSNR = {psnr}")
                
                if psnr_values:
                    best_epoch, best_psnr = max(psnr_values, key=lambda x: x[1])
                    print(f"\n   🏆 Best: Epoch {best_epoch} with PSNR {best_psnr}")
                    training_output.append((file, best_epoch, best_psnr))
    except Exception as e:
        pass

if not training_output:
    print("\n❌ No training logs found with PSNR values")
    print("Let's check the terminal output directly...")