import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log(filepath):
    epochs = []
    success_rates = []
    iei_vals = []
    sei_vals = []
    
    with open(filepath, 'r') as f:
        current_epoch = None
        current_success = None
        current_iei = None
        current_sei = None
        
        for line in f:
            epoch_match = re.search(r'^Epoch (\d+)', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                
            success_match = re.search(r'Success:\s+([\d\.]+)', line)
            if success_match:
                current_success = float(success_match.group(1))
                
            iei_match = re.search(r'IEI \(msg entropy\):\s+([\d\.]+)', line)
            if iei_match:
                current_iei = float(iei_match.group(1))
                
            sei_match = re.search(r'SEI \(cosine sim\):\s+([\d\.]+)', line)
            if sei_match:
                current_sei = float(sei_match.group(1))
                
                # Assuming SEI is sequentially the last printed metric per epoch
                if current_epoch is not None and current_success is not None and current_iei is not None:
                    epochs.append(current_epoch)
                    success_rates.append(current_success)
                    iei_vals.append(current_iei)
                    sei_vals.append(current_sei)
                    current_epoch = None
                    current_success = None
                    current_iei = None
                    current_sei = None
                    
    return np.array(epochs), np.array(success_rates), np.array(iei_vals), np.array(sei_vals)

def main():
    log_folder = 'logs'
    # Match pattern for L1 500 metrics 
    pattern = os.path.join(log_folder, '*_L1_500_metrics*.log')
    files = glob.glob(pattern)
    
    # Pre-defined colors for models for consistency across plots
    colors = {
        'MAGIC': 'blue',
        'TARMAC': 'orange',
        'GACOMM': 'green',
        'COMMNET': 'red',
        'IC3NET': 'purple'
    }
    
    plt.figure(figsize=(10, 6))
    for f in files:
        model_name = os.path.basename(f).split('_L1')[0].replace('train_', '').upper()
        epochs, success, iei, sei = parse_log(f)
        
        # Calculate ratio, add epsilon to avoid division by zero
        iei_ratio = iei / (success + 1e-8)
        
        color = colors.get(model_name, None)
        plt.plot(epochs, iei_ratio, label=model_name, color=color, alpha=0.8)
        
    plt.xlabel('Epoch')
    plt.ylabel('IEI / Success Rate')
    plt.title('IEI (Message Entropy) / Success Rate over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('iei_ratio_plot.png')
    plt.close()
    print("Saved iei_ratio_plot.png")
    
    plt.figure(figsize=(10, 6))
    for f in files:
        model_name = os.path.basename(f).split('_L1')[0].replace('train_', '').upper()
        epochs, success, iei, sei = parse_log(f)
        
        # Calculate ratio
        sei_ratio = sei / (success + 1e-8)
        
        color = colors.get(model_name, None)
        plt.plot(epochs, sei_ratio, label=model_name, color=color, alpha=0.8)
        
    plt.xlabel('Epoch')
    plt.ylabel('SEI / Success Rate')
    plt.title('SEI (Cosine Similarity) / Success Rate over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sei_ratio_plot.png')
    plt.close()
    print("Saved sei_ratio_plot.png")

if __name__ == '__main__':
    main()
