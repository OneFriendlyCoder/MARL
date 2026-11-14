import matplotlib.pyplot as plt
import re
import sys
import os

def main():
    logfile = 'eval_results_1000_runs_per_episode.txt'
    
    if not os.path.exists(logfile):
        print(f"Error: Could not find {logfile}. Run the evaluation first.")
        sys.exit(1)
        
    model_data = {}
    
    # Regex to capture: [MODEL] Episode 1 | Success: X | Cum_Success_Rate: 0.XXX
    pattern = re.compile(r'^\[([A-Z0-9_]+)\] Episode (\d+).*Cum_Success_Rate:\s*([0-9\.]+)')
    
    with open(logfile, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                model_name = match.group(1)
                episode = int(match.group(2))
                cum_success = float(match.group(3))
                
                if model_name not in model_data:
                    model_data[model_name] = {'episodes': [], 'success_rates': []}
                    
                model_data[model_name]['episodes'].append(episode)
                model_data[model_name]['success_rates'].append(cum_success)
                
    if not model_data:
        print("No evaluation success rate data found in log. Make sure evaluation has completed.")
        sys.exit(1)
        
    # Pre-defined colors for consistency
    colors = {
        'MAGIC': 'blue',
        'TARMAC': 'orange',
        'GACOMM': 'green',
        'COMMNET': 'red',
        'IC3NET': 'purple'
    }

    plt.figure(figsize=(10, 6))
    
    for model_name, data in model_data.items():
        color = colors.get(model_name, None)
        plt.plot(data['episodes'], data['success_rates'], label=model_name, color=color, alpha=0.9)
        
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Cumulative Success Rate')
    plt.title('Cumulative Success Rate over 1000 Evaluation Episodes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cumulative_success_rates.png')
    print("Saved cumulative_success_rates.png")

if __name__ == '__main__':
    main()
