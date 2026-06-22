import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can import from src if needed, though this script mostly just reads a CSV.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    monitor_file = os.path.join(script_dir, '..', 'tmp_rl_monitor', 'monitor.csv')
    
    if not os.path.exists(monitor_file):
        print(f"Error: Could not find {monitor_file}.")
        print("Please run experiments.py or train the model to generate the monitor logs.")
        return
    
    print(f"Loading {monitor_file}...")
    # The stable-baselines3 monitor.csv has a JSON header on the first line. 
    # The actual CSV columns start on the second line.
    df = pd.read_csv(monitor_file, skiprows=1)
    
    # df columns usually: r (reward), l (length), t (time)
    if 'r' not in df.columns or 'l' not in df.columns:
        print("Error: monitor.csv does not contain expected columns 'r' and 'l'.")
        return

    # Create a rolling window for smoother plots
    window_size = min(500, max(10, len(df) // 20))
    
    df['r_smooth'] = df['r'].rolling(window=window_size).mean()
    df['l_smooth'] = df['l'].rolling(window=window_size).mean()
    
    plots_dir = os.path.join(script_dir, '..', 'plots')
    paper_dir = os.path.join(script_dir, '..', 'paper')
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(paper_dir, exist_ok=True)
    
    # 1. Plot Training Reward / Loss proxy
    plt.figure(figsize=(8, 5))
    plt.plot(df.index, df['r'], alpha=0.3, color='#c44e52', label='Episodic Reward')
    plt.plot(df.index, df['r_smooth'], color='#c44e52', linewidth=2, label=f'Moving Avg ({window_size} eps)')
    plt.title('DQN Training Convergence (Reward)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'drl_training_loss.png'))
    plt.savefig(os.path.join(paper_dir, 'drl_training_loss.png'))
    print("Saved drl_training_loss.png")
    
    # 2. Plot Episode Length
    plt.figure(figsize=(8, 5))
    plt.plot(df.index, df['l'], alpha=0.3, color='#4c72b0', label='Episode Length')
    plt.plot(df.index, df['l_smooth'], color='#4c72b0', linewidth=2, label=f'Moving Avg ({window_size} eps)')
    plt.title('Agent Evaluation Rate per Candidate')
    plt.xlabel('Episode')
    plt.ylabel('Evaluated Candidates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'drl_episode_length.png'))
    plt.savefig(os.path.join(paper_dir, 'drl_episode_length.png'))
    print("Saved drl_episode_length.png")

if __name__ == "__main__":
    main()
