import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from config import *
from data import prepare_data
from rl_agent import run_rl_analysis, run_robust_inference
from evaluator import get_full_stats

def main():
    print("Loading data...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'merged_dataset.csv')
    full_data, train_data, df_rl, _ = prepare_data(data_path, STUDENT_COUNT, RANDOM_STATE)
    
    print("Training Baseline Model (Once)...")
    params = {"action_masking": True, "cubic_penalty": True, "safety_overrides": True, "preloading": True}
    
    df_test_initial = df_rl.sample(frac=1, random_state=0).reset_index(drop=True)
    _, _, _, _, _, model, _ = run_rl_analysis(
        train_data, df_test_initial, group_size=GROUP_SIZE, learning_rate=LEARNING_RATE,
        timesteps=TIMESTEPS, device=device, **params
    )
    
    print("\nRunning Inference Stability Test over 10 randomized seeds...")
    skill_variances = []
    
    for i in range(10):
        df_test_scrambled = df_rl.sample(frac=1, random_state=i).reset_index(drop=True)
        groups, deadlocks = run_robust_inference(
            model, df_test_scrambled, group_size=GROUP_SIZE,
            cubic_penalty=True, safety_overrides=True, action_masking=True
        )
        rl_gap, rl_eng, rl_div, rl_imb, _, _ = get_full_stats(groups)
        skill_variances.append(rl_gap)
        print(f"  Seed {i}: Skill Variance = {rl_gap:.4f}")
        
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), skill_variances, marker='o', linestyle='-', color='#4c72b0')
    
    mean_val = np.mean(skill_variances)
    plt.axhline(y=mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.3f}')
    
    plt.fill_between(range(1, 11), mean_val-0.02, mean_val+0.02, color='r', alpha=0.1, label='±0.02 Margin')
    
    plt.title('Seed Stability: Inference Across 10 Randomized Initializations')
    plt.xlabel('Randomized Initialization Seed')
    plt.ylabel('Average Weighted Skill Variance')
    
    # Ensure y-axis bounds look nice
    y_min = min(mean_val - 0.05, min(skill_variances) - 0.01)
    y_max = max(mean_val + 0.05, max(skill_variances) + 0.01)
    plt.ylim(y_min, y_max)
    plt.xticks(range(1, 11))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    paper_dir = os.path.join(os.path.dirname(__file__), '..', 'paper')
    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    
    os.makedirs(paper_dir, exist_ok=True)
    plt.savefig(os.path.join(paper_dir, 'drl_seed_stability.png'))
    
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'drl_seed_stability.png'))
    
    print("\nSaved drl_seed_stability.png")

if __name__ == "__main__":
    main()
