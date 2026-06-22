# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import tracemalloc

from config import *
from data import prepare_data
from rl_agent import run_rl_analysis
from evaluator import get_full_stats

# Define configurations for the ablation study
CONFIGS = {
    "Baseline (All ON)": {"action_masking": True, "cubic_penalty": True, "safety_overrides": True, "preloading": True},
    "No Action Masking": {"action_masking": False, "cubic_penalty": True, "safety_overrides": True, "preloading": True},
    "No Cubic Penalty": {"action_masking": True, "cubic_penalty": False, "safety_overrides": True, "preloading": True},
    "No Safety Overrides": {"action_masking": True, "cubic_penalty": True, "safety_overrides": False, "preloading": True},
    "No Preloading": {"action_masking": True, "cubic_penalty": True, "safety_overrides": True, "preloading": False}
}

N_RUNS = 10

def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'merged_dataset.csv')
    full_data, train_data, df_rl, _ = prepare_data(data_path, STUDENT_COUNT, RANDOM_STATE)
    
    if full_data is None:
        return
        
    results = {cfg: {"train_time": [], "inference_time": [], "skill_variance": [], 
                     "engagement": [], "diversity": [], "gender_imbalance": [], "deadlocks": []} for cfg in CONFIGS}
                     
    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Run experiments
    for config_name, params in CONFIGS.items():
        print(f"\nRunning Configuration: {config_name}")
        for i in range(N_RUNS):
            print(f"  Run {i+1}/{N_RUNS}...")
            
            # Scramble test data for robust evaluation across seeds
            df_test_scrambled = df_rl.sample(frac=1, random_state=i).reset_index(drop=True)
            
            rl_groups, train_time, inf_time, deadlocks, _, _, _ = run_rl_analysis(
                train_data, df_test_scrambled, group_size=GROUP_SIZE, learning_rate=LEARNING_RATE, 
                timesteps=TIMESTEPS, device=device, **params
            )
            
            rl_gap, rl_eng, rl_div, rl_imb, _, _ = get_full_stats(rl_groups)
            
            results[config_name]["train_time"].append(train_time)
            results[config_name]["inference_time"].append(inf_time)
            results[config_name]["skill_variance"].append(rl_gap)
            results[config_name]["engagement"].append(rl_eng)
            results[config_name]["diversity"].append(rl_div)
            results[config_name]["gender_imbalance"].append(rl_imb)
            results[config_name]["deadlocks"].append(deadlocks)
            
    # Statistical Analysis
    stats_df = pd.DataFrame()
    for config_name, metrics in results.items():
        row = {}
        for metric, values in metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            row[f"{metric}_mean"] = mean_val
            row[f"{metric}_std"] = std_val
            # 95% CI
            if N_RUNS > 1:
                ci = stats.t.interval(0.95, len(values)-1, loc=mean_val, scale=stats.sem(values))
                row[f"{metric}_ci_low"] = ci[0] if not np.isnan(ci[0]) else mean_val
                row[f"{metric}_ci_high"] = ci[1] if not np.isnan(ci[1]) else mean_val
            else:
                row[f"{metric}_ci_low"] = mean_val
                row[f"{metric}_ci_high"] = mean_val
        
        stats_df = pd.concat([stats_df, pd.DataFrame(row, index=[config_name])])
        
    stats_df.to_csv(os.path.join(results_dir, "experiment_stats.csv"))
    print("\nSaved statistics to experiment_stats.csv")
    
    # Load OMT stats if available to include in plots
    omt_stats_path = os.path.join(results_dir, "omt_stats.csv")
    if os.path.exists(omt_stats_path):
        omt_df = pd.read_csv(omt_stats_path, index_col=0)
        stats_df = pd.concat([stats_df, omt_df])
        print("Loaded Z3 OMT Solver results from omt_stats.csv for plotting")
        
    # Significance Testing (Independent t-tests against Baseline)
    baseline_metrics = results["Baseline (All ON)"]
    print("\nStatistical Significance (p-values vs Baseline):")
    for config_name in CONFIGS:
        if config_name == "Baseline (All ON)":
            continue
        print(f"[{config_name}]")
        if N_RUNS > 1:
            for metric in ["skill_variance", "engagement", "diversity", "gender_imbalance", "deadlocks"]:
                _, p_val = stats.ttest_ind(baseline_metrics[metric], results[config_name][metric])
                print(f"  {metric}: p={p_val:.4f} {'(Significant)' if p_val < 0.05 else ''}")
        else:
            print("  Skipping significance tests (N=1)")
            
    # Plots
    labels = list(CONFIGS.keys())
    x_pos = np.arange(len(labels))
    
    # Plot 1: Skill Variance
    means = stats_df['skill_variance_mean'].values
    yerr = stats_df['skill_variance_mean'].values - stats_df['skill_variance_ci_low'].values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos, means, yerr=yerr, capsize=5, color='#4c72b0', alpha=0.8)
    ax.set_ylabel('Avg Weighted Skill Variance')
    ax.set_title('Skill Variance per Configuration (Lower is Better)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'skill_variance_comparison.png'))
    
    # Plot 2: Deadlocks
    means = stats_df['deadlocks_mean'].values
    yerr = stats_df['deadlocks_mean'].values - stats_df['deadlocks_ci_low'].values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos, means, yerr=yerr, capsize=5, color='#dd8452', alpha=0.8)
    ax.set_ylabel('Number of Deadlocks (Fallback Triggers)')
    ax.set_title('Deadlocks per Configuration (Lower is Better)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'deadlocks_comparison.png'))
    
    # Plot 3: Gender Imbalance
    means = stats_df['gender_imbalance_mean'].values
    yerr = stats_df['gender_imbalance_mean'].values - stats_df['gender_imbalance_ci_low'].values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos, means, yerr=yerr, capsize=5, color='#9370DB', alpha=0.8)
    ax.set_ylabel('Avg Deviation from Balanced Gender Ratio')
    ax.set_title('Gender Imbalance per Configuration (Lower is Better)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'gender_imbalance_comparison.png'))
    
    # Plot 4: Training Time vs Inference Time
    train_means = stats_df['train_time_mean'].values
    inf_means = stats_df['inference_time_mean'].values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_pos - 0.2, train_means, width=0.4, label='Training Time', color='#55a868')
    ax.bar(x_pos + 0.2, inf_means, width=0.4, label='Inference Time', color='#c44e52')
    ax.set_ylabel('Time (Seconds) - Log Scale')
    ax.set_title('Time taken per Configuration')
    ax.set_yscale('log')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'time_comparison.png'))
    
    print("\nSaved plots to 'plots/' directory")

if __name__ == "__main__":
    main()
