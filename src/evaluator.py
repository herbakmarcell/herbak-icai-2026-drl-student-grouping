import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rl_agent import run_robust_inference

def get_full_stats(groups):
    if not groups: return 0,0,0,0,0,0
    avg_gaps = []
    avg_engagement = []
    avg_diversity = []
    avg_imbalance = []
    
    for g in groups:
        gap = (np.std(g[:,5])*0.7 + np.std(g[:,4])*0.3)
        avg_gaps.append(gap)
        eng = np.mean(g[:,0]) + np.mean(g[:,3])
        avg_engagement.append(eng)
        div = np.sum(g[:,6]) + np.sum(g[:,1])
        avg_diversity.append(div)
        
        males = np.sum(g[:, 2])
        diff = abs(males - (len(g)/2))
        avg_imbalance.append(diff)
        
    return np.mean(avg_gaps), np.mean(avg_engagement), np.mean(avg_diversity), \
           np.mean(avg_imbalance), np.mean(groups[0][:,5]), np.mean(groups[-1][:,5])

def test_seed_stability(model, df_test, n_runs=20, group_size=10, cubic_penalty=True, safety_overrides=True, action_masking=True):
    print(f"Running Seed Stability Test ({n_runs} iterations)...")
    run_variances = []
    
    for seed in range(n_runs):
        df_scrambled = df_test.sample(frac=1, random_state=seed).reset_index(drop=True)
        groups, _ = run_robust_inference(model, df_scrambled, group_size=group_size, 
                                         cubic_penalty=cubic_penalty, 
                                         safety_overrides=safety_overrides,
                                         action_masking=action_masking)
        gaps = []
        for g in groups:
            if len(g) >= 2: 
                weighted_gap = (np.std(g[:,5]) * 0.7) + (np.std(g[:,4]) * 0.3)
                gaps.append(weighted_gap)
                
        run_variances.append(np.mean(gaps))
        
    avg_total_variance = np.mean(run_variances)
    max_deviation = np.max(np.abs(np.array(run_variances) - avg_total_variance))
    
    print("-" * 30)
    print(f"Mean Skill Variance: {avg_total_variance:.4f}")
    print(f"Maximum Deviation:  ±{max_deviation:.4f}")
    
    if max_deviation <= 0.02:
        print("Verdict: POLICY IS ROBUST (Deviation <= 0.02)")
    else:
        print("Verdict: HIGH VARIANCE DETECTED")
        
    return run_variances
