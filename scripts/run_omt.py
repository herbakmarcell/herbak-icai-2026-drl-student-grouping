import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from config import *
from data import prepare_data
from smt_solver import run_smt
from evaluator import get_full_stats

def main():
    print("Loading data...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'merged_dataset.csv')
    full_data, train_data, df_test, _ = prepare_data(data_path, STUDENT_COUNT, RANDOM_STATE)
    
    if full_data is None:
        return
        
    print(f"\nRunning Z3 OMT Solver on {STUDENT_COUNT} students...")
    print("Warning: Exact mathematical solvers have extreme computational complexity.")
    print("This execution may take several hours to complete.")
    
    results = {"train_time": [], "inference_time": [], "skill_variance": [], 
               "engagement": [], "diversity": [], "gender_imbalance": [], "deadlocks": []}
               
    # Even running once takes massive time, but we keep the structure
    df_test_scrambled = df_test.sample(frac=1, random_state=0).reset_index(drop=True)
    
    omt_groups, omt_time = run_smt(df_test_scrambled, group_size=GROUP_SIZE)
    
    if len(omt_groups) > 0:
        omt_gap, omt_eng, omt_div, omt_imb, _, _ = get_full_stats(omt_groups)
    else:
        omt_gap, omt_eng, omt_div, omt_imb = 0, 0, 0, 0
        
    results["train_time"].append(0.0)
    results["inference_time"].append(omt_time)
    results["skill_variance"].append(omt_gap)
    results["engagement"].append(omt_eng)
    results["diversity"].append(omt_div)
    results["gender_imbalance"].append(omt_imb)
    results["deadlocks"].append(0)
    
    # Save statistics
    stats_df = pd.DataFrame()
    row = {}
    for metric, values in results.items():
        row[f"{metric}_mean"] = np.mean(values)
        row[f"{metric}_std"] = 0.0
        row[f"{metric}_ci_low"] = np.mean(values)
        row[f"{metric}_ci_high"] = np.mean(values)
        
    stats_df = pd.concat([stats_df, pd.DataFrame(row, index=["Z3 OMT Solver"])])
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    stats_df.to_csv(os.path.join(results_dir, "omt_stats.csv"))
    
    print(f"\nZ3 OMT Solver finished in {omt_time:.1f} seconds.")
    print("Results saved to results/omt_stats.csv. Run scripts/experiments.py to regenerate plots with OMT included.")

if __name__ == "__main__":
    main()
