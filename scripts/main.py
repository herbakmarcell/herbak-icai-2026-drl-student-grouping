import tracemalloc
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from config import *
from data import prepare_data
from rl_agent import run_rl_analysis
from evaluator import get_full_stats

def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'merged_dataset.csv')
    full_data, train_data, df_rl, _ = prepare_data(data_path, STUDENT_COUNT, RANDOM_STATE)
    
    if full_data is None:
        return

    print("\nStarting DRL Analysis Run...")
    tracemalloc.start()
    rl_groups, rl_train_time, rl_inf_time, rl_deadlocks, rl_log_dir, trained_model, _ = run_rl_analysis(
        train_data, df_rl, group_size=GROUP_SIZE, learning_rate=LEARNING_RATE, timesteps=TIMESTEPS, device=device
    )
    _, rl_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rl_gap, rl_eng, rl_div, rl_imbalance, rl_g1, rl_last = get_full_stats(rl_groups)
    
    print(f"\nRESULTS (N={STUDENT_COUNT})")
    print(f"Train Time: {rl_train_time:.4f}s")
    print(f"Inference Time: {rl_inf_time:.4f}s")
    print(f"Memory Peak: {rl_peak/10**6:.2f} MB")
    print(f"Avg Skill Variance: {rl_gap:.3f}")
    print(f"Avg Engagement: {rl_eng:.3f}")
    print(f"Avg Diversity: {rl_div:.1f}")
    print(f"Avg Gender Imbalance: {rl_imbalance:.1f}")
    print(f"Deadlocks: {rl_deadlocks}")

if __name__ == "__main__":
    main()
