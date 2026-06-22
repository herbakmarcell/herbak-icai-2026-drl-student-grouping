# Optimizing Collaborative Learning: A Hard-Constraint Reinforcement Learning Approach

This repository contains the full source code, datasets, and execution scripts for the paper: **"Optimizing Collaborative Learning: A Hard-Constraint Reinforcement Learning Approach to Fair Team Composition"**.

## Abstract
Collaborative learning requires pedagogically balanced teams to function effectively. However, satisfying multiple strict constraints transforms team composition into an NP-hard combinatorial optimization problem. This repository implements a Deep Reinforcement Learning (DRL) framework using Stable-Baselines3 (DQN) augmented with Safe RL constraints (Action Masking and Cubic Reward Shaping) and Deep Q-learning from Demonstrations (DQfD) to solve this bottleneck. The DRL framework achieves massive speedups over exact mathematical solvers (Z3 OMT) while strictly adhering to pedagogical rules.

## Installation

1. Clone this repository.
2. Ensure you have Python 3.9+ installed.
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Repository Structure

- `src/`: Core logic including the Gymnasium RL environment (`env.py`), the DQN agent logic (`rl_agent.py`), the Z3 solver (`smt_solver.py`), and hyperparameters (`config.py`).
- `scripts/`: Executable scripts to run experiments.
  - `experiments.py`: Runs the full 10-seed formal ablation study isolating the impacts of action masking, penalties, safety overrides, and DQfD preloading.
  - `main.py`: A simple entrypoint script.
  - `generate_stability.py`: Runs inference stability analysis across 10 randomized dataset seeds.
  - `plot_learning_curves.py`: Parses the training logs to generate learning curves.
- `data/`: Contains the evaluation dataset (`merged_dataset.csv`).
- `plots/`: Output directory for generated visualizations.
- `results/`: Output directory for generated statistics (`experiment_stats.csv`).
- `paper/`: The LaTeX manuscript and associated figures.

## Usage

**To run the primary ablation study:**
```bash
python scripts/experiments.py
```
*Note: This will execute the full 5-configuration ablation study over 10 random seeds. It will output metrics to `results/experiment_stats.csv` and generate comparative plots in `plots/`.*

**To reproduce the Seed Stability Test:**
```bash
python scripts/generate_stability.py
```

**To generate Learning Curves:**
```bash
python scripts/plot_learning_curves.py
```

## Citation
Please refer to the corresponding ICAI publication for citation details.
