import os
import time
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from env import StudentGroupingEnv, LossCallback

def run_rl_analysis(df_train, df_test, group_size=10, learning_rate=0.001, timesteps=50000, device=0, 
                    cubic_penalty=True, safety_overrides=True, preloading=True, action_masking=True):
    
    log_dir = "tmp_rl_monitor/"
    os.makedirs(log_dir, exist_ok=True)
    loss_callback = LossCallback()
    
    start_learning_time = time.time()
    
    env_train = Monitor(StudentGroupingEnv(df_train, group_size=group_size, 
                                           cubic_penalty=cubic_penalty, 
                                           safety_overrides=safety_overrides), log_dir)
    
    model = DQN("MlpPolicy", env_train, verbose=0, learning_rate=learning_rate, device=device, tau=0.01)
    
    if preloading:
        env_expert = StudentGroupingEnv(df_train, group_size=group_size, 
                                        cubic_penalty=cubic_penalty, 
                                        safety_overrides=safety_overrides)
        obs, _ = env_expert.reset()
        
        expert_steps = 2000 
        for _ in range(expert_steps):
            student = env_expert.students[env_expert.current_student_idx]
            this_gender = student[2]
            this_exam = student[5]
                
            males = sum(1 for s in env_expert.current_group if s[2] > 0.5)
            females = len(env_expert.current_group) - males
            limit = int(group_size * 0.6)
                
            action = 1 
                
            if (this_gender > 0.5 and males >= limit) or (this_gender <= 0.5 and females >= limit):
                action = 0 
                    
            if action == 1 and len(env_expert.current_group) > 0:
                current_group_avg = np.mean([s[5] for s in env_expert.current_group])
                if current_group_avg > 0.80 and this_exam > 0.80:
                    action = 0
                elif current_group_avg < 0.65 and this_exam < 0.65:
                    action = 0

            next_obs, reward, done, _, _ = env_expert.step(action)
            model.replay_buffer.add(obs, next_obs, np.array([action]), np.array([reward]), np.array([done]), [{}])
            obs = next_obs
            if done:
                obs, _ = env_expert.reset()
                
    model.learn(total_timesteps=timesteps, callback=loss_callback)
    training_time = time.time() - start_learning_time
    
    model_path = "drl_student_agent_final"
    model.save(model_path)
    
    # Inference phase
    start_clustering_time = time.time()
    
    ungrouped = df_test.sort_values(by='ExamScore', ascending=False).reset_index(drop=True)
    groups = []
    fallback_pool = pd.DataFrame()
    deadlocks = 0
    
    while len(ungrouped) >= group_size:
        WINDOW_SIZE = int(group_size * 1.5)
        window = ungrouped.iloc[:WINDOW_SIZE].reset_index(drop=True)
        temp_env = StudentGroupingEnv(window, group_size=group_size, 
                                      cubic_penalty=cubic_penalty, 
                                      safety_overrides=safety_overrides)
        obs, _ = temp_env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            # Action Masking
            if action_masking and action == 1:
                males = sum(1 for s in temp_env.current_group if s[2] > 0.5)
                females = len(temp_env.current_group) - males
                this_gender = temp_env.students[temp_env.current_student_idx][2]
                limit = int(temp_env.group_size * 0.6)
                if (this_gender > 0.5 and males >= limit) or (this_gender <= 0.5 and females >= limit):
                    action = 0
                    
            obs, _, done, _, _ = temp_env.step(action)
            
        indices = temp_env.current_group_indices
        
        if len(indices) == group_size:
            groups.append(window.iloc[indices].values)
            ungrouped = ungrouped.drop(ungrouped.index[indices]).reset_index(drop=True)
        else:
            deadlocks += 1
            fallback_pool = pd.concat([fallback_pool, ungrouped.iloc[[0]]])
            ungrouped = ungrouped.iloc[1:].reset_index(drop=True)
            
    if len(ungrouped) > 0: 
        fallback_pool = pd.concat([fallback_pool, ungrouped])
        
    if not fallback_pool.empty: 
        groups.append(fallback_pool.values)
        
    inference_time = time.time() - start_clustering_time
    
    return groups, training_time, inference_time, deadlocks, log_dir, model, loss_callback

def run_robust_inference(model, df_test, group_size=10, cubic_penalty=True, safety_overrides=True, action_masking=True):
    ungrouped = df_test.copy().reset_index(drop=True)
    groups = []
    fallback_pool = pd.DataFrame()
    deadlocks = 0
    
    while len(ungrouped) >= group_size:
        WINDOW_SIZE = int(group_size * 1.5)
        window = ungrouped.iloc[:WINDOW_SIZE].reset_index(drop=True)
        
        temp_env = StudentGroupingEnv(window, group_size=group_size, cubic_penalty=cubic_penalty, safety_overrides=safety_overrides)
        obs, _ = temp_env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if action_masking and action == 1:
                males = sum(1 for s in temp_env.current_group if s[2] > 0.5)
                females = len(temp_env.current_group) - males
                this_gender = temp_env.students[temp_env.current_student_idx][2]
                limit = int(temp_env.group_size * 0.6)
                if (this_gender > 0.5 and males >= limit) or (this_gender <= 0.5 and females >= limit):
                    action = 0
            obs, _, done, _, _ = temp_env.step(action)
            
        indices = temp_env.current_group_indices
        
        if len(indices) == group_size:
            groups.append(window.iloc[indices].values)
            ungrouped = ungrouped.drop(ungrouped.index[indices]).reset_index(drop=True)
        else:
            deadlocks += 1
            fallback_pool = pd.concat([fallback_pool, ungrouped.iloc[[0]]])
            ungrouped = ungrouped.iloc[1:].reset_index(drop=True)
            
    if len(ungrouped) > 0:
        fallback_pool = pd.concat([fallback_pool, ungrouped])
        
    if not fallback_pool.empty:
        groups.append(fallback_pool.values)
        
    return groups, deadlocks
