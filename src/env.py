import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback

class LossCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LossCallback, self).__init__(verbose)
        self.losses = []
        self.timesteps = []

    def _on_step(self) -> bool:
        if "train/loss" in self.logger.name_to_value:
            self.losses.append(self.logger.name_to_value["train/loss"])
            self.timesteps.append(self.num_timesteps)
        return True

class StudentGroupingEnv(gym.Env):
    def __init__(self, students_df, group_size=10, cubic_penalty=True, safety_overrides=True):
        super(StudentGroupingEnv, self).__init__()
        self.students = students_df.values
        self.num_students = len(students_df)
        self.group_size = group_size
        self.cubic_penalty = cubic_penalty
        self.safety_overrides = safety_overrides
        
        self.current_student_idx = 0
        
        # 11 inputs: 7 features + [Size, MaleRatio, PoolRatio, AvgExam]
        self.obs_dim = 11
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.current_group = []
        self.current_group_indices = [] 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_student_idx = 0
        self.current_group = []
        self.current_group_indices = []
        return self._get_obs(), {}

    def _get_obs(self):
        if self.current_student_idx >= self.num_students:
             return np.zeros(self.obs_dim, dtype=np.float32)
             
        student = self.students[self.current_student_idx]
        curr_size = len(self.current_group)
        curr_males = sum(1 for s in self.current_group if s[2] > 0.5)
        
        if curr_size > 0:
            grp_male_ratio = curr_males / curr_size
            grp_exam_mean = np.mean([s[5] for s in self.current_group])
        else:
            grp_male_ratio = 0.5
            grp_exam_mean = student[5]
            
        remaining_idx = self.num_students - (self.current_student_idx + 1)
        if remaining_idx > 0:
            future_data = self.students[self.current_student_idx+1:]
            pool_male_ratio = np.sum(future_data[:, 2]) / len(future_data)
        else:
            pool_male_ratio = 0.5
        
        norm_size = curr_size / self.group_size
        context = np.array([norm_size, grp_male_ratio, pool_male_ratio, grp_exam_mean])
        return np.concatenate([student, context]).astype(np.float32)

    def calculate_reward(self, group):
        group_array = np.array(group)
        
        # A. PENALTIES (Constraints)
        males = np.sum(group_array[:, 2])
        diff = abs(males - (len(group)/2))
        
        if self.cubic_penalty:
            gender_penalty = -(diff**3 * 2.0)
        else:
            gender_penalty = 0.0
            
        exam_std = np.std(group_array[:, 5])
        assign_std = np.std(group_array[:, 4])
        weighted_gap = (exam_std * 0.7) + (assign_std * 0.3)
        skill_penalty = -(weighted_gap * 100.0)
        
        # B. BONUSES (Extra Features)
        avg_attend = np.mean(group_array[:, 0])
        avg_discuss = np.mean(group_array[:, 3])
        engagement = (avg_attend * 5.0) + (avg_discuss * 5.0)
        
        has_edutech = 1 if np.sum(group_array[:, 6]) > 0 else 0
        has_extra = 1 if np.sum(group_array[:, 1]) > 0 else 0
        diversity = (has_edutech * 5.0) + (has_extra * 3.0)
        
        return 100 + gender_penalty + skill_penalty + engagement + diversity

    def step(self, action):
        done = False
        reward = 0
        remaining = self.num_students - self.current_student_idx
        needed = self.group_size - len(self.current_group)
        force = (remaining <= needed)
        
        if action == 1 or force:
            if not force:
                males = sum(1 for s in self.current_group if s[2] > 0.5)
                females = len(self.current_group) - males
                this_gender = self.students[self.current_student_idx][2]
                limit = int(self.group_size * 0.6)
                if (this_gender > 0.5 and males >= limit) or (this_gender <= 0.5 and females >= limit):
                    if self.safety_overrides:
                        valid_add = False
                    else:
                        valid_add = True
                    reward = -10 
                else:
                    valid_add = True
            else:
                valid_add = True
                
            if valid_add:
                self.current_group.append(self.students[self.current_student_idx])
                self.current_group_indices.append(self.current_student_idx)
        
        self.current_student_idx += 1
        
        if len(self.current_group) == self.group_size:
            reward += self.calculate_reward(self.current_group)
            done = True 
        elif self.current_student_idx >= self.num_students:
            done = True
            reward = -50
            
        next_obs = self._get_obs()
        return next_obs, reward, done, False, {}
