import torch

device = 0 if torch.cuda.is_available() else -1

LEARNING_RATE = 0.001
TIMESTEPS = 50000

STUDENT_COUNT = 40
GROUP_SIZE = 10

RANDOM_STATE = 42
