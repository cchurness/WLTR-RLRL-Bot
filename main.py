import rlgym
from stable_baselines3 import PPO
from rlgym_tools.sb3_utils import SB3SingleInstanceEnv

gym_env = rlgym.make(
    use_injector=True, 
    spawn_opponents=True, 
)

env = SB3SingleInstanceEnv(gym_env)

# Initialize PPO from stable_baselines3
model = PPO("MlpPolicy", env=env, verbose=1)

# Train our agent!
model.learn(total_timesteps=int(1e3))

# Save the model
from WLTR_Training.policy import Policy

policy = Policy(model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net).to('cpu')

import torch

model_scripted = torch.jit.script(policy)

import os

cur_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(cur_dir, 'WLTR_Bot', 'src', 'policy_model.pt')

model_scripted.save(model_path)