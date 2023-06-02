import rlgym
from stable_baselines3 import PPO
from rlgym_tools.sb3_utils import SB3SingleInstanceEnv

import pickle

gym_env = rlgym.make(
    use_injector=True, 
    spawn_opponents=True, 
)

env = SB3SingleInstanceEnv(gym_env)

# Initialize PPO from stable_baselines3
model = PPO("MlpPolicy", env=env, verbose=1)

# Train our agent!
model.learn(total_timesteps=int(1e4))

# Save our agent using pickle
with open("WLTR/src/model.p", "wb") as file:
    pickle.dump(model, file)