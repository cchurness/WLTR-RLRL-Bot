import os

from stable_baselines3 import PPO
import torch

from policy import Policy

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(cur_dir, 'ppo_model.zip')
    model = PPO.load(model_path, device="cpu")

    policy: Policy = Policy( # Convert PPO to Torch model
        model.policy.mlp_extractor,
        model.policy.action_net,
        model.policy.value_net
    )
    policy.eval()

    policy_path = os.path.join(cur_dir, 'policy_model.pt')
    policy_scripted = torch.jit.script(policy)
    policy_scripted.save(policy_path)