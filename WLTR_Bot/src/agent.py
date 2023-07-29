import os

import torch

class Agent:
    def __init__(self):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(cur_dir, 'policy_model.pt')

        self.policy: torch.nn.module = torch.jit.load(model_path)
        self.policy.eval()

    def act(self, state):
        # Evaluate your model here
        state = torch.from_numpy(state).float()
        action, _ = self.policy(state)
        action = action.detach().numpy()
        return action
