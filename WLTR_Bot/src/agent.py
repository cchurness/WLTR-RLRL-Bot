import os
from stable_baselines3 import PPO

from policy import Policy

class Agent:
    def __init__(self):
        # If you need to load your model from a file this is the time to do it
        # You can do something like:
        #
        # self.actor = # your Model
        #
        # cur_dir = os.path.dirname(os.path.realpath(__file__))
        # with open(os.path.join(cur_dir, 'model.p'), 'rb') as file:
        #     model = pickle.load(file)
        # self.actor.load_state_dict(model)
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(cur_dir, 'wltr_model')
        model = PPO.load(model_path, device='cpu')

        self.policy = Policy(model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net)

    def act(self, state):
        # Evaluate your model here
        action, _ = self.policy(state)
        return action
