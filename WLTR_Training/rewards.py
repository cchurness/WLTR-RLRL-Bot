from rlgym.utils import RewardFunction
from rlgym.utils.reward_functions.common_rewards import EventReward

class Reward(RewardFunction):
    def __init__(self):
        self.reset(None)

    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        return 0.0