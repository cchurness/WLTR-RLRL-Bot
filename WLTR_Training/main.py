from rlgym.envs import Match

from rlgym.utils.reward_functions.common_rewards import EventReward

from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.action_parsers import DefaultAction
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition

from rlgym.gamelaunch.launch import LaunchPreference

def get_match():
    return Match(
        reward_function=EventReward(goal=1, concede=-1),
        terminal_conditions=[TimeoutCondition(225), GoalScoredCondition()],
        obs_builder=DefaultObs(),
        action_parser=DefaultAction(),
        state_setter=DefaultState(),

        spawn_opponents=True,
        game_speed=100,
    )

from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from stable_baselines3 import PPO

def main():
    env = SB3MultipleInstanceEnv(
        match_func_or_matches=get_match, 
        num_instances=1, 
        launch_preference=LaunchPreference.EPIC_LOGIN_TRICK
    )

    model = PPO("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=1_000)

if __name__ == "__main__":
    main()