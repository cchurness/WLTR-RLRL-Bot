import numpy as np

from rlgym_sim.envs import Match
from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import EventReward, VelocityPlayerToBallReward, VelocityBallToGoalReward
from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition
from rlgym_sim.utils.state_setters import RandomState
from rlgym_sim_tools.extra_action_parsers.lookup_act import LookupAction
from rlgym_sim_tools.extra_obs.advanced_padder import AdvancedObsPadder
from rlgym_sim.utils.obs_builders import DefaultObs

from rlgym_sim_tools.sb3_utils import SB3MultipleInstanceEnv
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from stable_baselines3.common.callbacks import CheckpointCallback

if __name__ == "__main__":
    frame_skip = 6          # Number of ticks to repeat an action
    half_life_seconds = 8   # Easier to conceptualize, after this many seconds the reward discount is 0.5

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
    agents_per_match = 2
    num_instances = 10
    target_steps = 1_000_000
    steps = target_steps // (num_instances * agents_per_match) #making sure the experience counts line up properly
    batch_size = 100_000
    training_interval = 25_000_000
    mmr_save_frequency = 50_000_000

    def exit_save(model):
        model.save("models/ppo_model")

    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=1,
            reward_function=CombinedReward(
            (
                VelocityPlayerToBallReward(),
                VelocityBallToGoalReward(),
                EventReward(
                    team_goal=10.0,
                    concede=-10.0,
                    boost_pickup=2,
                ),
            ),
            (0.1, 1.0, 1.0)),
            # self_play=True,  in rlgym 1.2 'self_play' is depreciated. Uncomment line if using an earlier version and comment out spawn_opponents
            spawn_opponents=True,
            terminal_conditions=[TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition()],
            obs_builder=AdvancedObsPadder(expanding=True),
            state_setter=RandomState(),
            action_parser=LookupAction()
        )
    
    env = SB3MultipleInstanceEnv(get_match, num_instances, wait_time=0, tick_skip=8, dodge_deadzone=0.5, copy_gamestate_every_step=False)            # Start 1 instances, waiting 60 seconds between each
    env = VecCheckNan(env)                                # Optional
    env = VecMonitor(env)                                 # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Highly recommended, normalizes rewards

    try:
        model = PPO.load(
            "models/ppo_model",
            env,
            device="auto",
            custom_objects={"n_envs": env.num_envs, "n_steps" : steps}, #automatically adjusts to users changing instance count, may encounter shaping error otherwise
        )
        print("Loaded previous exit save.")
    except:
        print("No saved model found, creating new model.")
        from torch.nn import Tanh
        policy_kwargs = dict(
            activation_fn=Tanh,
            net_arch=dict(pi=[512, 512], vf=[512, 512, 512, 512]),
        )

        model = PPO(
            MlpPolicy,
            env,
            n_epochs=10,                 # PPO calls for multiple epochs
            policy_kwargs=policy_kwargs,
            learning_rate=5e-5,          # Around this is fairly common for PPO
            ent_coef=0.01,               # From PPO Atari
            vf_coef=1.,                  # From PPO Atari
            gamma=gamma,                 # Gamma as calculated using half-life
            verbose=3,                   # Print out all the info as we're going
            batch_size=batch_size,             # Batch size as high as possible within reason
            n_steps=steps,                # Number of steps to perform before optimizing network
            tensorboard_log="logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
            device="auto"                # Uses cpu
        )

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    # This saves to specified folder with a specified name
    callback = CheckpointCallback(round(5_000_000 / env.num_envs), save_path="models", name_prefix="rl_model")

    try:
        mmr_model_target_count = model.num_timesteps + mmr_save_frequency
        while True:
            #may need to reset timesteps when you're running a different number of instances than when you saved the model
            model.learn(training_interval, callback=callback, reset_num_timesteps=False, progress_bar=True) #can ignore callback if training_interval < callback target
            model.save("models/ppo_model")
            if model.num_timesteps >= mmr_model_target_count:
                model.save(f"mmr_models/{model.num_timesteps}")
                mmr_model_target_count += mmr_save_frequency
    except KeyboardInterrupt:
        print("Exiting training")

    print("Saving model")
    exit_save(model)
    print("Save complete")


# def main():
#     import argparse

#     # Parse command line arguments
#     parser = argparse.ArgumentParser(
#         prog="train.py",
#         description="Train an RL bot."
#     )

#     # Input file
#     parser.add_argument("-i", "--input",
#         type=str,
#         default=None,
#         dest="input",
#         metavar="<input_file>",
#         help="Path to a model to load and continue training.",
#         required=False
#     )

#     # Output file
#     parser.add_argument("-o", "--output",
#         type=str,
#         default=None,
#         dest="output",
#         metavar="<output_file>",
#         help="Path to save the trained model. If not specified, the model will be saved to input_file. \
#             If input_file is not specified, the model will be saved to \"policy_model.pt\" in the current \
#             working directory.",
#         required=False
#     )

#     # Training Duration (in hours)
#     parser.add_argument("-t", "--time",
#         type=float,
#         default=0.5, # 30 minutes
#         dest="duration",
#         metavar="<time>",
#         help="The duration to train the model in hours. Defaults to 1/2 hour.",
#         required=False
#     )

#     # Parse arguments into their own variables
#     args = parser.parse_args()

#     input_file: str = args.input
#     output_file: str = args.output
#     if input_file and not output_file:
#         output_file = input_file

#     duration: float = args.duration

#     import torch
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     device = torch.device("cpu")

#     # ------------------------------

#     # from policy import Policy
#     # if input_file:
#     #     print("Loading model from file...")
#     #     try:
#     #         policy: Policy = torch.jit.load(input_file).to(device)
#     #     except Exception as e:
#     #         print("Error loading model from file. Aborting.")
#     #         raise e

#     # ------------------------------

#     import rlgym as rlgym
#     from stable_baselines3 import PPO
#     from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv

#     print("Creating gym environment...")

#     from rlgym.utils.reward_functions.common_rewards import EventReward, LiuDistancePlayerToBallReward, LiuDistanceBallToGoalReward, TouchBallReward
#     from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, FaceBallReward, RewardIfBehindBall
#     from rlgym.utils.reward_functions.common_rewards import BallYCoordinateReward
#     from rlgym.utils.reward_functions import CombinedReward
#     from rlgym.envs import Match

#     from rlgym.utils.obs_builders import DefaultObs
#     from rlgym.utils.action_parsers import DefaultAction
#     from rlgym.utils.state_setters import DefaultState
#     from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition

#     from rlgym_tools.extra_action_parsers.lookup_act import LookupAction
#     from rlgym_tools.extra_obs.advanced_padder import AdvancedObsPadder

#     # Hit ball reward
#     hit_ball_reward = CombinedReward(reward_functions=[
#         EventReward(goal=1, concede=-1),
#         TouchBallReward(),
#         RewardIfBehindBall(FaceBallReward()),
#         RewardIfBehindBall(LiuDistancePlayerToBallReward()),
#         LiuDistanceBallToGoalReward(),
#     ], reward_weights=[
#         1_000_000,
#         1_000,
#         1,
#         5,
#         10,
#     ])

#     def get_match():
#         return Match(
#             reward_function=hit_ball_reward,
#             terminal_conditions=[TimeoutCondition(1000), GoalScoredCondition()],
#             obs_builder=DefaultObs(),
#             action_parser=DefaultAction(),
#             state_setter=DefaultState(),

#             spawn_opponents=True,
#             game_speed=5
#         )

#     # # Create gym environment
#     # gym_env = rlgym.make(
#     #     use_injector=True,
#     #     spawn_opponents=True,
#     #     reward_fn=EventReward(goal=1, concede=-1),
#     # )

#     # # Wrap the gym environment in a stable_baselines3 environment for training
#     # env = SB3SingleInstanceEnv(gym_env)

#     env = SB3MultipleInstanceEnv(
#         match_func_or_matches=get_match, 
#         num_instances=1, 
#         wait_time=0,

#         launch_preference="epic_login_trick"
#     )

#     def exit_gym():
#         env.close()
#         # gym_env.close()
#         print("Gym environment closed.")

#     # Initialize PPO from stable_baselines3
#     model = PPO("MlpPolicy", env=env, verbose=1, device=device, learning_rate=0.0001)

#     # ------------------------------

#     if input_file:
#         print("Loading model from file...")
#         try:
#             model.set_parameters(input_file, device=device)
#         except Exception as e:
#             print("Error loading model from file. Aborting.")
#             exit_gym()
#             raise e
        
#     # ------------------------------

#     # Set output file if not set
#     if not output_file:
#         import os
#         cur_dir = os.path.dirname(os.path.realpath(__file__))
#         output_file = os.path.join(cur_dir, 'policy_model.zip')

#     def save_model():
#         print("Saving model...")
#         try:
#             model.save(output_file)
#         except Exception as e:
#             print("Error saving model. Cancelling training.")
#             exit_gym()
#             raise e

#     # ------------------------------

#     from time import time

#     end_time = time() + (duration * 60 * 60) # Convert hours to seconds

#     from stable_baselines3.common.callbacks import BaseCallback

#     class SaveCallback(BaseCallback):
#         def __init__(self, verbose: int = 0, freq: int = 100_000):
#             super().__init__(verbose)

#             self.freq = freq

#         def _on_step(self) -> bool:
#             if self.n_calls % self.freq == 0:
#                 save_model()

#             return True

#     try:
#         # Train our agent!
#         print("Training...")
#         # while time() < end_time:
        
#         model.learn(total_timesteps=int(10_000_000_000), log_interval=5, progress_bar=True, reset_num_timesteps=False, callback=SaveCallback())
#     except KeyboardInterrupt:
#         save_model()
#         exit_gym()

# if __name__ == "__main__":
#     main()