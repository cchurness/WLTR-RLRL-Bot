def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train an RL bot."
    )

    # Input file
    parser.add_argument("-i", "--input",
        type=str,
        default=None,
        dest="input",
        metavar="<input_file>",
        help="Path to a model to load and continue training.",
        required=False
    )

    # Output file
    parser.add_argument("-o", "--output",
        type=str,
        default=None,
        dest="output",
        metavar="<output_file>",
        help="Path to save the trained model. If not specified, the model will be saved to input_file. \
            If input_file is not specified, the model will be saved to \"policy_model.pt\" in the current \
            working directory.",
        required=False
    )

    # Training Duration (in hours)
    parser.add_argument("-t", "--time",
        type=float,
        default=0.5, # 30 minutes
        dest="duration",
        metavar="<time>",
        help="The duration to train the model in hours. Defaults to 1/2 hour.",
        required=False
    )

    # Parse arguments into their own variables
    args = parser.parse_args()

    input_file: str = args.input
    output_file: str = args.output
    if input_file and not output_file:
        output_file = input_file

    duration: float = args.duration

    import torch
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # ------------------------------

    # from policy import Policy
    # if input_file:
    #     print("Loading model from file...")
    #     try:
    #         policy: Policy = torch.jit.load(input_file).to(device)
    #     except Exception as e:
    #         print("Error loading model from file. Aborting.")
    #         raise e

    # ------------------------------

    import rlgym_sim as rlgym
    from stable_baselines3 import PPO
    from rlgym_tools.sb3_utils import SB3SingleInstanceEnv, SB3MultipleInstanceEnv

    print("Creating gym environment...")

    from rlgym_sim.utils.reward_functions.common_rewards import EventReward, LiuDistancePlayerToBallReward, LiuDistanceBallToGoalReward, TouchBallReward
    from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, FaceBallReward, RewardIfBehindBall
    from rlgym_sim.utils.reward_functions.common_rewards import BallYCoordinateReward
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.envs import Match

    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.action_parsers import DefaultAction
    from rlgym_sim.utils.state_setters import DefaultState
    from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition

    # Hit ball reward
    hit_ball_reward = CombinedReward(reward_functions=[
        EventReward(goal=1, concede=-1),
        TouchBallReward(),
        RewardIfBehindBall(FaceBallReward()),
        RewardIfBehindBall(LiuDistancePlayerToBallReward()),
        LiuDistanceBallToGoalReward(),
    ], reward_weights=[
        1_000_000,
        1_000,
        1,
        5,
        10,
    ])

    def get_match():
        return Match(
            reward_function=hit_ball_reward,
            terminal_conditions=[TimeoutCondition(1000), GoalScoredCondition()],
            obs_builder=DefaultObs(),
            action_parser=DefaultAction(),
            state_setter=DefaultState(),

            spawn_opponents=True,
        )

    # # Create gym environment
    # gym_env = rlgym_sim.make(
    #     use_injector=True,
    #     spawn_opponents=True,
    #     reward_fn=EventReward(goal=1, concede=-1),
    # )

    # # Wrap the gym environment in a stable_baselines3 environment for training
    # env = SB3SingleInstanceEnv(gym_env)

    env = SB3MultipleInstanceEnv(
        match_func_or_matches=get_match, 
        num_instances=10, 
        wait_time=0,
    )

    def exit_gym():
        env.close()
        # gym_env.close()
        print("Gym environment closed.")

    # Initialize PPO from stable_baselines3
    model = PPO("MlpPolicy", env=env, verbose=1, device=device, learning_rate=0.0001)

    # ------------------------------

    if input_file:
        print("Loading model from file...")
        try:
            model.set_parameters(input_file, device=device)
        except Exception as e:
            print("Error loading model from file. Aborting.")
            exit_gym()
            raise e
        
    # ------------------------------

    # Set output file if not set
    if not output_file:
        import os
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        output_file = os.path.join(cur_dir, 'policy_model.zip')

    def save_model():
        print("Saving model...")
        try:
            model.save(output_file)
        except Exception as e:
            print("Error saving model. Cancelling training.")
            exit_gym()
            raise e

    # ------------------------------

    from time import time

    end_time = time() + (duration * 60 * 60) # Convert hours to seconds

    from stable_baselines3.common.callbacks import BaseCallback

    class SaveCallback(BaseCallback):
        def __init__(self, verbose: int = 0, freq: int = 100_000):
            super().__init__(verbose)

            self.freq = freq

        def _on_step(self) -> bool:
            if self.n_calls % self.freq == 0:
                save_model()

            return True

    try:
        # Train our agent!
        print("Training...")
        # while time() < end_time:
        
        model.learn(total_timesteps=int(1_000_000_000), log_interval=5, progress_bar=True, reset_num_timesteps=False, callback=SaveCallback())
    except KeyboardInterrupt:
        save_model()
        exit_gym()

if __name__ == "__main__":
    main()