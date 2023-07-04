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

    from policy import Policy
    if input_file:
        print("Loading model from file...")
        try:
            policy: Policy = torch.jit.load(input_file).to(device)
        except Exception as e:
            print("Error loading model from file. Aborting.")
            raise e

    # ------------------------------

    import rlgym
    from stable_baselines3 import PPO
    from rlgym_tools.sb3_utils import SB3SingleInstanceEnv, SB3MultipleInstanceEnv

    print("Creating gym environment...")

    from rlgym.utils.reward_functions.common_rewards import EventReward, LiuDistancePlayerToBallReward, LiuDistanceBallToGoalReward, VelocityBallToGoalReward, VelocityPlayerToBallReward
    from rlgym.utils.reward_functions import CombinedReward
    from rlgym.envs import Match

    from rlgym.utils.obs_builders import DefaultObs
    from rlgym.utils.action_parsers import DefaultAction
    from rlgym.utils.state_setters import DefaultState
    from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition

    def get_match():
        return Match(
            reward_function=CombinedReward(reward_functions=[
                EventReward(goal=100, concede=-100, touch=1),
                LiuDistancePlayerToBallReward(),
                LiuDistanceBallToGoalReward(),
                VelocityPlayerToBallReward(),
                VelocityBallToGoalReward(),
            ], reward_weights=[
                1,
                10,
                10,
                1,
                1,
            ]),
            terminal_conditions=[TimeoutCondition(1000), GoalScoredCondition()],
            obs_builder=DefaultObs(),
            action_parser=DefaultAction(),
            state_setter=DefaultState(),

            spawn_opponents=True,
            game_speed=1000,
        )

    # # Create gym environment
    # gym_env = rlgym.make(
    #     use_injector=True,
    #     spawn_opponents=True,
    #     reward_fn=EventReward(goal=1, concede=-1),
    # )

    # # Wrap the gym environment in a stable_baselines3 environment for training
    # env = SB3SingleInstanceEnv(gym_env)

    from rlgym.gamelaunch.launch import LaunchPreference

    env = SB3MultipleInstanceEnv(
        match_func_or_matches=get_match, 
        num_instances=5, 
        wait_time=60,
        launch_preference=LaunchPreference.EPIC_LOGIN_TRICK
    )

    def exit_gym():
        env.close()
        # gym_env.close()
        print("Gym environment closed.")

    # Initialize PPO from stable_baselines3
    model = PPO("MlpPolicy", env=env, verbose=1, device=device, learning_rate=0.0003)

    # ------------------------------

    # Inject the model into the PPO model (if specified)
    if input_file:
        print("Injecting model into PPO...")
        try:
            model.policy.mlp_extractor = policy.extractor
            model.policy.action_net = policy.action_net
            model.policy.value_net = policy.value_net
        except Exception as e:
            print("Error injecting model. Cancelling training.")
            exit_gym()
            raise e
        
    # ------------------------------

    # Set output file if not set
    if not output_file:
        import os
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        output_file = os.path.join(cur_dir, 'policy_model.pt')

    # Create a function to save the model,
    # This way we can save the model repeatedly
    # in case of a crash/termination while training
    def save_model():
        print("Saving model...")
        try:
            policy: Policy = Policy( # Convert PPO to Torch model
                model.policy.mlp_extractor, 
                model.policy.action_net, 
                model.policy.value_net
            )

            # I don't really know what scripting does,
            # but it seems to be necessary to save the model
            model_scripted = torch.jit.script(policy)

            model_scripted.save(output_file)
        except Exception as e:
            print("Error saving model. Cancelling training.")
            exit_gym()
            raise e

    # ------------------------------

    from time import time

    end_time = time() + (duration * 60 * 60) # Convert hours to seconds

    try:
        # Train our agent!
        print("Training...")
        while time() < end_time:
            model.learn(total_timesteps=int(5e5), log_interval=5, progress_bar=True)
            save_model()
    except KeyboardInterrupt:
        save_model()
        exit_gym()

if __name__ == "__main__":
    main()