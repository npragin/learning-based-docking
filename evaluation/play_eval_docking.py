import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import pandas as pd
import numpy as np

from rsl_rl.runners import OnPolicyRunner

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=1, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create dir to save logs into
    save_path = os.path.join("source", "results", "rsl_rl", agent_cfg.experiment_name, agent_cfg.load_run, agent_cfg.load_checkpoint[:-3] + "_play")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print(f"[INFO]: Saving results into: {save_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(ppo_runner.alg.actor_critic, path=export_model_dir, filename="policy.onnx")

    # reset environment
    obs, _ = env.get_observations()

    episode_ix = 1
    max_episodes = 25
    metrics = {
        "success": np.array([False for i in range(max_episodes)]),
        "distance_travelled": np.array([None for i in range(max_episodes)]),
        "dt": np.array([None for i in range(max_episodes)]),
        "jerk": np.array([None for i in range(max_episodes)]),
        "action_cost": np.array([None for i in range(max_episodes)])
    }

    successful_dock = False

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # env stepping
            obs, _, dones, _ = env.step(actions)

            original_env = env.unwrapped
            eval_metrics = original_env._evaluation_metrics[0]

            if eval_metrics[0] and not successful_dock:
                successful_dock = True

                distance_travelled = eval_metrics[2]
                dt = eval_metrics[3]
                jerk = eval_metrics[4]
                action_cost = eval_metrics[5]

                metrics["success"][episode_ix - 1] = True
                metrics["distance_travelled"][episode_ix - 1] = distance_travelled.item()
                metrics["dt"][episode_ix - 1] = dt.item()
                metrics["jerk"][episode_ix - 1] = jerk.item()
                metrics["action_cost"][episode_ix - 1] = action_cost.item()

            if dones[0]:
                if episode_ix == max_episodes:
                    break

                successful_dock = False
                episode_ix = episode_ix + 1

    # close the simulator
    env.close()

    success_rate = np.sum(metrics["success"]) / max_episodes
    distance_travelled_sum = 0
    dt_sum = 0
    jerk_sum = 0
    action_cost_sum = 0
    num_filled = 0
    for i in range(max_episodes):
        if metrics["success"][i] == True:
            num_filled = num_filled + 1
            distance_travelled_sum = distance_travelled_sum + metrics["distance_travelled"][i]
            dt_sum = dt_sum + metrics["dt"][i]
            jerk_sum = jerk_sum + metrics["jerk"][i]
            action_cost = action_cost + metrics["action_cost"][i]

    if num_filled == 0:
        num_filled = 1

    print(f"success rate: {success_rate * 100.0}")
    print(f"avg distance travelled: {distance_travelled_sum / num_filled}")
    print(f"avg dt: {dt_sum / num_filled}")
    print(f"avg jerk: {jerk_sum / num_filled}")
    print(f"avg action cost: {action_cost / num_filled}")

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()