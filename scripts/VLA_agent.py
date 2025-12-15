# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to the VLA model checkpoint.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

import Galaxea_Lab_External.tasks

from Galaxea_Lab_External.VLA.ACT.policy_wrapper import ACTPolicyWrapper, DiffusionPolicyWrapper, BCPolicyWrapper, DataReplayPolicyWrapper


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # Load ACT policy
    # checkpoint_path = '/scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/VLA/ACT/act/ckpt/policy_best.ckpt'
    if args_cli.checkpoint is None:
        print("No checkpoint path provided")
        exit()
    else:
        checkpoint_path = args_cli.checkpoint
    temporal_agg = True
    policy = ACTPolicyWrapper(checkpoint_path, temporal_agg=temporal_agg)

    # Test policy
    # qpos = torch.rand(1, 14, device=args_cli.device)
    # images = torch.rand(1, 3, 3, 240, 320, device=args_cli.device)
    # action = policy.predict(qpos, images)
    # print(f"action: {action}")
    # exit()

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # sample_every_n_steps = max(int(sample_period / env.step_dt), 1)
    print("env type: ", type(env))

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            obs, reward, terminated, truncated, info = env.step(actions)
            # print(f"obs: {obs}")
            left_arm_joint_pos = obs['policy']['left_arm_joint_pos']
            right_arm_joint_pos = obs['policy']['right_arm_joint_pos']
            left_gripper_joint_pos = obs['policy']['left_gripper_joint_pos']
            right_gripper_joint_pos = obs['policy']['right_gripper_joint_pos']
            qpos = torch.cat([left_arm_joint_pos, right_arm_joint_pos, left_gripper_joint_pos.unsqueeze(0), right_gripper_joint_pos.unsqueeze(0)], dim=-1)

            head_rgb = obs['policy']['head_rgb'].unsqueeze(0).permute(0, 1, 4, 2, 3)
            left_hand_rgb = obs['policy']['left_hand_rgb'].unsqueeze(0).permute(0, 1, 4, 2, 3)
            right_hand_rgb = obs['policy']['right_hand_rgb'].unsqueeze(0).permute(0, 1, 4, 2, 3)
            # print(f"left_hand_rgb shape: {left_hand_rgb.shape}")
            # print(f"right_hand_rgb shape: {right_hand_rgb.shape}")
            # print(f"head_rgb shape: {head_rgb.shape}")

            images = torch.cat([head_rgb, left_hand_rgb, right_hand_rgb], dim=1)
            # Change dtype of images to float32
            images = images.to(torch.float32)

            # Print shape of qpos and images
            print(f"qpos shape: {qpos.shape}")
            print(f"images shape: {images.shape}")
            # exit()
            
            action = policy.predict(qpos, images)
            print(f"action: {action}")
            # exit()

            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            env.step(actions)
            if terminated or truncated:
                env.reset()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
