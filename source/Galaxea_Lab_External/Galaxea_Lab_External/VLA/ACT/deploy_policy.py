#!/usr/bin/env python3
"""
Standardized Policy Deployment Script for Competition

This script provides a unified interface for deploying and evaluating different
policy types in the Isaac Lab environment. It automatically handles:
- Control frequency adaptation based on policy requirements
- Camera observation collection and formatting
- Episode execution and statistics
- Video recording (optional)

Usage:
    python deploy_policy.py --policy_type act --checkpoint act/ckpt/policy_best.ckpt --num_episodes 10

Author: Competition Organizers
Date: 2025-12-10
"""

import argparse
import numpy as np
import torch
import time
import os
import sys
from pathlib import Path

# Isaac Lab imports (must be before policy imports)
from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Standardized Policy Deployment for Competition")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--task", type=str, default="Template-Galaxea-Lab-External-Direct-v0", help="Task name")
parser.add_argument("--policy_type", type=str, required=True, choices=['act', 'diffusion', 'bc', 'replay'], 
                    help="Policy type (act/diffusion/bc/replay)")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint or data file (for replay)")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes")
parser.add_argument("--save_video", action="store_true", help="Save episode videos")
parser.add_argument("--temporal_agg", action="store_true", default=True, help="Use temporal aggregation (for ACT)")

# Add AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import Isaac Lab modules after launching
import gymnasium as gym
import torch

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

import Galaxea_Lab_External.tasks

# Import policy wrapper
sys.path.insert(0, str(Path(__file__).parent))
from policy_wrapper import ACTPolicyWrapper, DiffusionPolicyWrapper, BCPolicyWrapper, DataReplayPolicyWrapper


def load_replay_actions(data_path: str):
    """
    Load actions from HDF5 file for replay.
    
    Args:
        data_path: Path to HDF5 file
    
    Returns:
        actions: numpy array of shape (N, 14)
    """
    import h5py
    print(f"\n{'='*80}")
    print("Loading replay data...")
    print(f"Data path: {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        # Read all action components
        left_arm = f['/actions/left_arm_action'][:]  # (N, 6)
        right_arm = f['/actions/right_arm_action'][:]  # (N, 6)
        left_gripper = f['/actions/left_gripper_action'][:]  # (N,)
        right_gripper = f['/actions/right_gripper_action'][:]  # (N,)
        
        # Combine into 14-dim actions: [left_arm(6), right_arm(6), left_gripper(1), right_gripper(1)]
        actions = np.concatenate([
            left_arm,  # (N, 6)
            right_arm,  # (N, 6)
            left_gripper[:, np.newaxis],  # (N, 1)
            right_gripper[:, np.newaxis],  # (N, 1)
        ], axis=1)  # (N, 14)
        
        print(f"✓ Loaded {len(actions)} actions")
        print(f"  Action shape: {actions.shape}")
        print(f"  Action range: [{actions.min():.3f}, {actions.max():.3f}]")
        print(f"{'='*80}\n")
        
        return actions


def load_policy(policy_type: str, checkpoint_path: str, **kwargs):
    """
    Load policy based on type.
    
    Args:
        policy_type: Type of policy ('act', 'diffusion', 'bc', 'replay')
        checkpoint_path: Path to checkpoint file (or data file for replay)
        **kwargs: Additional arguments for policy
    
    Returns:
        PolicyWrapper instance
    """
    if policy_type == 'act':
        return ACTPolicyWrapper(checkpoint_path, temporal_agg=kwargs.get('temporal_agg', True))
    elif policy_type == 'diffusion':
        return DiffusionPolicyWrapper(checkpoint_path)
    elif policy_type == 'bc':
        return BCPolicyWrapper(checkpoint_path)
    elif policy_type == 'replay':
        return DataReplayPolicyWrapper(checkpoint_path)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def get_observations(env, policy_wrapper):
    """
    Get observations from environment in policy format.
    
    Args:
        env: Isaac Lab environment
        policy_wrapper: Policy wrapper instance
    
    Returns:
        qpos: Joint positions (batch_size, state_dim)
        images: Camera images (batch_size, num_cameras, 3, H, W)
    """
    device = policy_wrapper.device
    
    # Get joint positions from environment
    # Use environment's joint indices to get the correct 14 DoF
    # (left_arm: 6, right_arm: 6, left_gripper: 1, right_gripper: 1)
    env_unwrapped = env.unwrapped
    robot = env_unwrapped.scene["robot"]
    
    # Concatenate joint positions in correct order
    left_arm_pos = robot.data.joint_pos[:, env_unwrapped._left_arm_joint_idx]
    right_arm_pos = robot.data.joint_pos[:, env_unwrapped._right_arm_joint_idx]
    left_gripper_pos = robot.data.joint_pos[:, env_unwrapped._left_gripper_dof_idx]
    right_gripper_pos = robot.data.joint_pos[:, env_unwrapped._right_gripper_dof_idx]
    
    # Concatenate: [left_arm(6), right_arm(6), left_gripper(1), right_gripper(1)]
    joint_pos = torch.cat([
        left_arm_pos,
        right_arm_pos,
        left_gripper_pos,
        right_gripper_pos
    ], dim=-1)  # (num_envs, 14)
    
    qpos = joint_pos.clone().to(device)
    
    # Get camera images
    camera_images = []
    camera_name_mapping = {
        'head_rgb': 'head_camera',
        'left_hand_rgb': 'left_hand_camera',
        'right_hand_rgb': 'right_hand_camera'
    }
    
    for cam_name in policy_wrapper.camera_names:
        sensor_name = camera_name_mapping.get(cam_name)
        if sensor_name is None:
            raise ValueError(f"Unknown camera name: {cam_name}")
        
        # Get RGB image from sensor (shape: num_envs, H, W, C)
        rgb_data = env.unwrapped.scene[sensor_name].data.output["rgb"]  # (num_envs, 240, 320, 3)
        
        # Convert to tensor and rearrange to (num_envs, C, H, W)
        rgb_tensor = rgb_data.clone()  # (num_envs, 240, 320, 3)
        rgb_tensor = rgb_tensor.permute(0, 3, 1, 2)  # (num_envs, 3, 240, 320)
        
        # Normalize to [0, 1] if needed
        if rgb_tensor.max() > 1.0:
            rgb_tensor = rgb_tensor / 255.0
        
        camera_images.append(rgb_tensor)
    
    # Stack all cameras: (num_envs, num_cameras, 3, H, W)
    images = torch.stack(camera_images, dim=1).to(device)
    
    return qpos, images


def run_episode(env, policy_wrapper, episode_idx: int, save_video: bool = False):
    """
    Run one episode with the policy.
    
    Args:
        env: Isaac Lab environment
        policy_wrapper: Policy wrapper instance
        episode_idx: Episode index
        save_video: Whether to save video
    
    Returns:
        episode_reward: Total episode reward
        episode_length: Episode length in steps
        success: Whether episode succeeded
    """
    # Reset policy if it has reset method
    if hasattr(policy_wrapper, 'reset'):
        policy_wrapper.reset()
    
    # Reset environment
    obs, _ = env.reset()
    
    episode_reward = 0.0
    episode_length = 0
    success = False
    
    print(f"\n{'='*60}")
    print(f"Episode {episode_idx + 1}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    while True:
        # Get observations in policy format
        qpos, images = get_observations(env, policy_wrapper)
        
        # Predict action
        with torch.no_grad():
            action = policy_wrapper.predict(qpos, images)
        
        # Convert action to environment format (should be on env device)
        action_env = action.to(env.unwrapped.device)
        
        # Apply action
        obs, reward, terminated, truncated, info = env.step(action_env)
        
        # Handle reward (could be int or tensor)
        episode_reward += reward.item() if hasattr(reward, 'item') else reward
        episode_length += 1
        
        # Check if episode is done
        done = terminated.item() or truncated.item()
        if done:
            success = info.get('success', False)
            break
    
    elapsed = time.time() - start_time
    
    # Print episode summary
    print(f"\n{'-'*60}")
    print(f"Episode {episode_idx + 1} Summary:")
    print(f"  - Duration: {elapsed:.1f}s")
    print(f"  - Steps: {episode_length}")
    print(f"  - Total Reward: {episode_reward:.2f}")
    print(f"  - Success: {success}")
    print(f"  - FPS: {episode_length/elapsed:.1f}")
    print(f"{'-'*60}")
    
    return episode_reward, episode_length, success


def run_replay_episode(env, replay_actions, episode_idx):
    """
    Run a single episode using replay actions from recorded data.
    
    Args:
        env: Isaac Lab environment
        replay_actions: numpy array of actions, shape (N, 14)
        episode_idx: Episode index
    
    Returns:
        episode_reward: Total reward
        episode_length: Episode length
        success: Success flag
    """
    print(f"\n{'='*80}")
    print(f"Episode {episode_idx + 1} - Replaying recorded actions")
    print(f"{'='*80}")
    
    # Reset environment
    obs, _ = env.reset()
    
    episode_reward = 0
    episode_length = 0
    success = False
    
    start_time = time.time()
    
    # Run episode with replay actions
    for step_idx in range(len(replay_actions)):
        # Get action from replay data
        action_np = replay_actions[step_idx]  # (14,)
        action = torch.from_numpy(action_np).float().unsqueeze(0).to(env.unwrapped.device)  # (1, 14)
        
        # Print every 50 steps
        if step_idx % 50 == 0:
            print(f"\n[Replay Step {step_idx}/{len(replay_actions)}]")
            print(f"  Left arm:  {action_np[:6]}")
            print(f"  Right arm: {action_np[6:12]}")
            print(f"  Grippers:  L={action_np[12]:.3f}, R={action_np[13]:.3f}")
        
        # Apply action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Handle reward
        episode_reward += reward.item() if hasattr(reward, 'item') else reward
        episode_length += 1
        
        # Check if episode is done
        done = terminated.item() or truncated.item()
        if done:
            success = info.get('success', False)
            break
    
    elapsed = time.time() - start_time
    
    # Print episode summary
    print(f"\n{'-'*60}")
    print(f"Replay Episode {episode_idx + 1} Summary:")
    print(f"  - Duration: {elapsed:.1f}s")
    print(f"  - Steps: {episode_length}/{len(replay_actions)}")
    print(f"  - Total Reward: {episode_reward:.2f}")
    print(f"  - Success: {success}")
    print(f"  - FPS: {episode_length/elapsed:.1f}")
    print(f"{'-'*60}")
    
    return episode_reward, episode_length, success


def main():
    """Main deployment function"""
    
    print("\n" + "="*60)
    print("STANDARDIZED POLICY DEPLOYMENT")
    print("="*60)
    print(f"Policy type: {args_cli.policy_type}")
    print(f"Checkpoint: {args_cli.checkpoint}")
    print(f"Task: {args_cli.task}")
    print(f"Num episodes: {args_cli.num_episodes}")
    print(f"Num environments: {args_cli.num_envs}")
    print("="*60)
    
    # Special handling for replay mode
    if args_cli.policy_type == 'replay':
        # Load replay actions
        replay_actions = load_replay_actions(args_cli.checkpoint)
        policy_wrapper = None
    else:
        # Load policy
        print("\nLoading policy...")
        policy_wrapper = load_policy(
            args_cli.policy_type,
            args_cli.checkpoint,
            temporal_agg=args_cli.temporal_agg
        )
        replay_actions = None
    
    # Create environment with adapted configuration
    print(f"\nInitializing environment: {args_cli.task}")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = parse_env_cfg(args_cli.task, device=device_str, num_envs=args_cli.num_envs)
    
    # Adapt control frequency if policy requires specific frequency
    if policy_wrapper is not None:
        required_freq = policy_wrapper.required_control_frequency
        if required_freq is not None:
            # Calculate required decimation: decimation = 1 / (sim_dt * frequency)
            required_decimation = int(1.0 / (env_cfg.sim.dt * required_freq))
            original_decimation = env_cfg.decimation
            env_cfg.decimation = required_decimation
            
            print(f"\nControl frequency adaptation:")
            print(f"  - Policy requires: {required_freq} Hz")
            print(f"  - Original decimation: {original_decimation} (dt={env_cfg.sim.dt * original_decimation:.3f}s)")
            print(f"  - Adapted decimation: {required_decimation} (dt={env_cfg.sim.dt * required_decimation:.3f}s)")
            print(f"  - Actual frequency: {1.0 / (env_cfg.sim.dt * required_decimation):.1f} Hz")
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"✓ Environment created with {args_cli.num_envs} instance(s)")
    
    # Verify actual control frequency
    actual_dt = env.unwrapped.cfg.sim_dt * env.unwrapped.cfg.decimation
    actual_freq = 1.0 / actual_dt
    print(f"\nFinal environment configuration:")
    print(f"  - Control frequency: {actual_freq:.1f} Hz (DT={actual_dt:.3f}s)")
    print(f"  - Simulation dt: {env.unwrapped.cfg.sim_dt}s")
    print(f"  - Decimation: {env.unwrapped.cfg.decimation}")
    
    # Run episodes
    print(f"\nRunning {args_cli.num_episodes} evaluation episodes...")
    print("="*60)
    
    episode_rewards = []
    episode_lengths = []
    successes = []
    
    try:
        for episode_idx in range(args_cli.num_episodes):
            if policy_wrapper is not None:
                # Normal policy mode
                reward, length, success = run_episode(
                    env, 
                    policy_wrapper,
                    episode_idx,
                    save_video=args_cli.save_video
                )
            else:
                # Replay mode
                reward, length, success = run_replay_episode(
                    env,
                    replay_actions,
                    episode_idx
                )
            
            episode_rewards.append(reward)
            episode_lengths.append(length)
            successes.append(success)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    # Print final statistics
    if len(episode_rewards) > 0:
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Episodes completed: {len(episode_rewards)}")
        print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"Success rate: {np.mean(successes)*100:.1f}% ({sum(successes)}/{len(successes)})")
        print(f"Best reward: {np.max(episode_rewards):.2f}")
        print(f"Worst reward: {np.min(episode_rewards):.2f}")
        print("="*60)
    
    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
