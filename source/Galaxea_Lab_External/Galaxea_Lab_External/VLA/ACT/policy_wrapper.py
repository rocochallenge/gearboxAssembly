#!/usr/bin/env python3
"""
Standard Policy Wrapper Interface for Competition

This module provides a standardized interface for deploying different policy types
in the Isaac Lab environment. All competition participants should implement this
interface for their policies.

Author: Competition Organizers
Date: 2025-12-10
"""

import torch
import numpy as np
import h5py
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import sys
import os


class PolicyWrapper(ABC):
    """
    Abstract base class for policy wrappers.
    
    All policies must implement this interface to be compatible with the
    standardized deployment and evaluation pipeline.
    
    Key Features:
    - Unified predict() interface
    - Optional control frequency specification
    - Automatic device management
    - Clean separation from environment code
    """
    
    @abstractmethod
    def predict(self, qpos: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """
        Predict actions based on current state and images.
        
        Args:
            qpos: Joint positions, shape (batch_size, state_dim)
                  For dual-arm robot: state_dim = 14 (7 joints per arm)
            images: Camera images, shape (batch_size, num_cameras, C, H, W)
                    Default: (1, 3, 3, 240, 320) for 3 RGB cameras
        
        Returns:
            actions: Predicted actions, shape (batch_size, action_dim)
                    For dual-arm: action_dim = 14
        
        Note:
            - All tensors should be on the same device (cuda/cpu)
            - Images are expected to be normalized to [0, 1]
            - Implementation should handle batch processing
        """
        pass
    
    @property
    def required_control_frequency(self) -> Optional[float]:
        """
        Specify the required control frequency in Hz.
        
        Returns:
            float: Required frequency in Hz (e.g., 50.0 for ACT)
            None: Use environment default (20 Hz, decimation=5)
        
        The deployment script will automatically adjust environment
        decimation to match this frequency.
        """
        return None
    
    @property
    def camera_names(self) -> list:
        """
        Specify required camera names in order.
        
        Returns:
            List of camera names, e.g., ['head_rgb', 'left_hand_rgb', 'right_hand_rgb']
        """
        return ['head_rgb', 'left_hand_rgb', 'right_hand_rgb']
    
    @property
    def device(self) -> torch.device:
        """Get the device (cuda/cpu) used by the policy."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ACTPolicyWrapper(PolicyWrapper):
    """
    Wrapper for ACT (Action Chunking with Transformers) policy.
    
    This wrapper demonstrates how to integrate ACT with the standardized interface.
    It handles:
    - ACT model loading with modified argparse
    - Temporal aggregation for smooth action execution
    - Proper observation format conversion
    """
    
    def __init__(self, checkpoint_path: str, temporal_agg: bool = True):
        """
        Initialize ACT policy wrapper.
        
        Args:
            checkpoint_path: Path to ACT checkpoint file (.ckpt)
            temporal_agg: Whether to use temporal aggregation (recommended)
        """
        self.checkpoint_path = checkpoint_path
        self.temporal_agg = temporal_agg
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Temporal aggregation state
        self.all_time_actions = None
        self.timestep = 0
        self.num_queries = 100  # ACT chunk size
        
        # Normalization statistics (will be loaded from dataset_stats.pkl)
        self.action_mean = None
        self.action_std = None
        
        # Delayed import to avoid argparse conflicts
        self._load_act_policy()
        
    def _load_act_policy(self):
        """Load ACT policy with sys.argv isolation."""
        # Save original sys.argv
        original_argv = sys.argv.copy()
        
        try:
            # Clear sys.argv to prevent argparse conflicts
            sys.argv = ['policy_wrapper.py']
            
            # Add ACT to path
            act_dir = os.path.join(os.path.dirname(__file__), '..', 'act')
            if act_dir not in sys.path:
                sys.path.insert(0, act_dir)
                sys.path.insert(0, os.path.join(act_dir, 'detr'))
            
            # Import ACT modules
            from .act.policy import ACTPolicy
            from .act.constants import DT
            
            self.act_dt = DT  # Should be 0.02s (50Hz)
            
            # ACT configuration matching training
            policy_config = {
                'num_queries': 100,
                'kl_weight': 10,
                'hidden_dim': 512,
                'dim_feedforward': 3200,
                'lr_backbone': 1e-5,
                'backbone': 'resnet18',
                'enc_layers': 4,
                'dec_layers': 7,
                'nheads': 8,
                'camera_names': ['head_rgb', 'left_hand_rgb', 'right_hand_rgb'],
            }
            
            # Create and load policy
            self.policy = ACTPolicy(policy_config)
            ckpt = torch.load(self.checkpoint_path, map_location=self._device)
            self.policy.load_state_dict(ckpt)
            self.policy.to(self._device)
            self.policy.eval()
            
            # Load normalization statistics
            import pickle
            stats_path = os.path.join(os.path.dirname(self.checkpoint_path), 'dataset_stats.pkl')
            if os.path.exists(stats_path):
                with open(stats_path, 'rb') as f:
                    stats = pickle.load(f)
                self.action_mean = torch.from_numpy(stats['action_mean']).float().to(self._device)
                self.action_std = torch.from_numpy(stats['action_std']).float().to(self._device)
                print(f"✓ Loaded normalization stats from {stats_path}")
                print(f"  - Action mean range: [{self.action_mean.min():.3f}, {self.action_mean.max():.3f}]")
                print(f"  - Action std range: [{self.action_std.min():.3f}, {self.action_std.max():.3f}]")
            else:
                print(f"⚠ Warning: No dataset_stats.pkl found at {stats_path}")
                print(f"  Actions will NOT be denormalized - this may cause issues!")
            
            print(f"✓ ACT policy loaded successfully")
            print(f"  - Checkpoint: {self.checkpoint_path}")
            print(f"  - Parameters: {sum(p.numel() for p in self.policy.parameters()) / 1e6:.2f}M")
            print(f"  - Chunk size: {self.num_queries}")
            print(f"  - Temporal aggregation: {self.temporal_agg}")
            
        finally:
            # Restore original sys.argv
            sys.argv = original_argv
    
    @property
    def required_control_frequency(self) -> float:
        """ACT requires 50 Hz control frequency."""
        return 50.0
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    def predict(self, qpos: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """
        Predict action using ACT policy.
        
        Args:
            qpos: Joint positions (batch_size, 14)
            images: Camera images (batch_size, 3, 3, 240, 320)
        
        Returns:
            actions: Predicted action (batch_size, 14) - DENORMALIZED to real joint positions
        """
        with torch.no_grad():
            # Get action chunk from policy (normalized outputs)
            all_actions = self.policy(qpos, images)  # (batch_size, num_queries, 14)
            
            if self.temporal_agg:
                # Use temporal aggregation for smooth execution
                action = self._temporal_aggregation(all_actions)
            else:
                # Just use first action from chunk
                action = all_actions[:, 0, :]
            
            # Denormalize action to real joint positions
            if self.action_mean is not None and self.action_std is not None:
                action = action * self.action_std + self.action_mean
            
            self.timestep += 1
            
        return action
    
    def _temporal_aggregation(self, all_actions: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal aggregation with exponential weighting.
        
        Args:
            all_actions: Action chunk (batch_size, num_queries, action_dim)
        
        Returns:
            Aggregated action (batch_size, action_dim)
        """
        batch_size = all_actions.shape[0]
        action_dim = all_actions.shape[2]
        
        # Initialize buffer if needed
        if self.all_time_actions is None:
            max_timesteps = 3000
            self.all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + self.num_queries, action_dim],
                device=self._device
            )
        
        # Store current action chunk
        self.all_time_actions[self.timestep, self.timestep:self.timestep + self.num_queries] = all_actions[0]
        
        # Exponential weighting: more recent predictions have higher weight
        actions_for_curr_step = self.all_time_actions[:self.timestep + 1, self.timestep]
        weights = torch.exp(-0.1 * torch.arange(self.timestep + 1, device=self._device).flip(0))
        weights = weights / weights.sum()
        
        # Weighted average
        action = (actions_for_curr_step.T @ weights).unsqueeze(0)
        
        return action
    
    def reset(self):
        """Reset temporal aggregation buffers for new episode."""
        self.all_time_actions = None
        self.timestep = 0


# Example: Placeholder for other policy types
class DiffusionPolicyWrapper(PolicyWrapper):
    """Placeholder for Diffusion Policy wrapper."""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        raise NotImplementedError("Diffusion Policy wrapper not yet implemented")
    
    def predict(self, qpos: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BCPolicyWrapper(PolicyWrapper):
    """Placeholder for Behavior Cloning policy wrapper."""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        raise NotImplementedError("BC Policy wrapper not yet implemented")
    
    def predict(self, qpos: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


if __name__ == "__main__":
    """Test ACT policy wrapper loading and inference."""
    
    print("Testing ACT Policy Wrapper...")
    print("=" * 60)
    
    # Test checkpoint path
    checkpoint_path = "act/ckpt/policy_best.ckpt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please provide correct checkpoint path")
        exit(1)
    
    # Load policy
    print("\nLoading ACT policy...")
    wrapper = ACTPolicyWrapper(checkpoint_path, temporal_agg=True)
    
    # Test inference
    print("\nTesting inference...")
    batch_size = 1
    state_dim = 14
    num_cameras = 3
    
    # Create dummy inputs
    qpos = torch.randn(batch_size, state_dim).to(wrapper.device)
    images = torch.rand(batch_size, num_cameras, 3, 240, 320).to(wrapper.device)
    
    print(f"Input shapes:")
    print(f"  - qpos: {qpos.shape}")
    print(f"  - images: {images.shape}")
    
    # Predict
    action = wrapper.predict(qpos, images)
    
    print(f"\nOutput:")
    print(f"  - action shape: {action.shape}")
    print(f"  - action range: [{action.min().item():.3f}, {action.max().item():.3f}]")
    
    # Test properties
    print(f"\nPolicy properties:")
    print(f"  - Required frequency: {wrapper.required_control_frequency} Hz")
    print(f"  - Camera names: {wrapper.camera_names}")
    print(f"  - Device: {wrapper.device}")
    
    print("\n✓ All tests passed!")


class DataReplayPolicyWrapper(PolicyWrapper):
    """
    Data Replay Policy - replays actions from recorded demonstration data.
    
    This policy wrapper reads actions from an HDF5 file and replays them step by step.
    Useful for:
    1. Debugging deployment pipeline (excludes model performance issues)
    2. Verifying environment behavior matches data collection
    3. Testing action application without model inference
    
    The replay uses a simple step counter to index into the action array.
    """
    
    def __init__(
        self,
        data_path: str,
        device: str = "cuda",
        camera_names: Optional[list] = None,
    ):
        """
        Initialize data replay policy.
        
        Args:
            data_path: Path to HDF5 file containing recorded actions
            device: Device to use for tensors
            camera_names: Camera names (for compatibility with interface)
        """
        self.device = device
        self._camera_names = camera_names or ['head_rgb', 'left_hand_rgb', 'right_hand_rgb']
        
        print(f"\n{'='*80}")
        print("Initializing Data Replay Policy")
        print(f"{'='*80}")
        print(f"Data path: {data_path}")
        
        # Load action data from HDF5
        with h5py.File(data_path, 'r') as f:
            # Read all action components
            left_arm = f['/actions/left_arm_action'][:]  # (N, 6)
            right_arm = f['/actions/right_arm_action'][:]  # (N, 6)
            left_gripper = f['/actions/left_gripper_action'][:]  # (N,)
            right_gripper = f['/actions/right_gripper_action'][:]  # (N,)
            
            # Combine into 14-dim actions: [left_arm(6), right_arm(6), left_gripper(1), right_gripper(1)]
            self.actions = np.concatenate([
                left_arm,  # (N, 6)
                right_arm,  # (N, 6)
                left_gripper[:, np.newaxis],  # (N, 1)
                right_gripper[:, np.newaxis],  # (N, 1)
            ], axis=1)  # (N, 14)
            
            # Also load observations for reference
            self.qpos_data = f['/observations/left_arm_joint_pos'][:] if '/observations/left_arm_joint_pos' in f else None
            
            print(f"\nLoaded data:")
            print(f"  - Total timesteps: {len(self.actions)}")
            print(f"  - Action shape: {self.actions.shape}")
            print(f"  - Action range: [{self.actions.min():.3f}, {self.actions.max():.3f}]")
            
            # Statistics
            print(f"\nAction statistics per joint:")
            for i in range(14):
                joint_name = self._get_joint_name(i)
                mean_val = self.actions[:, i].mean()
                std_val = self.actions[:, i].std()
                min_val = self.actions[:, i].min()
                max_val = self.actions[:, i].max()
                print(f"  {joint_name:20s}: mean={mean_val:6.3f}, std={std_val:5.3f}, range=[{min_val:6.3f}, {max_val:6.3f}]")
        
        # Convert to torch tensor
        self.actions = torch.from_numpy(self.actions).float().to(self.device)
        
        # Step counter for indexing into action array
        self.step_idx = 0
        self.max_steps = len(self.actions)
        
        print(f"\n{'='*80}")
        print("✓ Data Replay Policy initialized successfully")
        print(f"{'='*80}\n")
    
    def _get_joint_name(self, idx: int) -> str:
        """Get human-readable joint name for index."""
        names = [
            "left_arm_joint_1",
            "left_arm_joint_2", 
            "left_arm_joint_3",
            "left_arm_joint_4",
            "left_arm_joint_5",
            "left_arm_joint_6",
            "right_arm_joint_1",
            "right_arm_joint_2",
            "right_arm_joint_3", 
            "right_arm_joint_4",
            "right_arm_joint_5",
            "right_arm_joint_6",
            "left_gripper",
            "right_gripper",
        ]
        return names[idx]
    
    def predict(self, qpos: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """
        Return the next action from recorded data.
        
        Args:
            qpos: Current joint positions (not used in replay, but kept for interface compatibility)
            images: Current camera images (not used in replay)
        
        Returns:
            action: Next action from recorded trajectory, shape (1, 14)
        """
        # Get current action from pre-loaded data
        if self.step_idx >= self.max_steps:
            print(f"[WARNING] Replay finished! Reached end of data at step {self.step_idx}/{self.max_steps}")
            print(f"          Looping back to beginning...")
            self.step_idx = 0
        
        action = self.actions[self.step_idx].unsqueeze(0)  # (1, 14)
        
        # Debug output every 50 steps
        if self.step_idx % 50 == 0:
            print(f"\n[Replay Step {self.step_idx}/{self.max_steps}]")
            print(f"  Left arm action:  {action[0, :6].cpu().numpy()}")
            print(f"  Right arm action: {action[0, 6:12].cpu().numpy()}")
            print(f"  Grippers: L={action[0, 12].item():.3f}, R={action[0, 13].item():.3f}")
        
        self.step_idx += 1
        return action
    
    @property
    def camera_names(self) -> list:
        """Return camera names."""
        return self._camera_names
    
    def reset(self):
        """Reset replay to beginning of trajectory."""
        print(f"\n[Data Replay] Resetting to step 0")
        self.step_idx = 0
