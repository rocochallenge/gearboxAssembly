# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import numpy as np
import time
from datetime import datetime
# from torchvision.utils import save_image
from PIL import Image

from collections.abc import Sequence
import os
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBase, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, euler_xyz_from_quat

from .galaxea_lab_agent_env_cfg import GalaxeaLabAgentEnvCfg

from pxr import Usd, Sdf, UsdPhysics, UsdGeom, Gf
from isaaclab.sim.spawners.materials import physics_materials, physics_materials_cfg
from isaaclab.sim.spawners.materials import spawn_rigid_body_material
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp

import isaacsim.core.utils.torch as torch_utils

from Galaxea_Lab_External.robots import GalaxeaRulePolicy
from isaaclab.sensors import Camera

import h5py

class GalaxeaLabAgentEnv(DirectRLEnv):
    cfg: GalaxeaLabAgentEnvCfg

    def __init__(self, cfg: GalaxeaLabAgentEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        print(f"--------------------------------INIT--------------------------------")

        self._left_arm_joint_idx, _ = self.robot.find_joints(self.cfg.left_arm_joint_dof_name)
        self._right_arm_joint_idx, _ = self.robot.find_joints(self.cfg.right_arm_joint_dof_name)
        self._left_gripper_dof_idx, _ = self.robot.find_joints(self.cfg.left_gripper_dof_name)
        self._right_gripper_dof_idx, _ = self.robot.find_joints(self.cfg.right_gripper_dof_name)

        self._left_arm_action = torch.zeros(self._left_arm_joint_idx, device=self.device)
        self._right_arm_action = torch.zeros(self._right_arm_joint_idx, device=self.device)
        self._left_gripper_action = torch.zeros(1, device=self.device)
        self._right_gripper_action = torch.zeros(1, device=self.device)

        self._torso_joint_idx, _ = self.robot.find_joints(self.cfg.torso_joint_dof_name)

        print(f"_torso_joint_idx: {self._torso_joint_idx}")

        self._torso_joint1_idx, _ = self.robot.find_joints(self.cfg.torso_joint1_dof_name)
        self._torso_joint2_idx, _ = self.robot.find_joints(self.cfg.torso_joint2_dof_name)
        self._torso_joint3_idx, _ = self.robot.find_joints(self.cfg.torso_joint3_dof_name)

        print(f"_left_arm_joint_idx: {self._left_arm_joint_idx}")
        print(f"_right_arm_joint_idx: {self._right_arm_joint_idx}")
        print(f"_left_gripper_dof_idx: {self._left_gripper_dof_idx}")
        print(f"_right_gripper_dof_idx: {self._right_gripper_dof_idx}")

        self._joint_idx = self._left_arm_joint_idx + self._right_arm_joint_idx + self._left_gripper_dof_idx + self._right_gripper_dof_idx

        self.left_arm_joint_pos = self.robot.data.joint_pos[:, self._left_arm_joint_idx]
        self.right_arm_joint_pos = self.robot.data.joint_pos[:, self._right_arm_joint_idx]
        self.left_gripper_joint_pos = self.robot.data.joint_pos[:, self._left_gripper_dof_idx]
        self.right_gripper_joint_pos = self.robot.data.joint_pos[:, self._right_gripper_dof_idx]

        self.left_arm_joint_vel = self.robot.data.joint_vel[:, self._left_arm_joint_idx]
        self.right_arm_joint_vel = self.robot.data.joint_vel[:, self._right_arm_joint_idx]
        self.left_gripper_joint_vel = self.robot.data.joint_vel[:, self._left_gripper_dof_idx]
        self.right_gripper_joint_vel = self.robot.data.joint_vel[:, self._right_gripper_dof_idx]
        
        print(f"left_arm_joint_pos: {self.left_arm_joint_pos}")
        print(f"right_arm_joint_pos: {self.right_arm_joint_pos}")
        print(f"left_gripper_joint_pos: {self.left_gripper_joint_pos}")
        print(f"right_gripper_joint_pos: {self.right_gripper_joint_pos}")

        print(f"left_arm_joint_vel: {self.left_arm_joint_vel}")
        print(f"right_arm_joint_vel: {self.right_arm_joint_vel}")
        print(f"left_gripper_joint_vel: {self.left_gripper_joint_vel}")
        print(f"right_gripper_joint_vel: {self.right_gripper_joint_vel}")

        self.joint_pos = self.robot.data.joint_pos[:, self._joint_idx]

    def _setup_scene(self):

        print(f"--------------------------------SETUP SCENE--------------------------------")

        self.robot = Articulation(self.cfg.robot_cfg)
        
        self.head_camera = Camera(self.cfg.head_camera_cfg)
        self.left_hand_camera = Camera(self.cfg.left_hand_camera_cfg)
        self.right_hand_camera = Camera(self.cfg.right_hand_camera_cfg)

        # self.table = sim_utils.spawn_from_usd("/World/envs/env_.*/Table", self.cfg.table_cfg.spawn,
        #     translation=self.cfg.table_cfg.init_state.pos, 
        #     orientation=self.cfg.table_cfg.init_state.rot)
        self.table = RigidObject(self.cfg.table_cfg)

        self.ring_gear = RigidObject(self.cfg.ring_gear_cfg)
        self.sun_planetary_gear_1 = RigidObject(self.cfg.sun_planetary_gear_1_cfg)
        self.sun_planetary_gear_2 = RigidObject(self.cfg.sun_planetary_gear_2_cfg)
        self.sun_planetary_gear_3 = RigidObject(self.cfg.sun_planetary_gear_3_cfg)
        self.sun_planetary_gear_4 = RigidObject(self.cfg.sun_planetary_gear_4_cfg)
        self.planetary_carrier = RigidObject(self.cfg.planetary_carrier_cfg)
        self.planetary_reducer = RigidObject(self.cfg.planetary_reducer_cfg)

        self.pin_local_positions = [
            torch.tensor([0.0, -0.054, 0.0], device=self.device),      # pin_0
            # torch.tensor([0.0465, 0.0268, 0.0], device=self.device),   # pin_1
            # torch.tensor([-0.0465, 0.0268, 0.0], device=self.device),  # pin_2
            torch.tensor([0.0471, 0.0268, 0.0], device=self.device),   # pin_1
            torch.tensor([-0.0471, 0.0268, 0.0], device=self.device),  # pin_2
        ]


        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.obj_dict = {"ring_gear": self.ring_gear,
                        "planetary_carrier": self.planetary_carrier,
                        "sun_planetary_gear_1": self.sun_planetary_gear_1,
                        "sun_planetary_gear_2": self.sun_planetary_gear_2,
                        "sun_planetary_gear_3": self.sun_planetary_gear_3,
                        "sun_planetary_gear_4": self.sun_planetary_gear_4,
                        "planetary_reducer": self.planetary_reducer}

        # self.head_cam = sim_utils.CameraCfg(
        #     prim_path="/World/envs/env_.*/Robot/left_camera",
        #     resolution=(1280, 720),
        #     fov=60.0,
        #     position=(0.0, 0.0, 0.0),
        #     orientation=(0.0, 0.0, 0.0),
        # )


        self._initialize_scene()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        print(f"--------------------------------PRE PHYSICS STEP at {mdp.observations.current_time_s(self).item()} seconds--------------------------------")
        # self.actions = actions.clone()
        # print(f"_pre_physics_step actions: {self.actions}")

        pass

    def _apply_action(self) -> None:
        start_time = time.time()
        # print(f"Time: {self.rule_policy.count * self.sim.get_physics_dt()}, Apply action")
        current_time_s = mdp.observations.current_time_s(self)
        print(f"Apply action: {current_time_s.item()} seconds")
        # action, joint_ids = self.rule_policy.get_action()
        action = self.env_step_action

        # Apply action to the robot
        self.robot.set_joint_position_target(action, self._joint_idx)

        self.rule_policy.count += 1
        sim_dt = self.sim.get_physics_dt()

        for obj_name, obj in self.obj_dict.items():
            obj.update(sim_dt)

        for cam in [self.head_camera, self.left_hand_camera, self.right_hand_camera]:
            cam.update(dt=sim_dt)

        end_time = time.time()
        # print(f"Apply action time cost: {end_time - start_time} seconds")

    def _get_observations(self) -> dict:
        # print(f"Time: {self.rule_policy.count * self.sim.get_physics_dt()}, Get observations")
        current_time_s = mdp.observations.current_time_s(self)
        print(f"--------------------------------Get observations at {current_time_s.item()} seconds--------------------------------")
        data_type = "rgb"

        head_rgb = self.head_camera.data.output[data_type]
        left_hand_rgb = self.left_hand_camera.data.output[data_type]
        right_hand_rgb = self.right_hand_camera.data.output[data_type]

        data_type = "distance_to_image_plane"
        head_depth = self.head_camera.data.output[data_type]
        left_hand_depth = self.left_hand_camera.data.output[data_type]
        right_hand_depth = self.right_hand_camera.data.output[data_type]

        self.left_arm_joint_pos = self.robot.data.joint_pos[:, self._left_arm_joint_idx]
        self.right_arm_joint_pos = self.robot.data.joint_pos[:, self._right_arm_joint_idx]
        self.left_gripper_joint_pos = self.robot.data.joint_pos[:, self._left_gripper_dof_idx[0]]
        self.right_gripper_joint_pos = self.robot.data.joint_pos[:, self._right_gripper_dof_idx[0]]
        self.left_arm_joint_vel = self.robot.data.joint_vel[:, self._left_arm_joint_idx]
        self.right_arm_joint_vel = self.robot.data.joint_vel[:, self._right_arm_joint_idx]
        self.left_gripper_joint_vel = self.robot.data.joint_vel[:, self._left_gripper_dof_idx[0]]
        self.right_gripper_joint_vel = self.robot.data.joint_vel[:, self._right_gripper_dof_idx[0]]

        self.obs = dict(head_rgb=head_rgb, left_hand_rgb=left_hand_rgb, right_hand_rgb=right_hand_rgb,
            head_depth=head_depth, left_hand_depth=left_hand_depth, right_hand_depth=right_hand_depth,
            left_arm_joint_pos=self.left_arm_joint_pos, left_arm_joint_vel=self.left_arm_joint_vel, 
            left_gripper_joint_pos=self.left_gripper_joint_pos, left_gripper_joint_vel=self.left_gripper_joint_vel,
            right_arm_joint_pos=self.right_arm_joint_pos, right_arm_joint_vel=self.right_arm_joint_vel,
            right_gripper_joint_pos=self.right_gripper_joint_pos, right_gripper_joint_vel=self.right_gripper_joint_vel)

        # print(f'obs: {obs}')
            
        observations = {"policy": self.obs}
        return observations


    def get_key_points(self):
        # Pin positions
        # Calculate world positions of all pins
        planetary_carrier_pos = self.planetary_carrier.data.root_state_w[:, :3].clone()
        planetary_carrier_quat = self.planetary_carrier.data.root_state_w[:, 3:7].clone()

        pin_world_positions = []
        pin_world_quats = []
        for pin_local_pos in self.pin_local_positions:
            pin_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

            pin_world_quat, pin_world_pos = torch_utils.tf_combine(
                planetary_carrier_quat, planetary_carrier_pos, pin_quat.unsqueeze(0), pin_local_pos.unsqueeze(0))

            pin_world_positions.append(pin_world_pos)
            pin_world_quats.append(pin_world_quat)

        gear_world_positions = []
        gear_world_quats = []
        
        gear_names = ['sun_planetary_gear_1', 'sun_planetary_gear_2',
                        'sun_planetary_gear_3', 'sun_planetary_gear_4']
        for gear_name in gear_names:
            gear_obj = self.obj_dict[gear_name]
            gear_pos = gear_obj.data.root_state_w[:, :3].clone()
            gear_quat = gear_obj.data.root_state_w[:, 3:7].clone()

            gear_world_positions.append(gear_pos)
            gear_world_quats.append(gear_quat)
        
        carrier_world_pos = self.planetary_carrier.data.root_state_w[:, :3].clone()
        carrier_world_quat = self.planetary_carrier.data.root_state_w[:, 3:7].clone()

        ring_gear_world_pos = self.ring_gear.data.root_state_w[:, :3].clone()
        ring_gear_world_quat = self.ring_gear.data.root_state_w[:, 3:7].clone()

        reducer_world_pos = self.planetary_reducer.data.root_state_w[:, :3].clone()
        reducer_world_quat = self.planetary_reducer.data.root_state_w[:, 3:7].clone()

        return pin_world_positions, pin_world_quats, gear_world_positions, gear_world_quats, planetary_carrier_pos, planetary_carrier_quat, ring_gear_world_pos, ring_gear_world_quat, reducer_world_pos, reducer_world_quat

    def evaluate_score(self):
        pin_world_positions, pin_world_quats, gear_world_positions, gear_world_quats, planetary_carrier_pos, planetary_carrier_quat, ring_gear_world_pos, ring_gear_world_quat, reducer_world_pos, reducer_world_quat = self.get_key_points()
        score = 0

        for gear_idx in range(len(gear_world_positions)):
            gear_world_pos = gear_world_positions[gear_idx]
            gear_world_quat = gear_world_quats[gear_idx]

            # print(f"gear_world_pos: {gear_world_pos}, gear_world_quat: {gear_world_quat}")
            # Search how many gears are mounted to the planetary carrier
            num_mounted_gears = 0
            for pin_idx in range(len(pin_world_positions)):
                pin_world_pos = pin_world_positions[pin_idx]
                pin_world_quat = pin_world_quats[pin_idx]
                # print(f"pin_world_pos: {pin_world_pos}")
                # print(f"pin_world_quat: {pin_world_quat}")
                distance = torch.norm(gear_world_pos[:, :2] - pin_world_pos[:, :2])
                height_diff = gear_world_pos[:, 2] - pin_world_pos[:, 2]
                # Evaluate the angle between gear_world_quat and pin_world_quat
                angle = torch.acos(torch.dot(gear_world_quat.squeeze(0), pin_world_quat.squeeze(0)))
                # print(f"distance: {distance}")
                # print(f"angle: {angle}")
                if distance < 0.002 and angle < 0.1 and height_diff < 0.012:
                    num_mounted_gears += 1
            score += num_mounted_gears

        # Check whether the planetary carrier is mounted to the ring gear
        distance = torch.norm(planetary_carrier_pos[:, :2] - ring_gear_world_pos[:, :2])
        height_diff = planetary_carrier_pos[:, 2] - ring_gear_world_pos[:, 2]
        angle = torch.acos(torch.dot(planetary_carrier_quat.squeeze(0), ring_gear_world_quat.squeeze(0)))
        if distance < 0.005 and angle < 0.1 and height_diff < 0.004:
            score += 1

        # Check whehter the gear is mount in the middle
        for gear_idx in range(len(gear_world_positions)):
            gear_world_pos = gear_world_positions[gear_idx]
            gear_world_quat = gear_world_quats[gear_idx]
            distance = torch.norm(gear_world_pos[:, :2] - ring_gear_world_pos[:, :2])
            height_diff = gear_world_pos[:, 2] - ring_gear_world_pos[:, 2]
            angle = torch.acos(torch.dot(gear_world_quat.squeeze(0), ring_gear_world_quat.squeeze(0)))
            if distance < 0.005 and angle < 0.1 and height_diff < 0.004:
                score += 1

        # Check whether the reducer is mounted to the gear
        for gear_idx in range(len(gear_world_positions)):
            gear_world_pos = gear_world_positions[gear_idx]
            gear_world_quat = gear_world_quats[gear_idx]
            distance = torch.norm(gear_world_pos[:, :2] - reducer_world_pos[:, :2])
            height_diff = gear_world_pos[:, 2] - reducer_world_pos[:, 2]
            angle = torch.acos(torch.dot(gear_world_quat.squeeze(0), reducer_world_quat.squeeze(0)))
            if distance < 0.005 and angle < 0.1 and height_diff < 0.002:
                score += 1

        time_cost = self.rule_policy.count * self.sim.get_physics_dt()

        return score, time_cost

    def _get_rewards(self) -> torch.Tensor:
        print(f"Get rewards at {self.rule_policy.count * self.sim.get_physics_dt()} seconds")
        self.score, time_cost = self.evaluate_score()
        print(f"score: {self.score}")

        return self.score

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        print(f"--------------------------------Get dones at {self.rule_policy.count * self.sim.get_physics_dt()} seconds--------------------------------")
        finish_task = torch.tensor(self.evaluate_score() == 6, device=self.device) or self.rule_policy.count >= self.rule_policy.total_time_steps
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return finish_task, time_out

    def _initialize_scene(self):
        gripper_mat_cfg = physics_materials_cfg.RigidBodyMaterialCfg(
            static_friction=self.cfg.gripper_friction_coefficient,
            dynamic_friction=self.cfg.gripper_friction_coefficient,
            restitution=0.0,
            # (optional) combination modes if you need them:
            friction_combine_mode="average"
        )
        spawn_rigid_body_material("/World/Materials/gripper_material", gripper_mat_cfg)
        # mat_cfg.func("{ENV_REGEX_NS}/Robot/left_gripper_*", mat_cfg)
        # mat_cfg.func("{ENV_REGEX_NS}/Robot/right_gripper_*", mat_cfg)
        # sim_utils.bind_physics_material("/World/envs/env_0/Robot/left_gripper_link1/collisions", "/World/Materials/gripper_material")

        num_envs = self.scene.num_envs
        for env_idx in range(num_envs):
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/Robot/left_gripper_link1/collisions", "/World/Materials/gripper_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/Robot/left_gripper_link2/collisions", "/World/Materials/gripper_material")  
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/Robot/right_gripper_link1/collisions", "/World/Materials/gripper_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/Robot/right_gripper_link2/collisions", "/World/Materials/gripper_material")

        gear_mat_cfg = physics_materials_cfg.RigidBodyMaterialCfg(
            static_friction=self.cfg.gears_friction_coefficient,
            dynamic_friction=self.cfg.gears_friction_coefficient,
            restitution=0.0,
            friction_combine_mode="average"
        )
        spawn_rigid_body_material("/World/Materials/gear_material", gear_mat_cfg)
        for env_idx in range(num_envs):
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/ring_gear/node_/mesh_", "/World/Materials/gear_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/sun_planetary_gear_1/node_/mesh_", "/World/Materials/gear_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/sun_planetary_gear_2/node_/mesh_", "/World/Materials/gear_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/sun_planetary_gear_3/node_/mesh_", "/World/Materials/gear_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/sun_planetary_gear_4/node_/mesh_", "/World/Materials/gear_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/planetary_carrier/node_/mesh_", "/World/Materials/gear_material")
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/planetary_reducer/node_/mesh_", "/World/Materials/gear_material")
        
        table_mat_cfg = physics_materials_cfg.RigidBodyMaterialCfg(
            static_friction=self.cfg.table_friction_coefficient,
            dynamic_friction=self.cfg.table_friction_coefficient,
            restitution=0.0,
            friction_combine_mode="average"
        )
        spawn_rigid_body_material("/World/Materials/table_material", table_mat_cfg)
        for env_idx in range(num_envs):
            sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/Table/table/body_whiteLarge", "/World/Materials/table_material")
        
    def _randomize_object_positions(self, object_list: list, object_names: list,
                              safety_margin: float = 0.02, max_attempts: int = 1000):
        """Randomize positions of objects on table without overlapping

        This function places objects on the table surface ensuring they don't overlap by checking
        their bounding radii. Each object type has a defined approximate radius that represents
        its circular footprint on the table.

        Args:
            sim: Simulation context
            scene: Interactive scene containing the objects
            object_names: List of object names to randomize
            safety_margin: Additional safety distance to add between objects (in meters)
            max_attempts: Maximum attempts to find a non-overlapping position per object per environment

        Note:
            Adjust the OBJECT_RADII dictionary below if your objects have different sizes.
            To measure the radius of an object, observe it in the simulation and estimate
            the distance from its center to its furthest edge in the XY plane.
        """
        # Define approximate radii for each object type (in meters)
        # These values represent the circular bounding area of each object on the table surface
        # Adjust these based on your actual object sizes
        OBJECT_RADII = {
            'ring_gear': 0.1,              # Largest gear
            'sun_planetary_gear_1': 0.035,  # Small planetary gears
            'sun_planetary_gear_2': 0.035,
            'sun_planetary_gear_3': 0.035,
            'sun_planetary_gear_4': 0.035,
            'planetary_carrier': 0.07,     # Medium-large carrier
            'planetary_reducer': 0.04,     # Medium reducer
        }

        initial_root_state = {obj_name: torch.zeros((self.scene.num_envs, 7), device=self.device) for obj_name in object_names}

        num_envs = self.scene.num_envs

        # Store positions and object names of already placed objects for each environment
        # Each entry is a tuple: (position_tensor, object_name)
        placed_objects = [[] for _ in range(num_envs)]

        # for obj_name in object_names:
        for obj_idx, obj in enumerate(object_list):
            obj_name = object_names[obj_idx]
            # obj_cfg = SceneEntityCfg(obj_name, body_names=['node_'])
            # print(f"obj_cfg: {obj_cfg}")
            # obj_cfg.resolve(self.scene)

            # obj = self.scene[obj_name]
            # func = getattr(self, obj_name)
            # func = globals().get(obj_name)
            root_state = obj.data.default_root_state.clone()

            # Get radius for current object
            current_radius = OBJECT_RADII.get(obj_name, 0.05)  # Default to 0.05m if not specified

            # Generate non-overlapping positions for each environment
            for env_idx in range(num_envs):
                position_found = False

                for attempt in range(max_attempts):
                    # Generate random position
                    x = torch.rand(1, device=self.device).item() * 0.2 + 0.3 + self.cfg.x_offset  # range [0.3, 0.6]
                    y = torch.rand(1, device=self.device).item() * 0.8 - 0.4  # range [-0.4, 0.4]
                    z = 0.92

                    # if obj_name == "ring_gear":
                        # x = 0.26 + self.cfg.x_offset
                        # y = 0.0
                    if obj_name == "planetary_carrier":
                        x = 0.4 + self.cfg.x_offset 
                        y = 0.0
                    elif obj_name == "sun_planetary_gear_1":
                        y = torch.rand(1, device=self.device).item() * 0.4
                    elif obj_name == "sun_planetary_gear_2":
                        y = torch.rand(1, device=self.device).item() * 0.4
                    elif obj_name == "sun_planetary_gear_3":
                        y = -torch.rand(1, device=self.device).item() * 0.4
                    elif obj_name == "sun_planetary_gear_4":
                        y = -torch.rand(1, device=self.device).item() * 0.4

                    pos = torch.tensor([x, y, z], device=self.device)

                    # Check for overlaps with already placed objects in this environment
                    is_valid = True
                    for placed_pos, placed_obj_name in placed_objects[env_idx]:
                        # Get radius of the already placed object
                        placed_radius = OBJECT_RADII.get(placed_obj_name, 0.05)

                        # Calculate minimum required distance (sum of radii + safety margin)
                        min_distance = current_radius + placed_radius + safety_margin

                        # Check only x, y distance (ignore z for table surface)
                        distance = torch.norm(pos[:2] - placed_pos[:2]).item()
                        if distance < min_distance:
                            is_valid = False
                            break

                    if is_valid:
                        # Position is valid, use it
                        root_state[env_idx, :3] = pos
                        placed_objects[env_idx].append((pos, obj_name))
                        position_found = True
                        break

                if not position_found:
                    # Max attempts reached, use the last generated position anyway with a warning
                    print(f"[WARN] Could not find non-overlapping position for {obj_name} in env {env_idx} after {max_attempts} attempts.")
                    print(f"       This may indicate the table area is too crowded. Consider reducing the number of objects")
                    print(f"       or increasing the table area (x: [0.2, 0.5], y: [-0.3, 0.3]).")
                    root_state[env_idx, :3] = pos
                    placed_objects[env_idx].append((pos, obj_name))

            # Write the state to simulation
            obj.write_root_state_to_sim(root_state)
            initial_root_state[obj_name] = root_state.clone()

        return initial_root_state




    def _reset_idx(self, env_ids: Sequence[int] | None):
        print(f"--------------------------------RESET--------------------------------")
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self.rule_policy = GalaxeaRulePolicy(sim_utils.SimulationContext.instance(), self.scene, self.obj_dict)
        self.initial_root_state = None

        self.env_step_action = None
        self.env_step_joint_ids = None

        self.act = dict()
        self.obs = dict()

        self.score = 0

        self.save_hdf5_file_name = './data/data_' + datetime.now().strftime("%Y%m%d_%H%M%S") + '.hdf5'

        self.initial_root_state = self._randomize_object_positions([self.planetary_carrier, self.ring_gear, 
                                        self.sun_planetary_gear_1, self.sun_planetary_gear_2,
                                        self.sun_planetary_gear_3, self.sun_planetary_gear_4,
                                        self.planetary_reducer], ['planetary_carrier', 'ring_gear', 
                                        'sun_planetary_gear_1', 'sun_planetary_gear_2',
                                        'sun_planetary_gear_3', 'sun_planetary_gear_4',
                                        'planetary_reducer'])
        
        for obj_name, obj in self.obj_dict.items():
            obj.update(self.sim.get_physics_dt())

        self.rule_policy.set_initial_root_state(self.initial_root_state)
        self.rule_policy.prepare_mounting_plan()

        joint_pos = self.robot.data.default_joint_pos[env_ids, self._joint_idx]
        # joint_pos[:, self._joint_idx] += sample_uniform(
        #     self.cfg.initial_joint_angle_range[0] * math.pi,
        #     self.cfg.initial_joint_angle_range[1] * math.pi,
        #     joint_pos[:, self._joint_idx].shape,
        #     joint_pos.device,
        # )

        # default_root_state = self.robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self.scene.env_origins[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        # print(f"default_root_state: {default_root_state}")

        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        # self.arm_joint_vel[env_ids] = arm_joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self.robot.write_joint_state_to_sim(joint_pos, None, self._joint_idx, env_ids)
        self.robot.write_joint_position_to_sim(joint_pos, self._joint_idx, env_ids)
        self.robot.set_joint_position_target(joint_pos, self._joint_idx, env_ids)

        # Write the default torso joint position to simulation
        self.robot.write_joint_position_to_sim(torch.tensor([self.cfg.initial_torso_joint1_pos, self.cfg.initial_torso_joint2_pos, self.cfg.initial_torso_joint3_pos], device=self.device), self._torso_joint_idx, env_ids)

        # Set torso joint position limit
        self.robot.write_joint_position_limit_to_sim(torch.tensor([self.cfg.initial_torso_joint1_pos, self.cfg.initial_torso_joint1_pos], device=self.device), self._torso_joint1_idx, env_ids)
        self.robot.write_joint_position_limit_to_sim(torch.tensor([self.cfg.initial_torso_joint2_pos, self.cfg.initial_torso_joint2_pos], device=self.device), self._torso_joint2_idx, env_ids)
        self.robot.write_joint_position_limit_to_sim(torch.tensor([self.cfg.initial_torso_joint3_pos, self.cfg.initial_torso_joint3_pos], device=self.device), self._torso_joint3_idx, env_ids)


        # self.head_camera.reset(env_ids)


    def step(self, action: torch.Tensor):
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """


        current_time_s = mdp.observations.current_time_s(self)
        print(f"--------------------------------RL step at {current_time_s.item()} seconds--------------------------------")
        print(f"####################################################Before step####################################################")

        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        # self.env_step_action, self.env_step_joint_ids = self.rule_policy.get_action()
        self.env_step_action = action

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()


        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        

        print(f"####################################################Post step####################################################")
        current_pos = self.robot.data.joint_pos
        self._left_arm_action = current_pos[:, self._left_arm_joint_idx]
        self._right_arm_action = current_pos[:, self._right_arm_joint_idx]
        self._left_gripper_action = current_pos[:, self._left_gripper_dof_idx[0]]
        self._right_gripper_action = current_pos[:, self._right_gripper_dof_idx[0]]
        
        if self.env_step_joint_ids == self._left_arm_joint_idx:
            self._left_arm_action = self.env_step_action.clone()
        elif self.env_step_joint_ids == self._right_arm_joint_idx:
            self._right_arm_action = self.env_step_action.clone()
        elif self.env_step_joint_ids == self._left_arm_joint_idx + self._right_arm_joint_idx:
            self._left_arm_action = self.env_step_action.clone()[:, :6]
            self._right_arm_action = self.env_step_action.clone()[:, 6:12]
        elif self.env_step_joint_ids == self._left_gripper_dof_idx:
            self._left_gripper_action = self.env_step_action[0].clone()
        elif self.env_step_joint_ids == self._right_gripper_dof_idx:
            self._right_gripper_action = self.env_step_action[0].clone()
        self.act = dict(left_arm_action=self._left_arm_action, right_arm_action=self._right_arm_action,
            left_gripper_action=self._left_gripper_action, right_gripper_action=self._right_gripper_action)



        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

