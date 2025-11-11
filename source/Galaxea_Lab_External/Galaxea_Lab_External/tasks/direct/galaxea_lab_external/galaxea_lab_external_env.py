# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBase, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .galaxea_lab_external_env_cfg import GalaxeaLabExternalEnvCfg

from pxr import Usd, Sdf, UsdPhysics, UsdGeom
from isaaclab.sim.spawners.materials import physics_materials, physics_materials_cfg
from isaaclab.sim.spawners.materials import spawn_rigid_body_material
from isaaclab.managers import SceneEntityCfg

import isaacsim.core.utils.torch as torch_utils

from Galaxea_Lab_External.robots import GalaxeaRulePolicy
from isaaclab.sensors import Camera

class GalaxeaLabExternalEnv(DirectRLEnv):
    cfg: GalaxeaLabExternalEnvCfg

    def __init__(self, cfg: GalaxeaLabExternalEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._left_arm_joint_idx, _ = self.robot.find_joints(self.cfg.left_arm_joint_dof_name)
        self._right_arm_joint_idx, _ = self.robot.find_joints(self.cfg.right_arm_joint_dof_name)
        self._left_gripper_dof_idx, _ = self.robot.find_joints(self.cfg.left_gripper_dof_name)
        self._right_gripper_dof_idx, _ = self.robot.find_joints(self.cfg.right_gripper_dof_name)

        self._torso_joint_idx, _ = self.robot.find_joints(self.cfg.torso_joint_dof_name)

        print(f"_left_arm_joint_idx: {self._left_arm_joint_idx}")
        print(f"_right_arm_joint_idx: {self._right_arm_joint_idx}")
        print(f"_left_gripper_dof_idx: {self._left_gripper_dof_idx}")
        print(f"_right_gripper_dof_idx: {self._right_gripper_dof_idx}")

        self._joint_idx = self._left_arm_joint_idx + self._right_arm_joint_idx + self._left_gripper_dof_idx + self._right_gripper_dof_idx

        self.left_arm_joint_pos = self.robot.data.joint_pos[:, self._left_arm_joint_idx]
        self.right_arm_joint_pos = self.robot.data.joint_pos[:, self._right_arm_joint_idx]
        self.left_gripper_joint_pos = self.robot.data.joint_pos[:, self._left_gripper_dof_idx]
        self.right_gripper_joint_pos = self.robot.data.joint_pos[:, self._right_gripper_dof_idx]
        
        print(f"left_arm_joint_pos: {self.left_arm_joint_pos}")
        print(f"right_arm_joint_pos: {self.right_arm_joint_pos}")
        print(f"left_gripper_joint_pos: {self.left_gripper_joint_pos}")
        print(f"right_gripper_joint_pos: {self.right_gripper_joint_pos}")

        self.joint_pos = self.robot.data.joint_pos[:, self._joint_idx]

        self.rule_policy = GalaxeaRulePolicy(sim_utils.SimulationContext.instance(), self.scene, self.obj_dict)
        self.initial_root_state = None

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # self.table = Articulation(self.cfg.table_cfg)
        # self.cfg.table_cfg.func("/World/envs/env_.*/Table", self.cfg.table_cfg)
        self.head_camera = Camera(self.cfg.head_camera_cfg)
        self.left_hand_camera = Camera(self.cfg.left_hand_camera_cfg)
        self.right_hand_camera = Camera(self.cfg.right_hand_camera_cfg)

        self.table = sim_utils.spawn_from_usd("/World/envs/env_.*/Table", self.cfg.table_cfg.spawn,
            translation=self.cfg.table_cfg.init_state.pos, 
            orientation=self.cfg.table_cfg.init_state.rot)

        self.ring_gear = RigidObject(self.cfg.ring_gear_cfg)
        self.sun_planetary_gear_1 = RigidObject(self.cfg.sun_planetary_gear_1_cfg)
        self.sun_planetary_gear_2 = RigidObject(self.cfg.sun_planetary_gear_2_cfg)
        self.sun_planetary_gear_3 = RigidObject(self.cfg.sun_planetary_gear_3_cfg)
        self.sun_planetary_gear_4 = RigidObject(self.cfg.sun_planetary_gear_4_cfg)
        self.planetary_carrier = RigidObject(self.cfg.planetary_carrier_cfg)
        self.planetary_reducer = RigidObject(self.cfg.planetary_reducer_cfg)

        self.pin_local_positions = [
            torch.tensor([0.0, -0.054, 0.0], device=self.device),      # pin_0
            torch.tensor([0.0465, 0.0268, 0.0], device=self.device),   # pin_1
            torch.tensor([-0.0465, 0.0268, 0.0], device=self.device),  # pin_2
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
        self.actions = actions.clone()
        # print(f"_pre_physics_step actions: {self.actions}")

    def _apply_action(self) -> None:
        self.action, joint_ids = self.rule_policy.get_action()
        if self.action is not None:
            self.robot.set_joint_position_target(self.action, joint_ids=joint_ids)
        # else:
        #     joint_pos = self.robot.data.default_joint_pos[:, self._joint_idx]
        #     self.robot.write_joint_position_to_sim(joint_pos, self._joint_idx, None)
        self.rule_policy.count += 1
        sim_dt = self.sim.get_physics_dt()
        print(f"Time: {self.rule_policy.count * sim_dt}")
        # print(f"action: {self.action}")
        # print(f"joint_ids: {joint_ids}")

        # pos = self.scene["sun_planetary_gear_1"].data.root_state_w[:, :3].clone()
        # pos = self.sun_planetary_gear_1.get_world_pose()
        # print(f"1 scene pos root: {pos}")

        for obj_name, obj in self.obj_dict.items():
            obj.update(sim_dt)

    def _get_observations(self) -> dict:
        data_type = "rgb"
        rgb = self.head_camera.data.output[data_type]
        left_hand_rgb = self.left_hand_camera.data.output[data_type]
        right_hand_rgb = self.right_hand_camera.data.output[data_type]
        print(f"rgb: {rgb.shape}")
        print(f"left_hand_rgb: {left_hand_rgb.shape}")
        print(f"right_hand_rgb: {right_hand_rgb.shape}")

        # obs = torch.cat(
        #     (
        #         rgb,
        #         left_hand_rgb,
        #         right_hand_rgb,
        #         self.left_arm_joint_pos.unsqueeze(dim=1),
        #         self.right_arm_joint_pos.unsqueeze(dim=1),
        #         self.left_gripper_joint_pos.unsqueeze(dim=1),
        #         self.right_gripper_joint_pos.unsqueeze(dim=1),
        #     ),
        #     dim=-1,
        # )
        obs = dict(rgb=rgb, left_hand_rgb=left_hand_rgb, right_hand_rgb=right_hand_rgb,
            left_arm_joint_pos=self.left_arm_joint_pos, right_arm_joint_pos=self.right_arm_joint_pos,
            left_gripper_joint_pos=self.left_gripper_joint_pos, right_gripper_joint_pos=self.right_gripper_joint_pos)
            
        observations = {"policy": obs}
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
        score, time_cost = self.evaluate_score()
        print(f"score: {score}")

        return score

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        finish_task = torch.tensor(self.evaluate_score() == 6, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return finish_task, time_out

    def _initialize_scene(self):
        gripper_mat_cfg = physics_materials_cfg.RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=2.0,
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
            static_friction=0.12,
            dynamic_friction=0.12,
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
            static_friction=0.5,
            dynamic_friction=0.5,
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

        x_offset = 0.2

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
                    x = torch.rand(1, device=self.device).item() * 0.2 + 0.3 + x_offset  # range [0.3, 0.6]
                    y = torch.rand(1, device=self.device).item() * 0.6 - 0.3  # range [-0.3, 0.3]
                    z = 0.92

                    if obj_name == "ring_gear":
                        x = 0.24 + x_offset
                        y = 0.0
                    elif obj_name == "planetary_carrier":
                        x = 0.42 + x_offset 
                        y = 0.0
                    elif obj_name == "sun_planetary_gear_1":
                        y = torch.rand(1, device=self.device).item() * 0.3
                    elif obj_name == "sun_planetary_gear_2":
                        y = torch.rand(1, device=self.device).item() * 0.3
                    elif obj_name == "sun_planetary_gear_3":
                        y = -torch.rand(1, device=self.device).item() * 0.3
                    elif obj_name == "sun_planetary_gear_4":
                        y = -torch.rand(1, device=self.device).item() * 0.3

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
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self.initial_root_state = self._randomize_object_positions([self.ring_gear, self.planetary_carrier,
                                        self.sun_planetary_gear_1, self.sun_planetary_gear_2,
                                        self.sun_planetary_gear_3, self.sun_planetary_gear_4,
                                        self.planetary_reducer], ['ring_gear', 'planetary_carrier',
                                        'sun_planetary_gear_1', 'sun_planetary_gear_2',
                                        'sun_planetary_gear_3', 'sun_planetary_gear_4',
                                        'planetary_reducer'])

        self.rule_policy.set_initial_root_state(self.initial_root_state)
        self.rule_policy.prepare_mounting_plan()

        joint_pos = self.robot.data.default_joint_pos[env_ids, self._joint_idx]
        # joint_pos[:, self._joint_idx] += sample_uniform(
        #     self.cfg.initial_joint_angle_range[0] * math.pi,
        #     self.cfg.initial_joint_angle_range[1] * math.pi,
        #     joint_pos[:, self._joint_idx].shape,
        #     joint_pos.device,
        # )

        # print(f"shape of joint_pos: {joint_pos.shape}")

        # default_root_state = self.robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self.scene.env_origins[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        # print(f"default_root_state: {default_root_state}")

        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # print(f"default_root_state: {default_root_state}")
        # print(f"self.scene.env_origins[env_ids]: {self.scene.env_origins[env_ids]}")

        self.joint_pos[env_ids] = joint_pos
        # self.arm_joint_vel[env_ids] = arm_joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self.robot.write_joint_state_to_sim(joint_pos, None, self._joint_idx, env_ids)
        self.robot.write_joint_position_to_sim(joint_pos, self._joint_idx, env_ids)
        self.robot.set_joint_position_target(joint_pos, self._joint_idx, env_ids)

        # self.robot.write_joint_position_to_sim(torch.tensor([28.6479 / 180.0 * math.pi, -45.8366 / 180.0 * math.pi, 28.6479 / 180.0 * math.pi], device=self.device), self._torso_joint_idx, env_ids)

        

