# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg
import math

# from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
# from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
# from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
# from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
# from isaaclab.sim import SimulationContext
# from isaaclab.utils import configclass
# from isaaclab.controllers import (
#     DifferentialIKController,
#     DifferentialIKControllerCfg,
# )

from Galaxea_Lab_External.robots import (
    ACTIVE_ROBOT_BUNDLE,
    RobotBundle,
    TABLE_CFG,
    RING_GEAR_CFG,
    SUN_PLANETARY_GEAR_CFG,
    PLANETARY_CARRIER_CFG,
    PLANETARY_REDUCER_CFG,
)

@configclass
class GalaxeaLabExternalEnvCfg(DirectRLEnvCfg):

    # Record data
    record_data = True
    record_freq = 5

    # env
    sim_dt = 0.01
    decimation = 5
    episode_length_s = 60.0
    # - spaces definition
    action_space = 14
    observation_space = 14
    state_space = 0
    num_rerenders_on_reset = 5

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=sim_dt, render_interval=decimation)

    # viewport camera (captured manually from Isaac Sim viewport)
    viewer: ViewerCfg = ViewerCfg(eye=(1.935, 1.238, 1.911), lookat=(0.676, 0.131, 0.821))

    # robot(s)
    robot_bundle: RobotBundle = ACTIVE_ROBOT_BUNDLE
    robot_cfg: ArticulationCfg = ACTIVE_ROBOT_BUNDLE.articulation_cfg.replace(prim_path="/World/envs/env_.*/Robot")
    rule_policy_class: type = ACTIVE_ROBOT_BUNDLE.recovery_rule_policy_class

    # table_cfg: AssetBaseCfg = TABLE_CFG.copy()
    table_cfg: RigidObjectCfg = TABLE_CFG.replace(prim_path="/World/envs/env_.*/Table")

    ring_gear_cfg: RigidObjectCfg = RING_GEAR_CFG.replace(prim_path="/World/envs/env_.*/ring_gear",
                                                                       init_state=RigidObjectCfg.InitialStateCfg(
                                                                           pos=(0.45, 0.0, 1.0),
                                                                           rot=(1.0, 0.0, 0.0, 0.0),
                                                                       ))


    sun_planetary_gear_1_cfg: RigidObjectCfg = SUN_PLANETARY_GEAR_CFG.replace(prim_path="/World/envs/env_.*/sun_planetary_gear_1",
                                                                       init_state=RigidObjectCfg.InitialStateCfg(
                                                                           pos=(0.4, -0.2, 1.0),
                                                                           rot=(1.0, 0.0, 0.0, 0.0),
                                                                       ))
    
    sun_planetary_gear_2_cfg: RigidObjectCfg = SUN_PLANETARY_GEAR_CFG.replace(prim_path="/World/envs/env_.*/sun_planetary_gear_2",
                                                                       init_state=RigidObjectCfg.InitialStateCfg(
                                                                           pos=(0.5, -0.25, 1.0),
                                                                           rot=(1.0, 0.0, 0.0, 0.0),
                                                                       ))
    sun_planetary_gear_3_cfg: RigidObjectCfg = SUN_PLANETARY_GEAR_CFG.replace(prim_path="/World/envs/env_.*/sun_planetary_gear_3",
                                                                       init_state=RigidObjectCfg.InitialStateCfg(
                                                                           pos=(0.45, -0.15, 1.0),
                                                                           rot=(1.0, 0.0, 0.0, 0.0),
                                                                       ))
    sun_planetary_gear_4_cfg: RigidObjectCfg = SUN_PLANETARY_GEAR_CFG.replace(prim_path="/World/envs/env_.*/sun_planetary_gear_4",
                                                                       init_state=RigidObjectCfg.InitialStateCfg(
                                                                           pos=(0.55, -0.3, 1.0),
                                                                           rot=(1.0, 0.0, 0.0, 0.0),
                                                                       ))
    planetary_carrier_cfg: RigidObjectCfg = PLANETARY_CARRIER_CFG.replace(prim_path="/World/envs/env_.*/planetary_carrier",
                                                                       init_state=RigidObjectCfg.InitialStateCfg(
                                                                           pos=(0.5, 0.25, 1.0),
                                                                           rot=(1.0, 0.0, 0.0, 0.0),
                                                                       ))
    planetary_reducer_cfg: RigidObjectCfg = PLANETARY_REDUCER_CFG.replace(prim_path="/World/envs/env_.*/planetary_reducer",
                                                                       init_state=RigidObjectCfg.InitialStateCfg(
                                                                           pos=(0.3, 0.1, 1.0),
                                                                        #    rot=(0.7071068 , 0.0, 0.0, 0.7071068),
                                                                           rot=(1.0, 0.0, 0.0, 0.0),
                                                                       ))
    # Physics
    table_friction_coefficient = 0.4
    gears_friction_coefficient = 0.01
    gripper_friction_coefficient = 2.0

    # Camera
    head_camera_cfg: CameraCfg = ACTIVE_ROBOT_BUNDLE.head_camera_cfg
    left_hand_camera_cfg: CameraCfg = ACTIVE_ROBOT_BUNDLE.left_hand_camera_cfg
    right_hand_camera_cfg: CameraCfg = ACTIVE_ROBOT_BUNDLE.right_hand_camera_cfg

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    left_arm_joint_dof_name: str = ACTIVE_ROBOT_BUNDLE.left_arm_joint_pattern
    right_arm_joint_dof_name: str = ACTIVE_ROBOT_BUNDLE.right_arm_joint_pattern
    left_gripper_dof_name: str = ACTIVE_ROBOT_BUNDLE.left_gripper_dof_name
    right_gripper_dof_name: str = ACTIVE_ROBOT_BUNDLE.right_gripper_dof_name

    torso_joint_dof_name: str = ACTIVE_ROBOT_BUNDLE.torso_joint_pattern  # excludes torso_joint4 (R1-only)
    torso_joint1_dof_name = "torso_joint1"
    torso_joint2_dof_name = "torso_joint2"
    torso_joint3_dof_name = "torso_joint3"
    torso_joint4_dof_name = "torso_joint4"

    # Robot initial torso joint position (sourced from the active robot bundle)
    initial_torso_pos: tuple[float, float, float] = ACTIVE_ROBOT_BUNDLE.initial_torso_pos

    x_offset = 0.2