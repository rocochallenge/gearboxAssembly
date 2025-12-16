# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
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
    GALAXEA_R1_CHALLENGE_CFG,
    GALAXEA_HEAD_CAMERA_CFG,
    GALAXEA_HAND_CAMERA_CFG,
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

    # robot(s)
    robot_cfg: ArticulationCfg = GALAXEA_R1_CHALLENGE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

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
    head_camera_cfg: CameraCfg = GALAXEA_HEAD_CAMERA_CFG.replace(prim_path="/World/envs/env_.*/Robot/zed_link/head_cam/head_cam")
    left_hand_camera_cfg: CameraCfg = GALAXEA_HAND_CAMERA_CFG.replace(prim_path="/World/envs/env_.*/Robot/left_realsense_link/left_hand_cam/left_hand_cam")
    right_hand_camera_cfg: CameraCfg = GALAXEA_HAND_CAMERA_CFG.replace(prim_path="/World/envs/env_.*/Robot/right_realsense_link/right_hand_cam/right_hand_cam")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    left_arm_joint_dof_name = "left_arm_joint.*"
    right_arm_joint_dof_name = "right_arm_joint.*"
    left_gripper_dof_name = "left_gripper_axis1"
    right_gripper_dof_name = "right_gripper_axis1"

    torso_joint_dof_name = "torso_joint[1-3]" # Since in current task, torso_joint4 will always be fixed at 0.0
    torso_joint1_dof_name = "torso_joint1"
    torso_joint2_dof_name = "torso_joint2"
    torso_joint3_dof_name = "torso_joint3"
    torso_joint4_dof_name = "torso_joint4"

    # Robot initial torso joint position
    initial_torso_joint1_pos = 0.5
    initial_torso_joint2_pos = -0.8
    initial_torso_joint3_pos = 0.5

    x_offset = 0.2