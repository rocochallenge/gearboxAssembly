# -*- coding: utf-8 -*-
# Copyright (c) 2024 Galaxea

"""Configuration for the Galaxea R1 robot (production date: 0604)
"""

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from Galaxea_Lab_External import GALAXEA_LAB_ASSETS_DIR
import math


##
# Configuration
##

GALAXEA_R1_CHALLENGE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{GALAXEA_LAB_ASSETS_DIR}/Robots/Galaxea/r1_DVT_colored_cam_pos.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.1,
                angular_damping=0.1,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=192,
                # max_contact_impulse=1e32,
                max_contact_impulse=1e3,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=192,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.05, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
            "left_arm_joint1": -20.0 / 180.0 * math.pi,
            "left_arm_joint2": 100.6 / 180.0 * math.pi,
            "left_arm_joint3": -24.0 / 180.0 * math.pi,
            "left_arm_joint4": 17.8 / 180.0 * math.pi,
            "left_arm_joint5": 38.7 / 180.0 * math.pi,
            "left_arm_joint6": 20.1 / 180.0 * math.pi,
            "left_gripper_axis1": 0.04,
            # "left_gripper_axis2": 0.04,
            "right_arm_joint1": -20.0 / 180.0 * math.pi,
            "right_arm_joint2": 100.8 / 180.0 * math.pi,
            "right_arm_joint3": -22.0 / 180.0 * math.pi,
            "right_arm_joint4": -40 / 180.0 * math.pi,
            "right_arm_joint5": -67.6 / 180.0 * math.pi,
            "right_arm_joint6": 18.1 / 180.0 * math.pi,
            "right_gripper_axis1": 0.04,
            # "right_gripper_axis2": 0.04,
            # "torso_joint1": 28.6479 / 180.0 * math.pi,
            # "torso_joint2": -45.8366 / 180.0 * math.pi,
            # "torso_joint3": 28.6479 / 180.0 * math.pi,
            # "torso_joint1": 0.5,
            # "torso_joint2": -0.8,
            # "torso_joint3": 0.8,
            "torso_joint1": 0.0,
            "torso_joint2": 0.0,
            "torso_joint3": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "r1_arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_arm_joint[1-5]"],
                stiffness=1050.0,
                damping=100.0,
                friction=0.0,
                armature=0.1,
                effort_limit_sim=87,
                # velocity_limit_sim=124.6,
                velocity_limit_sim=10,
            ),
            # "r1_eefs": ImplicitActuatorCfg(
            #     joint_names_expr=[".*_arm_joint6"],
            #     stiffness=10000.0,
            #     damping=200.0,
            #     friction=0.0,
            #     armature=0.0,
            #     effort_limit_sim=12,
            #     velocity_limit_sim=149.5,
            # ),
            "r1_eefs": ImplicitActuatorCfg(
                joint_names_expr=[".*_arm_joint6"],
                stiffness=1050.0,
                damping=100.0,
                friction=0.0,
                armature=0.1,
                effort_limit_sim=87,
                # velocity_limit_sim=124.6,
                velocity_limit_sim=10,
            ),
            "r1_grippers": ImplicitActuatorCfg(
                joint_names_expr=[".*_gripper_axis1"],
                effort_limit_sim=100.0,
                velocity_limit_sim=0.07,
                stiffness=25000.0,
                damping=1000.0,
                friction=0.2,
                armature=0.2,
            ),
            "r1_torso": ImplicitActuatorCfg(
                joint_names_expr=["torso_joint[1-5]"],
                stiffness=1050.0,
                damping=100.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=87,
                velocity_limit_sim=124.6,
            ),
        },
)

GALAXEA_R1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{GALAXEA_LAB_ASSETS_DIR}/Robots/Galaxea/r1_DVT_colored_cam.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "left_arm_joint1": 0.076,
            "left_arm_joint2": 0.058,
            "left_arm_joint3": -0.020,
            "left_arm_joint4": 0.502,
            "left_arm_joint5": -0.279,
            "left_arm_joint6": -0.218,
            "left_gripper_axis1": 0.03,
            "left_gripper_axis2": 0.03,
            "right_arm_joint1": -0.800,
            "right_arm_joint2": -0.502,
            "right_arm_joint3": 0.0,
            "right_arm_joint4": 0.718,
            "right_arm_joint5": -0.761,
            "right_arm_joint6": 2.326,
            "right_gripper_axis1": 0.03,
            "right_gripper_axis2": 0.03,
        },
    ),
    actuators={
        "r1_arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_joint[1-5]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "r1_eefs": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_joint6"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "r1_grippers": ImplicitActuatorCfg(
            joint_names_expr=[".*_gripper_axis.*"],
            effort_limit=200.0,
            velocity_limit=0.25,
            stiffness=1e6,  # 1e7,
            damping=1e4,  # 1e5,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

GALAXEA_R1_HIGH_PD_CFG = GALAXEA_R1_CFG.copy()
GALAXEA_R1_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = False
GALAXEA_R1_HIGH_PD_CFG.actuators["r1_arms"].stiffness = 400.0
GALAXEA_R1_HIGH_PD_CFG.actuators["r1_arms"].damping = 80.0
GALAXEA_R1_HIGH_PD_CFG.actuators["r1_eefs"].stiffness = 1000.0
GALAXEA_R1_HIGH_PD_CFG.actuators["r1_eefs"].damping = 200.0

GALAXEA_R1_HIGH_PD_GRIPPER_CFG = GALAXEA_R1_HIGH_PD_CFG.copy()
GALAXEA_R1_HIGH_PD_GRIPPER_CFG.actuators["r1_grippers"].stiffness = 1e3
GALAXEA_R1_HIGH_PD_GRIPPER_CFG.actuators["r1_grippers"].damping = 1e2
# GALAXEA_R1_HIGH_PD_GRIPPER_CFG.actuators["r1_grippers"].stiffness = 1e4
# GALAXEA_R1_HIGH_PD_GRIPPER_CFG.actuators["r1_grippers"].damping = 1e3

GALAXEA_CAMERA_CFG = CameraCfg(
    prim_path="/World/envs/env_.*/Camera",  # should be replaced with the actual parent frame
    update_period=1 / 60.0,  # 30 Hz
    height=240,
    width=320,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=12,
        focus_distance=100.0,
        horizontal_aperture=20.955,
        clipping_range=(0.01, 100),
    ),
    offset=CameraCfg.OffsetCfg(  # offset from the parent frame
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        convention="ros",
    ),
)

GALAXEA_HEAD_CAMERA_CFG = CameraCfg(
    prim_path="/World/envs/env_.*/Robot/zed_link/head_cam/head_cam",  # should be replaced with the actual parent frame
    update_period=0.0,
    height=240,
    width=320,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=2.12,
        focus_distance=100.0,
        horizontal_aperture=6.055,
        clipping_range=(0.01, 100),
    ),
    offset=CameraCfg.OffsetCfg(  # offset from the parent frame
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        # rot=(0.99027, -0.13917, 0.0, 0.0), # Rotate the camera to have the best view of the table
        convention="opengl",
    ),
)

GALAXEA_HAND_CAMERA_CFG = CameraCfg(
    prim_path="/World/envs/env_.*/Robot/left_realsense_link/left_hand_cam/left_hand_cam",  # should be replaced with the actual parent frame
    update_period=0.0,
    height=240,
    width=320,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=2.12,
        focus_distance=100.0,
        horizontal_aperture=6.055,
        clipping_range=(0.01, 100),
    ),
    offset=CameraCfg.OffsetCfg(  # offset from the parent frame
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        convention="opengl",
    ),
)
