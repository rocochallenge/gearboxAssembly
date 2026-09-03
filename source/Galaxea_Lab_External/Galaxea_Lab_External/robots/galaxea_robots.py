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
                solver_position_iteration_count=128,
                solver_velocity_iteration_count=128,
                # max_contact_impulse=1e32,
                max_contact_impulse=1e3,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=128,
                solver_velocity_iteration_count=128,
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
            # NOTE: r1_DVT_colored_cam_pos.usd has torso joint limits baked
            # to [0, 0]. Init pose must satisfy that (so we keep these zero);
            # the actual runtime torso target lives in
            # GALAXEA_R1_BUNDLE.initial_torso_pos and is applied in _reset_idx
            # after write_joint_position_limit_to_sim widens the limits.
            "torso_joint1": 0.0,
            "torso_joint2": 0.0,
            "torso_joint3": 0.0,
            "torso_joint4": 0.0,
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

# GALAXEA_R1_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{GALAXEA_LAB_ASSETS_DIR}/Robots/Galaxea/r1_DVT_colored_cam.usd",
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             max_depenetration_velocity=5.0,
#         ),
#         activate_contact_sensors=False,
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True,
#             solver_position_iteration_count=8,
#             solver_velocity_iteration_count=0,
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         joint_pos={
#             "left_arm_joint1": 0.076,
#             "left_arm_joint2": 0.058,
#             "left_arm_joint3": -0.020,
#             "left_arm_joint4": 0.502,
#             "left_arm_joint5": -0.279,
#             "left_arm_joint6": -0.218,
#             "left_gripper_axis1": 0.03,
#             "left_gripper_axis2": 0.03,
#             "right_arm_joint1": -0.800,
#             "right_arm_joint2": -0.502,
#             "right_arm_joint3": 0.0,
#             "right_arm_joint4": 0.718,
#             "right_arm_joint5": -0.761,
#             "right_arm_joint6": 2.326,
#             "right_gripper_axis1": 0.03,
#             "right_gripper_axis2": 0.03,
#         },
#     ),
#     actuators={
#         "r1_arms": ImplicitActuatorCfg(
#             joint_names_expr=[".*_arm_joint[1-5]"],
#             effort_limit=87.0,
#             velocity_limit=2.175,
#             stiffness=80.0,
#             damping=4.0,
#         ),
#         "r1_eefs": ImplicitActuatorCfg(
#             joint_names_expr=[".*_arm_joint6"],
#             effort_limit=12.0,
#             velocity_limit=2.61,
#             stiffness=80.0,
#             damping=4.0,
#         ),
#         "r1_grippers": ImplicitActuatorCfg(
#             joint_names_expr=[".*_gripper_axis.*"],
#             effort_limit=200.0,
#             velocity_limit=0.25,
#             stiffness=1e6,  # 1e7,
#             damping=1e4,  # 1e5,
#         ),
#     },
#     soft_joint_pos_limit_factor=1.0,
# )

# GALAXEA_R1_HIGH_PD_CFG = GALAXEA_R1_CFG.copy()
# GALAXEA_R1_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = False
# GALAXEA_R1_HIGH_PD_CFG.actuators["r1_arms"].stiffness = 400.0
# GALAXEA_R1_HIGH_PD_CFG.actuators["r1_arms"].damping = 80.0
# GALAXEA_R1_HIGH_PD_CFG.actuators["r1_eefs"].stiffness = 1000.0
# GALAXEA_R1_HIGH_PD_CFG.actuators["r1_eefs"].damping = 200.0

# GALAXEA_R1_HIGH_PD_GRIPPER_CFG = GALAXEA_R1_HIGH_PD_CFG.copy()
# GALAXEA_R1_HIGH_PD_GRIPPER_CFG.actuators["r1_grippers"].stiffness = 1e3
# GALAXEA_R1_HIGH_PD_GRIPPER_CFG.actuators["r1_grippers"].damping = 1e2
# # GALAXEA_R1_HIGH_PD_GRIPPER_CFG.actuators["r1_grippers"].stiffness = 1e4
# # GALAXEA_R1_HIGH_PD_GRIPPER_CFG.actuators["r1_grippers"].damping = 1e3

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


##
# R1_Lite (vendor URDF: mmp_revB_invconfig_upright_a1x). Coexists with R1.
##

GALAXEA_R1_LITE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{GALAXEA_LAB_ASSETS_DIR}/Robots/R1_Lite/r1_lite.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            linear_damping=0.1,
            angular_damping=0.1,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=False,
            solver_position_iteration_count=128,
            solver_velocity_iteration_count=128,
            max_contact_impulse=1e3,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=128,
            solver_velocity_iteration_count=128,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.05, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Initial gesture tuned for the gearbox-assembly task. Torso pose
            # leans the upper body toward the table; arms pre-folded at the
            # elbow so DLS starts in the same branch it converges to during
            # gripper-down picks. Gripper fingers initialized open.
            "left_arm_joint1": -20.0 / 180.0 * math.pi,
            "left_arm_joint2":  1.2,
            "left_arm_joint3": -0.98,
            "left_arm_joint4":   0.636,
            "left_arm_joint5":   0.09,
            "left_arm_joint6":   -0.18,
            "left_gripper_finger_joint1": 0.04,
            "right_arm_joint1": -20.0 / 180.0 * math.pi,
            "right_arm_joint2":  1.2,
            "right_arm_joint3": -0.98,
            "right_arm_joint4":   0.636,
            "right_arm_joint5":   0.09,
            "right_arm_joint6":   -0.18,
            "right_gripper_finger_joint1": 0.04,
            "torso_joint1": 0.4,
            "torso_joint2": -0.66,
            "torso_joint3": -0.8,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "r1_lite_arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_joint[1-5]"],
            stiffness=1050.0,
            damping=100.0,
            friction=0.0,
            armature=0.1,
            effort_limit_sim=87,
            velocity_limit_sim=10,
        ),
        "r1_lite_eefs": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_joint6"],
            stiffness=1050.0,
            damping=100.0,
            friction=0.0,
            armature=0.1,
            effort_limit_sim=87,
            velocity_limit_sim=10,
        ),
        "r1_lite_grippers": ImplicitActuatorCfg(
            # joint2 mimics joint1 (URDF <mimic> tag, multiplier=-1) so only joint1 is independently driven.
            joint_names_expr=[".*_gripper_finger_joint1"],
            effort_limit_sim=100.0,
            velocity_limit_sim=0.07,
            stiffness=25000.0,
            damping=1000.0,
            friction=0.2,
            armature=0.2,
        ),
        "r1_lite_torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_joint[1-3]"],
            stiffness=1050.0,
            damping=100.0,
            friction=0.0,
            armature=0.0,
            effort_limit_sim=87,
            velocity_limit_sim=124.6,
        ),
        "r1_lite_wheels": ImplicitActuatorCfg(
            joint_names_expr=["(steer|wheel)_motor_joint[1-3]"],
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=0.0,
            velocity_limit_sim=0.0,
        ),
    },
)

GALAXEA_R1_LITE_HEAD_CAMERA_CFG = CameraCfg(
    prim_path="/World/envs/env_.*/Robot/camera_head_left_link/head_cam",
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
    offset=CameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        convention="opengl",
    ),
)

GALAXEA_R1_LITE_HAND_CAMERA_CFG = CameraCfg(
    prim_path="/World/envs/env_.*/Robot/left_D405_link/left_hand_cam",
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
    offset=CameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        convention="opengl",
    ),
)


##
# R1Pro (vendor URDF: r1pro_2026, G1Z gripper). Coexists with R1 and R1_Lite.
#
# Converted with scripts/convert_r1_pro_urdf.py. Compared with R1_Lite:
#   - 7-DOF arms (left/right_arm_joint1..7), IK end-effector is *_arm_link7.
#   - 4-DOF torso: joint1..3 pitch, joint4 waist yaw (kept at 0).
#   - Gripper extends along link7 -Z; "gripper down" is the identity link7 pose.
#   - Fingers: joint1 in [0, 0.065] (0 = closed), joint2 mimics joint1 with -1.
#   - Arm reach (shoulder -> link7) is 0.30..0.57 m and the gripper adds
#     ~0.27 m below link7, so the torso lean below was chosen by offline
#     URDF-IK against the randomized scene (carrier fixed at (0.45, 0), gears
#     at x in [0.45, 0.65] and |y| <= 0.4 on each arm's side, lifts up to
#     +0.17 m). All of it is reachable with the gripper vertical except a gear
#     at the far centre (x ~= 0.65, |y| < 0.05), which is ~3 cm short — see
#     docs/r1_pro_integration.md.
##

GALAXEA_R1_PRO_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{GALAXEA_LAB_ASSETS_DIR}/Robots/R1_Pro/r1_pro.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            linear_damping=0.1,
            angular_damping=0.1,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=False,
            solver_position_iteration_count=128,
            solver_velocity_iteration_count=128,
            max_contact_impulse=1e3,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=128,
            solver_velocity_iteration_count=128,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.05, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # "Ready" pose: link7 at (0.47, +-0.46, 1.26) world, i.e. 35 cm above
            # the table top (the G1Z gripper hangs 0.27 m below link7 — the
            # fingertips must clear the table at z = 0.909). Gripper points
            # straight down with link7 +X yawed -135/+135 deg (left/right),
            # matching R1ProRulePolicy's GRIPPER_DOWN_QUAT*; every arm joint
            # >= 0.39 rad from its limit; right arm is the mirror of the left.
            # Found by offline IK on the URDF; in-sim FK agrees to < 1 cm.
            "left_arm_joint1": -1.767,
            "left_arm_joint2":  1.370,
            "left_arm_joint3": -1.860,
            "left_arm_joint4": -1.669,
            "left_arm_joint5": -0.321,
            "left_arm_joint6": -0.401,
            "left_arm_joint7": -1.044,
            "left_gripper_finger_joint1": 0.055,
            "right_arm_joint1": -1.767,
            "right_arm_joint2": -1.370,
            "right_arm_joint3":  1.860,
            "right_arm_joint4": -1.669,
            "right_arm_joint5":  0.321,
            "right_arm_joint6": -0.401,
            "right_arm_joint7":  1.044,
            "right_gripper_finger_joint1": 0.055,
            # Torso: link1 23 deg fwd, link2 back to vertical, upper body 23 deg
            # fwd -> shoulders at ~(0.21, +-0.17, 1.39), head at (0.24, 1.46),
            # camera looking ~45 deg down onto the table. Higher shoulders than
            # the first attempt (1.29) so that gears spawning straight ahead of
            # a shoulder at x ~= 0.45 stay outside the elbow's ~0.30 m minimum
            # fold radius, while x = 0.65 corners stay inside the 0.57 m reach.
            "torso_joint1":  0.4,
            "torso_joint2": -0.4,
            "torso_joint3": -0.4,
            "torso_joint4":  0.0,
        },
        # Robot base at the world origin (the table's front edge is at x = 0.34).
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "r1_pro_arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_joint[1-6]"],
            stiffness=1050.0,
            damping=100.0,
            friction=0.0,
            armature=0.1,
            effort_limit_sim=87,
            velocity_limit_sim=10,
        ),
        "r1_pro_eefs": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_joint7"],
            stiffness=1050.0,
            damping=100.0,
            friction=0.0,
            armature=0.1,
            effort_limit_sim=87,
            velocity_limit_sim=10,
        ),
        "r1_pro_grippers": ImplicitActuatorCfg(
            # joint2 mimics joint1 (URDF <mimic> tag, multiplier=-1, as for R1_Lite;
            # a rigid PhysX mimic joint after conversion) so only joint1 is driven.
            joint_names_expr=[".*_gripper_finger_joint1"],
            effort_limit_sim=100.0,
            velocity_limit_sim=0.07,
            stiffness=25000.0,
            damping=1000.0,
            friction=0.2,
            armature=0.2,
        ),
        "r1_pro_torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_joint[1-4]"],
            stiffness=1050.0,
            damping=100.0,
            friction=0.0,
            armature=0.0,
            effort_limit_sim=87,
            velocity_limit_sim=124.6,
        ),
        "r1_pro_wheels": ImplicitActuatorCfg(
            joint_names_expr=["(steer|wheel)_motor_joint[1-3]"],
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=0.0,
            velocity_limit_sim=0.0,
        ),
    },
)

# The vendor camera links are ROS optical frames (+Z = optical axis, +X = image
# right, +Y = image down): camera_head_*_link rpy=(-1.92, 0, -1.57) relative to
# head_link and *_d405_link rpy=(2.44, 0, -1.57) relative to the gripper link.
# Hence convention="ros" with an identity offset (an "opengl" offset would look
# along -Z, i.e. away from the scene).
GALAXEA_R1_PRO_HEAD_CAMERA_CFG = CameraCfg(
    prim_path="/World/envs/env_.*/Robot/camera_head_left_link/head_cam",
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
    offset=CameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        convention="ros",
    ),
)

GALAXEA_R1_PRO_HAND_CAMERA_CFG = CameraCfg(
    prim_path="/World/envs/env_.*/Robot/left_d405_link/left_hand_cam",
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
    offset=CameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        convention="ros",
    ),
)
