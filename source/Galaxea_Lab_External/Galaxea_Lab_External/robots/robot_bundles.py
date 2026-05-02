# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-selection bundles.

A ``RobotBundle`` groups everything that varies per-robot (articulation
config, camera prim paths, joint dof-name strings, rule-policy classes)
into one frozen dataclass so each task env_cfg can read a single source
of truth. Switching between R1 and R1_Lite is one symbol edit:

    ACTIVE_ROBOT_BUNDLE = GALAXEA_R1_LITE_BUNDLE  # was GALAXEA_R1_BUNDLE

The two non-recovery env_cfgs read ``ACTIVE_ROBOT_BUNDLE.rule_policy_class``
for their rule-policy reference; the gearbox-recovery env_cfg reads
``ACTIVE_ROBOT_BUNDLE.recovery_rule_policy_class``.
"""

from dataclasses import dataclass

from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import CameraCfg

from .galaxea_robots import (
    GALAXEA_R1_CHALLENGE_CFG,
    GALAXEA_R1_LITE_CFG,
    GALAXEA_HEAD_CAMERA_CFG,
    GALAXEA_HAND_CAMERA_CFG,
    GALAXEA_R1_LITE_HEAD_CAMERA_CFG,
    GALAXEA_R1_LITE_HAND_CAMERA_CFG,
)
from .galaxea_rule_policy import GalaxeaRulePolicy
from .recovery_rule_policy import RecoveryRulePolicy
from .r1_lite_rule_policy import R1LiteRulePolicy
from .r1_lite_recovery_rule_policy import R1LiteRecoveryRulePolicy


def _torso_pos_from_cfg(cfg: ArticulationCfg) -> tuple[float, ...]:
    """Return torso joint positions from an articulation cfg's
    ``init_state.joint_pos``, ordered by the trailing index.

    R1 has four torso joints (``torso_joint1..4``), R1_Lite has three
    (``torso_joint1..3``); we return whatever's declared so the bundle's
    ``initial_torso_pos`` stays in sync regardless of the chain length.
    """
    jp = cfg.init_state.joint_pos
    keys = sorted(
        (k for k in jp if k.startswith("torso_joint")),
        key=lambda k: int(k[len("torso_joint"):]),
    )
    return tuple(jp[k] for k in keys)


@dataclass(frozen=True)
class RobotBundle:
    """Per-robot configuration aggregate.

    ``articulation_cfg``'s ``prim_path`` is overridden per env_cfg via
    ``.replace(prim_path=...)``, which returns a copy and does not
    mutate this frozen instance. The three camera cfgs are pre-set to
    the robot's frame paths.
    """

    name: str
    articulation_cfg: ArticulationCfg
    head_camera_cfg: CameraCfg
    left_hand_camera_cfg: CameraCfg
    right_hand_camera_cfg: CameraCfg
    left_arm_joint_pattern: str
    right_arm_joint_pattern: str
    left_gripper_dof_name: str
    right_gripper_dof_name: str
    gripper_collision_link_names: tuple[str, ...]
    torso_joint_pattern: str
    initial_torso_pos: tuple[float, ...]
    rule_policy_class: type
    recovery_rule_policy_class: type


GALAXEA_R1_BUNDLE = RobotBundle(
    name="r1",
    articulation_cfg=GALAXEA_R1_CHALLENGE_CFG,
    head_camera_cfg=GALAXEA_HEAD_CAMERA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/zed_link/head_cam/head_cam",
    ),
    left_hand_camera_cfg=GALAXEA_HAND_CAMERA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/left_realsense_link/left_hand_cam/left_hand_cam",
    ),
    right_hand_camera_cfg=GALAXEA_HAND_CAMERA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/right_realsense_link/right_hand_cam/right_hand_cam",
    ),
    left_arm_joint_pattern="left_arm_joint.*",
    right_arm_joint_pattern="right_arm_joint.*",
    left_gripper_dof_name="left_gripper_axis1",
    right_gripper_dof_name="right_gripper_axis1",
    gripper_collision_link_names=(
        "left_gripper_link1",
        "left_gripper_link2",
        "right_gripper_link1",
        "right_gripper_link2",
    ),
    torso_joint_pattern="torso_joint[1-4]",
    # R1's USD (r1_DVT_colored_cam_pos.usd) has torso joint limits baked
    # to [0, 0], so the cfg's init_state.joint_pos must be zero — we
    # can't derive this from cfg via _torso_pos_from_cfg like R1_Lite does.
    # Runtime _reset_idx widens the limits and writes this target.
    initial_torso_pos=(0.5, -0.8, 0.5, 0.0),
    rule_policy_class=GalaxeaRulePolicy,
    recovery_rule_policy_class=RecoveryRulePolicy,
)

GALAXEA_R1_LITE_BUNDLE = RobotBundle(
    name="r1_lite",
    articulation_cfg=GALAXEA_R1_LITE_CFG,
    head_camera_cfg=GALAXEA_R1_LITE_HEAD_CAMERA_CFG,
    left_hand_camera_cfg=GALAXEA_R1_LITE_HAND_CAMERA_CFG,
    right_hand_camera_cfg=GALAXEA_R1_LITE_HAND_CAMERA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/right_D405_link/right_hand_cam",
    ),
    left_arm_joint_pattern="left_arm_joint.*",
    right_arm_joint_pattern="right_arm_joint.*",
    left_gripper_dof_name="left_gripper_finger_joint1",
    right_gripper_dof_name="right_gripper_finger_joint1",
    gripper_collision_link_names=(
        "left_gripper_link",
        "left_gripper_finger_link1",
        "left_gripper_finger_link2",
        "right_gripper_link",
        "right_gripper_finger_link1",
        "right_gripper_finger_link2",
    ),
    torso_joint_pattern="torso_joint[1-3]",
    initial_torso_pos=_torso_pos_from_cfg(GALAXEA_R1_LITE_CFG),
    rule_policy_class=R1LiteRulePolicy,
    recovery_rule_policy_class=R1LiteRecoveryRulePolicy,
)


# The single switch. Edit this line to flip the active robot for all tasks.
ACTIVE_ROBOT_BUNDLE: RobotBundle = GALAXEA_R1_LITE_BUNDLE
# ACTIVE_ROBOT_BUNDLE: RobotBundle = GALAXEA_R1_BUNDLE
