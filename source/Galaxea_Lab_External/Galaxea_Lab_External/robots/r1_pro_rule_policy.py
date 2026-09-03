"""Rule policies for the Galaxea R1Pro (2026, G1Z gripper).

R1Pro differs from R1_Lite in three ways that matter to the rule policies:

* **7-DOF arms.** The IK end-effector is ``*_arm_link7`` (the link the gripper
  bolts onto), not ``*_arm_link6``. ``DifferentialIKController`` handles the
  extra column of the Jacobian without changes.
* **Gripper extends along link7 -Z** (URDF ``*_gripper_joint`` origin
  ``(-0.0295, 0, -0.1637)``, fingers reach down to z ~= -0.11 in the gripper
  frame). "Gripper down" is therefore link7 -Z = world -Z with a free yaw; we
  use a mirrored yaw of -/+135 deg (left/right) so link7 +X points inward and
  slightly back, which keeps the redundant wrist away from its limits. R1_Lite uses +90 deg about Y
  for its link6 +X gripper. Fingers open along link7 +Y in both robots.
* **No single wrist joint spins the gripper.** The wrist is Z-Y-X
  (``joint5``, ``joint6``, ``joint7``) while the gripper axis is link7 -Z, so
  the tooth-meshing wiggle is done by rotating the IK target about the vertical
  TCP axis (``ROTATE_VIA_IK``) instead of nudging one joint.

Everything else (mounting plan, phase timings, TCP auto-tune from finger
geometry, per-arm DLS controllers) is inherited from the R1_Lite policies.
"""

from .r1_lite_rule_policy import R1LiteRulePolicy
from .r1_lite_recovery_rule_policy import R1LiteRecoveryRulePolicy


class R1ProRulePolicy(R1LiteRulePolicy):
    EE_LINK_SUFFIX = "_arm_link7"
    # Finger joint 2 mimics joint 1 (URDF <mimic>, as for R1_Lite): drive joint 1 only.
    GRIPPER_JOINT_SUFFIX = "_gripper_finger_joint1"
    GRIPPER_AXIS_LOCAL = (0.0, 0.0, -1.0)
    # "Gripper down" with link7 +X yawed toward the body midline and slightly back:
    # yaw -135 deg for the left arm, +135 deg for the right. Offline DLS emulation
    # on the URDF (PD lag + velocity cap modelled) showed the redundant arm tracks
    # every pick/lift/swing/descend target to ~1 mm at this yaw, while 0/180 deg
    # stall 0.3 m short on the mount swing and +-90 deg leaves 1-4 cm on gears
    # spawning straight ahead of a shoulder.
    GRIPPER_DOWN_QUAT = (0.3826834, 0.0, 0.0, -0.9238795)
    GRIPPER_DOWN_QUAT_RIGHT = (0.3826834, 0.0, 0.0, 0.9238795)
    ROTATE_VIA_IK = True
    # Full DLS steps, but capped at 0.5 rad per joint per update (= the 10 rad/s
    # actuator velocity cap over one 0.05 s control step), plus 25 % longer phases.
    # Emulating the sim's PD lag offline: a 0.5 step fraction left ~1 cm at grasp
    # within the phase time, the clamp does not. Total timetable stays under the
    # 60 s episode limit (45 s * 1.25).
    IK_STEP_FRACTION = 1.0
    IK_MAX_JOINT_STEP = 0.5
    PHASE_TIME_SCALE = 1.25
    # Open wide: the converted G1Z has a 0.065 m stroke per finger; 0.055 gives an
    # 11 cm jaw gap, i.e. +-2.5 cm of centring tolerance around a 6.2 cm gear
    # (R1_Lite's 0.04 left only +-1 cm and a pad regularly landed on the rim).
    GRIPPER_OPEN_POS = 0.055
    # Aim mounts with the measured in-hand offset: a held gear can sit a few cm
    # off the nominal TCP, which would otherwise miss the 2 mm pin tolerance.
    MOUNT_USES_IN_HAND_OFFSET = True
    # Pick where the gear is now, not where it was at reset (the other arm's swings
    # and the carrier nudge things around).
    PICK_USES_CURRENT_POSE = True
    # The G1Z fingers reach 0.1201 m below the finger-link origin (STL extent).
    # The policy puts the TCP at table_height + grasping_height = 0.905, but the
    # table top is really at z = 0.909, so the TCP has to sit ~7 mm *beyond* the
    # tips to keep them ~3 mm clear of the table (0.118 put them 6 mm into it,
    # which dragged the pads on the table during the close and let gears slip
    # out on the lift; 0.085 shoved gear and carrier across the table). The flat
    # pad colliders still overlap a 2.4 cm gear by ~2 cm.
    FINGERTIP_EXTENSION = 0.127


class R1ProRecoveryRulePolicy(R1LiteRecoveryRulePolicy):
    EE_LINK_SUFFIX = "_arm_link7"
    GRIPPER_JOINT_SUFFIX = "_gripper_finger_joint1"
    GRIPPER_AXIS_LOCAL = (0.0, 0.0, -1.0)
    GRIPPER_DOWN_QUAT = (0.3826834, 0.0, 0.0, -0.9238795)
    GRIPPER_DOWN_QUAT_RIGHT = (0.3826834, 0.0, 0.0, 0.9238795)
    ROTATE_VIA_IK = True
    IK_STEP_FRACTION = 1.0
    IK_MAX_JOINT_STEP = 0.5
    PHASE_TIME_SCALE = 1.25
    GRIPPER_OPEN_POS = 0.055
    MOUNT_USES_IN_HAND_OFFSET = True
    PICK_USES_CURRENT_POSE = True
    FINGERTIP_EXTENSION = 0.127
