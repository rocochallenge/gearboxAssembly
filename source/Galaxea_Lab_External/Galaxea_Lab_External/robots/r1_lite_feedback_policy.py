"""Feedback assembly for R1 Lite's six-axis arms and +X-facing grippers.

Reuse the verified pickup, insertion and meshing state machines, with R1 Lite
tool geometry and grasp selection. The legacy R1LiteRulePolicy remains separate
for comparisons and as R1Pro's base; no R1Pro frame constants are substituted.
Central-gear placement remains experimental and can fail during meshing or release.
"""

import math

import torch
from isaaclab.utils.math import quat_apply, quat_conjugate

from .r1_lite_rule_policy import R1LiteRulePolicy
from .r1_pro_planetary import R1ProPlanetaryMixin
from .r1_pro_sun import R1ProSunMixin


class R1LiteFeedbackPolicy(R1ProSunMixin, R1ProPlanetaryMixin, R1LiteRulePolicy):
    ASSEMBLY_NAME = "R1 Lite"
    IK_MAX_JOINT_STEP = 0.3
    # Six-axis Lite can retain wrist yaw across this workspace; keep that
    # constraint to limit changes in the grasp during transport.
    PLANETARY_FREE_YAW = False
    # At the rear pin the 80 mm path puts the carried gear into the torso.
    # This clears the 30 mm pins while keeping the gear below the chest.
    PLANETARY_TRANSFER_CLEARANCE = 0.045
    # Retain Lite's stock jaw ceiling for the shallow rim grasp. The 8 N
    # Pro setting lost this grasp in Lite trials; axial preload stays bounded.
    SUN_GRIP_EFFORT_N = 100.0
    GRIPPER_OPEN_POS = 0.05
    ROTATE_VIA_IK = True
    MOUNT_USES_IN_HAND_OFFSET = True
    PICK_USES_CURRENT_POSE = True
    # The STL tip is at finger-link +X=0.041465 m. Move the virtual TCP
    # beyond the tip so the shared pickup target leaves table clearance.
    FINGERTIP_EXTENSION = 0.055
    FINGER_TIP_X = 0.041465
    FINGER_PAD_OUTER_Y = 0.008327
    FINGER_PAD_HALF_Z = 0.007074
    # With +90 degrees about Y, local Z lies horizontally in world X.
    GRASP_SHIFT_AXIS_LOCAL = (0.0, 0.0, 1.0)

    def _choose_grasp(self):
        """Check the narrow Lite pad sweep against nearby tabletop parts.

        The finger URDF origins are (+0.03689, +/-0.013453, +/-0.00012059)
        in the gripper frame. At the distal 13 mm, the mesh gives an 8.327 mm
        outer pad offset from jaw position and a 7.074 mm half-width in Z.
        """
        gear_name = f"sun_planetary_gear_{self._planetary_gear}"
        gear = self.obj_dict[gear_name].data.root_state_w[0, :3].tolist()
        obstacles = []
        for name, obj in self.obj_dict.items():
            if name == gear_name:
                continue
            position = obj.data.root_state_w[0, :3].tolist()
            if abs(position[2] - gear[2]) > 0.05:
                continue
            radius = {"ring_gear": 0.10, "planetary_carrier": 0.07, "planetary_reducer": 0.04}.get(name, 0.032)
            obstacles.append((position[0] - gear[0], position[1] - gear[1], radius))

        candidates = []
        for degrees in range(-90, 91, 5):
            yaw = math.radians(degrees)
            c, s = math.cos(yaw), math.sin(yaw)
            for opening in (0.05, 0.045, 0.04, 0.036):
                for shift in (0.0, -0.008, 0.008):
                    clearance = float("inf")
                    outer = opening + self.FINGER_PAD_OUTER_Y
                    for dx, dy, radius in obstacles:
                        x, y = c * dx + s * dy - shift, -s * dx + c * dy
                        for low, high in ((0.030, outer), (-outer, -0.030)):
                            gap = math.hypot(
                                max(abs(x) - self.FINGER_PAD_HALF_Z, 0.0), max(low - y, 0.0, y - high)
                            ) - radius
                            clearance = min(clearance, gap)
                    if clearance >= 0.006:
                        cost = abs(degrees) + (self.GRIPPER_OPEN_POS - opening) * 2500 + abs(shift) * 1000
                        candidates.append((cost, yaw, opening, shift))
        if not candidates:
            return False
        _, self._grasp_yaw_delta, self._planetary_opening, self._grasp_shift = min(candidates)
        print(
            f"[R1 Lite assembly] grasp yaw={math.degrees(self._grasp_yaw_delta):.0f}deg "
            f"jaw={self._planetary_opening:.3f} shift={self._grasp_shift:.3f}"
        )
        return True

    def _pickup_height_offset(self):
        if getattr(self, "_sun_active", False):
            # Leave the tips 18 mm above the root, gripping the upper 6 mm of
            # the rim so the fingers clear the planets during partial meshing.
            return 0.018 - (self.FINGERTIP_EXTENSION - self.FINGER_TIP_X)
        return super()._pickup_height_offset()

    def _pickup_clearance(self):
        # With the Lite torso pose, 35 mm clears the loose gears while keeping
        # a far-edge, midline pickup below wrist joint4's +90 degree limit.
        return 0.035

    def _retry_pick(self, ee, reason):
        result = super()._retry_pick(ee, reason)
        if self._planetary_state == "retry_pick":
            self._retreat_position[:, 2] = ee[:, 2] + self._pickup_clearance()
        return result

    def _sun_fingertip_corners(self, arm):
        local_tcp = quat_apply(quat_conjugate(self._gripper_down_quat(arm)), -self._tcp_offset(arm))
        gap = self.FINGERTIP_EXTENSION - self.FINGER_TIP_X
        half_width = self.GRIPPER_OPEN_POS + self.FINGER_PAD_OUTER_Y
        return local_tcp + torch.tensor(
            [[-gap, y, z] for y in (-half_width, half_width)
             for z in (-self.FINGER_PAD_HALF_Z, self.FINGER_PAD_HALF_Z)],
            device=self.device,
        )
