"""Feedback assembly for R1 Lite's six-axis arms and +X-facing grippers.

Reuse the verified pickup, insertion and meshing state machines, with R1 Lite
tool geometry and grasp selection. The legacy R1LiteRulePolicy remains separate
for comparisons and as R1Pro's base; no R1Pro frame constants are substituted.
The central gear and outer ring are checked again after release and parking.
"""

import math

import torch
from isaaclab.utils.math import quat_apply, quat_conjugate

from .r1_lite_rule_policy import R1LiteRulePolicy
from .r1_lite_ring import R1LiteRingMixin
from .r1_pro_planetary import R1ProPlanetaryMixin
from .r1_pro_sun import R1ProSunMixin


class R1LiteFeedbackPolicy(R1LiteRingMixin, R1ProSunMixin, R1ProPlanetaryMixin, R1LiteRulePolicy):
    ASSEMBLY_NAME = "R1 Lite"
    IK_MAX_JOINT_STEP = 0.3
    # Six-axis Lite can retain wrist yaw across this workspace; keep that
    # constraint to limit changes in the grasp during transport.
    PLANETARY_FREE_YAW = False
    # Keep 5 mm above the 30 mm pins. With the fixed table, a 45 mm path
    # catches the chest when transferring to the rear pin from the left.
    PLANETARY_TRANSFER_CLEARANCE = 0.035
    # Keep the hardware ceiling; bound the closing target relative to the
    # measured jaws below instead of driving through the light gear to zero.
    SUN_GRIP_EFFORT_N = 100.0
    SUN_JAW_SPEED_M_S = 0.008
    SUN_MAX_JAW_TARGET_ERROR_M = 0.0004
    # Engage the teeth while the upper-rim fingers still clear the planets.
    # The remaining drop to the 9 mm seat is only 16 mm, avoiding a tumble
    # from a high release. Verify seating after retreat and parking.
    SUN_APPROACH_HEIGHT_M = 0.025
    SUN_GRAVITY_SEATING = True
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

    def _planetary_command(self, arm, gripper, position, orientation, opening, dt, speed=0.08, contact=False):
        if getattr(self, "_sun_active", False) and opening == 0.0:
            actual = float(self.scene["robot"].data.joint_pos[0, gripper.joint_ids].mean())
            travel = min(self.SUN_JAW_SPEED_M_S * dt, self.SUN_MAX_JAW_TARGET_ERROR_M)
            opening = max(0.0, actual - travel)
        return super()._planetary_command(arm, gripper, position, orientation, opening, dt, speed, contact)

    def _pick_action(self, arm, gripper, ee, gear, pin, down, elapsed, dt):
        if getattr(self, "_sun_active", False) and self._planetary_state == "close":
            robot = self.scene["robot"]
            gap = float(robot.data.joint_pos[0, gripper.joint_ids].mean())
            velocity = float(robot.data.joint_vel[0, gripper.joint_ids].abs().max())
            # The 63 mm rim stops each jaw near 30 mm. A timer alone can
            # start lifting while the gradually closing fingers are still open.
            if self._stable(elapsed > 0.6 and 0.020 < gap < 0.033 and velocity < 0.003, 0.3):
                self._transition("lift")
            elif elapsed > 6.0:
                result = self._retry_pick(ee, "rim grasp did not close")
                if result is not None:
                    return result
                return self._release_action(arm, gripper, ee, gear, pin, down, 0.0, dt)
            return self._planetary_command(arm, gripper, self._grasp_position, down, 0.0, dt)
        return super()._pick_action(arm, gripper, ee, gear, pin, down, elapsed, dt)

    def _planetary_insertion_yaw(self, yaw, pin):
        """Make the three planets compatible with one central-gear phase.

        For equal external gears, planet yaw minus twice its bearing from
        the centre is constant modulo the 30-degree tooth pitch. Align the
        carried gear above its pin; the first mounted gear is the reference.
        """
        if self._planetary_gear == 1:
            return yaw, True
        first = self._held_object(1).data.root_state_w
        centre = self.planetary_carrier.data.root_state_w[:, :3]
        q = first[:, 3:7]
        first_yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                                1 - 2 * (q[:, 2].square() + q[:, 3].square()))
        first_delta = first[:, :2] - centre[:, :2]
        delta = pin[:, :2] - centre[:, :2]
        requested = first_yaw + 2 * (
            torch.atan2(delta[:, 1], delta[:, 0]) - torch.atan2(first_delta[:, 1], first_delta[:, 0])
        )
        difference = 12 * (requested - yaw)
        error = torch.atan2(difference.sin(), difference.cos()) / 12
        return yaw + error.clamp(-0.05, 0.05), float(error.abs()) < 0.015

    def _choose_grasp(self):
        """Check the narrow Lite pad sweep against nearby tabletop parts.

        The finger URDF origins are (+0.03689, +/-0.013453, +/-0.00012059)
        in the gripper frame. At the distal 13 mm, the mesh gives an 8.327 mm
        outer pad offset from jaw position and a 7.074 mm half-width in Z.
        """
        if getattr(self, "_ring_active", False):
            return super()._choose_grasp()
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
        if getattr(self, "_ring_active", False):
            return super()._pickup_clearance()
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
