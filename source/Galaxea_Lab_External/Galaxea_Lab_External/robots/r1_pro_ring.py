"""G1Z tool calibration for R1Pro outer-ring feedback assembly."""

from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from .gearbox_ring import GearboxRingMixin


class R1ProRingMixin(GearboxRingMixin):
    # Pro's shoulder and chest geometry differ from Lite. Keep its calibrated
    # torso pose for this path; do not apply Lite's torso motion unconditionally.
    RING_RETRACT_TORSO = False
    # A shifted carrier can require a 380 mm transfer at 40 mm/s. Allow
    # time to settle after that travel, retaining the original stable gate.
    RING_TRANSFER_TIMEOUT_S = 15.0
    # Allow slight settling after the 1.5 mm insertion alignment gate. Keep
    # this tighter than the unchanged 5 mm final ring scoring tolerance, and
    # retain the separate drift, tilt, and post-release retention checks.
    RING_HOLD_XY_TOLERANCE = 0.002

    def _start_ring(self):
        self._ring_forward_lift_token = None
        self._ring_forward_lift_offset = 0.0
        return super()._start_ring()

    def _pickup_lift_target(self, arm, ee, gear, elapsed):
        target = super()._pickup_lift_target(arm, ee, gear, elapsed)
        if not getattr(self, "_ring_active", False) or self._planetary_state != "lift":
            return target
        token = (self._planetary_gear, self._pick_attempt)
        if getattr(self, "_ring_forward_lift_token", None) != token:
            self._ring_forward_lift_token = token
            self._ring_forward_lift_offset = 0.0
        robot = self.scene["robot"]
        joints = robot.data.joint_pos[:, arm.joint_ids]
        limits = robot.data.joint_pos_limits[:, arm.joint_ids]
        # Joint four's lower bound is the elbow's folding limit. Once the
        # payload is clear of the table, a bounded forward waypoint moves it
        # out of that minimum-reach region. Keep the original lift/pose gates.
        elbow_margin = joints[:, 3] - limits[:, 3, 0]
        if (
            self._ring_forward_lift_offset == 0.0
            and elapsed >= 1.0
            and float(gear[0, 2]) > self._pick_height + 0.030
            and float((target - ee[:, :3]).norm()) > 0.008
            and float(elbow_margin.min()) < 0.008
        ):
            self._ring_forward_lift_offset = 0.030
            self._stable_since = None
        target[:, 0] += self._ring_forward_lift_offset
        return target

    def _choose_grasp(self):
        if getattr(self, "_ring_active", False):
            self._ring_grasp_orientation = None
        return super()._choose_grasp()

    def _ring_target_transform(self, ee, ring, level):
        # The boss can settle in the pads during transport and phase alignment
        # above the teeth. Use that settled grasp, then freeze it for insertion:
        # re-measuring under tooth contact ratchets G1Z wrist tilt and grip slip.
        if (
            getattr(self, "_planetary_state", None) in ("transfer", "approach")
            or getattr(self, "_ring_grasp_orientation", None) is None
        ):
            self._ring_grasp_orientation = quat_mul(quat_conjugate(ring[:, 3:7]), ee[:, 3:7]).clone()
        orientation = quat_mul(level, self._ring_grasp_orientation)
        correction = quat_mul(orientation, quat_conjugate(ee[:, 3:7]))
        return orientation, quat_apply(correction, ee[:, :3] - ring[:, :3])

    def _ring_tip_to_tcp_height(self):
        # The converted G1Z pad box ends 120.05 mm below its finger origin.
        # With the 127 mm virtual TCP extension the tips are 6.95 mm above TCP.
        return self.FINGERTIP_EXTENSION - 0.12005
