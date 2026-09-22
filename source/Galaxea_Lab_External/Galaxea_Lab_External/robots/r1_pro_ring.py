"""G1Z tool calibration for R1Pro outer-ring feedback assembly."""

from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from .gearbox_ring import GearboxRingMixin


class R1ProRingMixin(GearboxRingMixin):
    # Pro's shoulder and chest geometry differ from Lite. Keep its calibrated
    # torso pose for this path; do not apply Lite's torso motion unconditionally.
    RING_RETRACT_TORSO = False
    # Allow slight settling after the 1.5 mm insertion alignment gate. Keep
    # this tighter than the unchanged 5 mm final ring scoring tolerance, and
    # retain the separate drift, tilt, and post-release retention checks.
    RING_HOLD_XY_TOLERANCE = 0.002

    def _choose_grasp(self):
        if getattr(self, "_ring_active", False):
            self._ring_grasp_orientation = None
        return super()._choose_grasp()

    def _ring_target_transform(self, ee, ring, level):
        # Preserve the wrist-to-ring orientation measured before insertion.
        # Updating it under tooth contact ratchets G1Z wrist tilt while the ring
        # stays level, shifts the grip, and can trigger a false slip recovery.
        if getattr(self, "_ring_grasp_orientation", None) is None:
            self._ring_grasp_orientation = quat_mul(quat_conjugate(ring[:, 3:7]), ee[:, 3:7]).clone()
        orientation = quat_mul(level, self._ring_grasp_orientation)
        correction = quat_mul(orientation, quat_conjugate(ee[:, 3:7]))
        return orientation, quat_apply(correction, ee[:, :3] - ring[:, :3])

    def _ring_tip_to_tcp_height(self):
        # The converted G1Z pad box ends 120.05 mm below its finger origin.
        # With the 127 mm virtual TCP extension the tips are 6.95 mm above TCP.
        return self.FINGERTIP_EXTENSION - 0.12005
