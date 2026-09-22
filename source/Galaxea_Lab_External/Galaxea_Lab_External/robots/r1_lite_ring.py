"""R1 Lite tool calibration for the shared outer-ring feedback controller."""

from .gearbox_ring import GearboxRingMixin


class R1LiteRingMixin(GearboxRingMixin):
    def _ring_tip_to_tcp_height(self):
        return self.FINGERTIP_EXTENSION - self.FINGER_TIP_X
