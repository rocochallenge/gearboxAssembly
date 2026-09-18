"""Shared geometric anchors for the gearbox assembly.

The carrier USD contains the three pins as part of a single rigid mesh; they
are not separate rigid bodies that Isaac Lab can query at runtime.  These
values therefore describe the pin centres in the carrier's *local* frame.
They must always be transformed with the carrier pose read from the live
simulation state before being used as a world-space target or criterion.
"""

# Carrier-local pin centres (metres), ordered by their geometry only.  These
# are not world-space coordinates and must not be used without the current
# planetary-carrier transform.
PLANETARY_PIN_LOCAL_POSITIONS: tuple[tuple[float, float, float], ...] = (
    (0.0, -0.054, 0.0),
    (0.0471, 0.0268, 0.0),
    (-0.0471, 0.0268, 0.0),
)

# The central hole is ~25 mm in radius, while the sun teeth reach 31.5 mm.
# Their lower faces rest on the carrier plate at local Z = 9 mm, not at its
# origin. These dimensions are verified against the committed USD meshes.
CENTRE_GEAR_LOCAL_POSITION = (0.0, 0.0, 0.009)


def centre_gear_seat_position(carrier_position, carrier_quaternion):
    """World position of a seated sun gear's root, following the live carrier."""
    from isaaclab.utils.math import quat_apply

    local = carrier_position.new_tensor(CENTRE_GEAR_LOCAL_POSITION).expand_as(carrier_position)
    return carrier_position + quat_apply(carrier_quaternion, local)
