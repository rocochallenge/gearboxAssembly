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

The assembly policy uses feedback-controlled pickup/insertion for the first
three gears (``R1ProPlanetaryMixin``), then uses a
high-rim grasp and rotational meshing search for the central gear
(``R1ProSunMixin``), then feedback ring insertion with G1Z grasp calibration
(``R1ProRingMixin``). The mounting plan, TCP calibration and per-arm DLS
controllers come from R1_Lite. Recovery
tasks retain their separate R1_Lite recovery sequence.
"""

import itertools

import torch
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from .r1_lite_rule_policy import R1LiteRulePolicy
from .r1_lite_recovery_rule_policy import R1LiteRecoveryRulePolicy
from .r1_pro_planetary import R1ProPlanetaryMixin
from .r1_pro_sun import R1ProSunMixin
from .r1_pro_ring import R1ProRingMixin


class R1ProRulePolicy(R1ProRingMixin, R1ProSunMixin, R1ProPlanetaryMixin, R1LiteRulePolicy):
    # Switch to the bounded contact search above the planetary teeth. A
    # 25 mm approach already pushes into contact before that search starts.
    SUN_APPROACH_HEIGHT_M = 0.040
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
    # actuator velocity cap over one 0.05 s control step), plus 25 % longer phases
    # for the legacy stages after the three planetary gears.
    # Emulating the sim's PD lag offline: a 0.5 step fraction left ~1 cm at grasp
    # within the phase time, the clamp does not. Total timetable stays under the
    # original 60 s timetable. The feedback assembly policy has a larger budget.
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

    def _assign_planetary_pins(self):
        super()._assign_planetary_pins()
        # Preserve the reach assignment, then aim at the actual carrier mesh
        # axes. The nominal front anchors are offset by about 0.39 mm.
        # Shared scoring geometry and its tolerances remain unchanged.
        carrier = self.planetary_carrier.data.root_state_w[:, :7]
        axes = carrier.new_tensor((
            (-0.0000000058710575, -0.0539999945163727, 0.0),
            (0.0467653796672821, 0.0270000021457672, 0.0),
            (-0.0467653832435608, 0.0269999965429306, 0.0),
        ))
        for gear_id in range(1, 4):
            assignment = self.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]
            local = axes[assignment["pin"]].clone()
            assignment["pin_local_pos"] = local
            assignment["pin_world_pos"] = carrier[:, :3] + quat_apply(carrier[:, 3:7], local.unsqueeze(0))

    def _start_planetary(self):
        self._planetary_phase_token = None
        return super()._start_planetary()

    def _use_free_planetary_yaw(self):
        gear_id = getattr(self, "_planetary_gear", None)
        if gear_id in (2, 3):
            token = (gear_id, self._pick_attempt)
            if getattr(self, "_planetary_phase_token", None) == token:
                return False
        return super()._use_free_planetary_yaw()

    def _planetary_insertion_yaw(self, yaw, pin):
        if self._planetary_gear not in (2, 3):
            return yaw, True
        token = (self._planetary_gear, self._pick_attempt)
        if getattr(self, "_planetary_phase_token", None) != token:
            xy, z, tilt = self._gear_errors(self._planetary_gear)
            if (
                self._planetary_state == "transfer"
                and xy < 0.002
                and abs(z - self.PLANETARY_TRANSFER_CLEARANCE) < 0.004
                and tilt < 0.04
            ):
                # Fixing yaw across the whole transfer saturates Pro's wrist.
                # Keep travel free, then prepare compatible teeth while the
                # gear is clear above its pin. Retain phase during insertion.
                self._planetary_phase_token = token
            else:
                return yaw, False
        first = self._held_object(1).data.root_state_w
        centre = self.planetary_carrier.data.root_state_w[:, :3]
        q = first[:, 3:7]
        first_yaw = torch.atan2(
            2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
            1 - 2 * (q[:, 2].square() + q[:, 3].square()),
        )
        first_delta = first[:, :2] - centre[:, :2]
        delta = pin[:, :2] - centre[:, :2]
        requested = first_yaw + 2 * (
            torch.atan2(delta[:, 1], delta[:, 0]) - torch.atan2(first_delta[:, 1], first_delta[:, 0])
        )
        # Twelve teeth: choose the nearest equivalent phase, with a bounded
        # correction. The first mounted planet is the live phase reference.
        difference = 12 * (requested - yaw)
        error = torch.atan2(difference.sin(), difference.cos()) / 12
        return yaw + error.clamp(-0.05, 0.05), float(error.abs()) < 0.015

    def _start_sun(self):
        self._sun_grasp_orientation = None
        self._sun_touch_wrist_z = None
        self._sun_realign_reanchor_token = None
        return super()._start_sun()

    def _sun_target_transform(self, ee, gear, level):
        state = self._planetary_state
        if state == "realign":
            token = (self._planetary_gear, self._insert_attempt, self._state_started)
            if (
                float(gear[0, 2] - self._live_pin(4)[0, 2]) > 0.050
                and getattr(self, "_sun_realign_reanchor_token", None) != token
            ):
                # Contact may permanently shift the shallow grasp. Observe
                # that change once the gear has lifted clear, then freeze it
                # again before approaching the teeth.
                self._sun_grasp_orientation = quat_mul(quat_conjugate(gear[:, 3:7]), ee[:, 3:7]).clone()
                self._sun_realign_reanchor_token = token
        # Preserve the free-space grasp tilt. Re-measuring it under blocked
        # teeth adopts wrist deflection and tilts the pads toward the planets.
        if state == "transfer" or getattr(self, "_sun_grasp_orientation", None) is None:
            self._sun_grasp_orientation = quat_mul(quat_conjugate(gear[:, 3:7]), ee[:, 3:7]).clone()
        orientation = quat_mul(level, self._sun_grasp_orientation)
        if state in ("approach", "phase_align", "realign", "search", "mesh_hold"):
            # Yaw can slip inside the shallow grasp. Apply the measured tooth
            # correction to the current wrist yaw; a frozen yaw relationship
            # can otherwise reverse that correction and keep spinning the gear.
            def yaw(q):
                return torch.atan2(
                    2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                    1 - 2 * (q[:, 2].square() + q[:, 3].square()),
                )

            phase_error = yaw(level) - yaw(gear[:, 3:7])
            phase_error = torch.atan2(phase_error.sin(), phase_error.cos())
            turn_angle = yaw(ee[:, 3:7]) + phase_error - yaw(orientation)
            turn = torch.zeros_like(orientation)
            turn[:, 0], turn[:, 3] = torch.cos(turn_angle / 2), torch.sin(turn_angle / 2)
            orientation = quat_mul(turn, orientation)
        correction = quat_mul(orientation, quat_conjugate(ee[:, 3:7]))
        return orientation, quat_apply(correction, ee[:, :3] - gear[:, :3])

    def _planetary_command(self, arm, gripper, position, orientation, opening, dt, speed=0.08, contact=False):
        if getattr(self, "_sun_active", False):
            if self._planetary_state in ("transfer", "approach", "realign"):
                self._sun_touch_wrist_z = None
            elif contact and getattr(self, "_sun_touch_started", False):
                # Bound preload from the wrist height at first tooth contact.
                # Reapplying it below a deflected wrist would otherwise keep
                # lowering the fingers as the shallow grasp slips or tilts.
                actual = self.scene["robot"].data.body_state_w[:, arm.body_ids[0], :3]
                if getattr(self, "_sun_touch_wrist_z", None) is None:
                    self._sun_touch_wrist_z = actual[:, 2].clone()
                position = position.clone()
                position[:, 2] = torch.maximum(position[:, 2], self._sun_touch_wrist_z - self.SUN_PRELOAD_M)
        return super()._planetary_command(arm, gripper, position, orientation, opening, dt, speed, contact)

    def _pickup_clearance(self):
        # A 100 mm planetary hover can lie inside Pro's minimum folding
        # radius. The 50 mm path still clears the 30 mm pins and loose gears.
        if getattr(self, "_planetary_gear", None) in (1, 2, 3):
            return 0.050
        return super()._pickup_clearance()

    def _planetary_target_transform(self, ee, gear, level, pin):
        orientation, offset = super()._planetary_target_transform(ee, gear, level, pin)
        state = self._planetary_state
        # Recalibrate only in free space. Under peg contact a shrinking
        # wrist-to-gear gap must not keep moving the insertion target down.
        clear_realign = state == "realign" and float(gear[0, 2] - pin[0, 2]) > 0.050
        if state == "transfer" or clear_realign or getattr(self, "_planetary_depth_gear", None) != self._planetary_gear:
            self._planetary_grasp_offset_z = offset[:, 2].clone()
            self._planetary_depth_gear = self._planetary_gear
        else:
            offset[:, 2] = self._planetary_grasp_offset_z
        return orientation, offset

    def _planetary_action(self):
        gear_id = getattr(self, "_planetary_gear", None)
        if (
            getattr(self, "_planetary_state", None) == "retry_pick"
            and gear_id in (1, 2, 3)
            and all(self._seated(i) for i in range(1, gear_id + 1))
        ):
            # A slipping grasp can leave the gear correctly seated. Open at
            # the observed pose and use the normal retention checks before
            # advancing, instead of reaching back into the mounted gears.
            side = self.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]["arm"]
            arm = getattr(self, f"{side}_arm_entity_cfg")
            ee = self.scene["robot"].data.body_state_w[:, arm.body_ids[0], :7]
            self._release_position = ee[:, :3].clone()
            self._release_orientation = ee[:, 3:7].clone()
            self._transition("release")
        return super()._planetary_action()

    def _open_pad_clearance(self, ee, opening, gear_count=3):
        # Converted G1Z pad boxes in link7, including both finger-joint
        # origins. Both pads share the gripper mount's -29.5 mm X offset.
        corners = [
            (-0.0295 + dx, side * (0.016897 + opening) + dy, -0.2215 + dz)
            for side in (-1.0, 1.0)
            for dx, dy, dz in itertools.product((-0.028, 0.028), (-0.01575, 0.01575), (-0.05005, 0.05005))
        ]
        local = torch.tensor(corners, device=self.device, dtype=ee.dtype)
        lowest_pad = ee[0, 2] + quat_apply(ee[:, 3:7].expand(len(corners), -1), local)[:, 2].min()
        gears = torch.cat([self._held_object(i).data.root_state_w for i in range(1, gear_count + 1)])
        up = quat_apply(
            gears[:, 3:7],
            torch.tensor([[0.0, 0.0, 1.0]], device=self.device, dtype=ee.dtype).expand(gear_count, -1),
        )
        highest_gear = (gears[:, 2] + 0.024 * up[:, 2] + 0.0315 * up[:, :2].norm(dim=-1)).max()
        return float(lowest_pad - highest_gear)

    def _release_action(self, arm, gripper, ee, gear, pin, down, elapsed, dt):
        # The third mount is crowded by two seated planets. Preserve the
        # earlier release motion, which establishes the carrier's approach pose.
        staged = self._planetary_gear == 3 and self._planetary_state in ("release", "retreat", "retry_pick")
        if staged:
            token = (self._planetary_gear, self._pick_attempt)
            if getattr(self, "_staged_jaw_token", None) != token:
                actual = self.scene["robot"].data.joint_pos[:, gripper.joint_ids]
                normal_opening = float(getattr(self, "_planetary_opening", self.GRIPPER_OPEN_POS))
                # Latch once, rather than adding 3 mm on every control tick.
                self._staged_jaw_narrow = min(normal_opening, float(actual[0, 0]) + 0.003)
                self._staged_jaw_token = token
                self._staged_jaw_full = False
        if self._planetary_state == "retreat" and self._planetary_gear in (1, 2, 3, 4) and elapsed >= 2.5:
            delta = ee[:, :3] - self._release_position
            target = self._release_position.clone()
            target[:, 2] += 0.10
            robot = self.scene["robot"]
            joints = robot.data.joint_pos[:, arm.joint_ids]
            limits = robot.data.joint_pos_limits[:, arm.joint_ids]
            margin = torch.minimum(joints - limits[..., 0], limits[..., 1] - joints)
            lateral = float(delta[:, :2].norm())
            clear_lateral = lateral < 0.012
            if self._planetary_gear == 4 and lateral < 0.030 and not clear_lateral:
                opening = max(float(getattr(self, "_planetary_opening", self.GRIPPER_OPEN_POS)), self.GRIPPER_OPEN_POS)
                # The central wrist can saturate after lifting fully clear.
                # Require the entire opened-pad envelope above all four
                # mounted gears before accepting a larger lateral residual.
                clear_lateral = self._open_pad_clearance(ee, opening, gear_count=4) > 0.030
            # A wrist limit may leave a few millimetres of lateral error
            # after the pads have lifted clear. Only this arm-limited case
            # can use the alternate gate; seating must still verify below.
            if (
                float((target - ee[:, :3]).norm()) >= 0.005
                and float(delta[0, 2]) >= 0.085
                and clear_lateral
                and float(margin.min()) < 0.010
                and all(self._seated(i) for i in range(1, self._planetary_gear + 1))
            ):
                self._transition("verify")
                elapsed = 0.0
        action, ids = super()._release_action(arm, gripper, ee, gear, pin, down, elapsed, dt)
        if staged and action is not None and ids is not None:
            normal_opening = float(getattr(self, "_planetary_opening", self.GRIPPER_OPEN_POS))
            if self._open_pad_clearance(ee, normal_opening) > 0.002:
                self._staged_jaw_full = True
            if not self._staged_jaw_full:
                # Release the payload, then lift before opening the long
                # fingers fully between neighbouring planetary gears.
                action = action.clone()
                for joint_id in gripper.joint_ids:
                    action[:, ids.index(joint_id)] = self._staged_jaw_narrow
        return action, ids


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
