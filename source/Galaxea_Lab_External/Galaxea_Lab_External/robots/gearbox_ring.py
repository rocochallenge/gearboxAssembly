"""Shared feedback pickup and insertion of the outer gearbox ring.

The ring is held by its raised central boss. Its lower internal teeth must
clear the mounted gears during transport, then mesh before releasing the boss.
All targets follow measured object poses; completion includes an empty-gripper
retreat and a final retention check. Robot adapters supply fingertip calibration
and select whether their transfer path needs torso retraction.
"""

import math

import torch
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul


class GearboxRingMixin:
    RING_TIMEOUT_S = 65.0
    RING_RETRACT_TORSO = True
    RING_TRANSFER_HEIGHT = 0.050
    RING_PRELOAD_M = 0.001
    RING_SEARCH_SPEED_RAD_S = math.radians(2.0)
    RING_SEARCH_SWEEP_RAD = math.radians(5.0)
    # Existing ring mesh: lid top 27.45 mm, boss top 47.45 mm from its root.
    # A 31 mm fingertip height clears the lid while engaging 16 mm of the boss.
    RING_GRASP_TIP_HEIGHT = 0.031
    RING_SEAT_HEIGHT = 0.009
    RING_HOLD_XY_TOLERANCE = 0.0015
    RING_TORSO_TARGET = -0.10
    RING_TORSO_MOVE_S = 3.0

    @property
    def assembly_complete(self):
        return bool(getattr(self, "ring_complete", False))

    def get_action(self):
        if not getattr(self, "sun_complete", False):
            return super().get_action()
        if self.assembly_complete:
            return None, None
        if not getattr(self, "_ring_active", False):
            self._start_ring()
        return self._ring_action()

    def _start_ring(self):
        self._ring_active = True
        self.ring_complete = False
        self._ring_started = self.count
        self._ring_disturbed_since = None
        self._ring_torso_retracted = not self.RING_RETRACT_TORSO
        self._planetary_gear = 5
        side = "left" if float(self.ring_gear.data.root_state_w[0, 1]) >= 0.0 else "right"
        self.gear_to_pin_map["ring_gear"] = {"arm": side}
        self._planetary_last_count = self.count
        self._pick_attempt = self._insert_attempt = 1
        self._motion_target = self._position_bias = self._orientation_bias = None
        self._grasp_yaw_delta = self._grasp_shift = 0.0
        self._planetary_opening = self.GRIPPER_OPEN_POS
        self.total_time_steps = int(self.EPISODE_LENGTH_S / self.sim_dt)
        self._transition("hover")

    def _choose_grasp(self):
        if getattr(self, "_ring_active", False):
            # The 70 mm boss fits between both robots' open jaws. The pads stay
            # above the lid, so the outer 195 mm rim is outside their sweep.
            self._grasp_yaw_delta = self._grasp_shift = 0.0
            self._planetary_opening = self.GRIPPER_OPEN_POS
            return True
        return super()._choose_grasp()

    def _pickup_height_offset(self):
        if getattr(self, "_ring_active", False):
            return self.RING_GRASP_TIP_HEIGHT - self._ring_tip_to_tcp_height()
        return super()._pickup_height_offset()

    def _pickup_clearance(self):
        if getattr(self, "_ring_active", False):
            return 0.050
        return super()._pickup_clearance()

    def _live_pin(self, gear_id):
        if gear_id == 5:
            return self.planetary_carrier.data.root_state_w[:, :3]
        return super()._live_pin(gear_id)

    def _seated(self, gear_id):
        if gear_id == 5:
            xy, z, tilt = self._gear_errors(5)
            return xy < 0.005 and abs(z) < 0.010 and tilt < 0.05
        return super()._seated(gear_id)

    def _ring_mesh_yaw(self, current_yaw, centre):
        """Closest 36-tooth internal-ring phase for the three 12-tooth planets.

        Internal gears turn in the same direction. The tooth-tip phase datum
        comes from the existing meshes (planet 22.17 deg, ring about 2.05 deg).
        This is an initial estimate; insertion still requires measured seating.
        """
        phases = []
        datum = math.radians(12 * 22.17 - 36 * 2.05 - 180)
        for i in range(1, 4):
            gear = self._held_object(i).data.root_state_w
            q = gear[:, 3:7]
            yaw = torch.atan2(
                2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2].square() + q[:, 3].square())
            )
            delta = gear[:, :2] - centre[:, :2]
            direction = torch.atan2(delta[:, 1], delta[:, 0])
            phases.append(24 * direction + 12 * yaw + datum)
        phase = torch.stack(phases)
        nearest = torch.atan2(phase.sin().sum(dim=0), phase.cos().sum(dim=0)) / 36
        difference = 36 * (nearest - current_yaw)
        return current_yaw + torch.atan2(difference.sin(), difference.cos()) / 36

    def _ring_action(self):
        if self._planetary_state == "failed":
            return None, None
        if (self.count - self._ring_started) * self.sim_dt > self.RING_TIMEOUT_S:
            return self._fail_planetary("ring time budget exhausted")
        state = self._planetary_state
        if all(self._seated(i) for i in range(1, 5)):
            self._ring_disturbed_since = None
        elif self._ring_disturbed_since is None:
            self._ring_disturbed_since = self.count
        elif (self.count - self._ring_disturbed_since) * self.sim_dt > 0.5:
            return self._fail_planetary("a mounted gear moved during ring placement")

        dt = max(self.sim_dt, (self.count - self._planetary_last_count) * self.sim_dt)
        self._planetary_last_count = self.count
        side = self.gear_to_pin_map["ring_gear"]["arm"]
        arm = getattr(self, f"{side}_arm_entity_cfg")
        gripper = getattr(self, f"{side}_gripper_entity_cfg")
        ee = self.scene["robot"].data.body_state_w[:, arm.body_ids[0], :7]
        ring = self.ring_gear.data.root_state_w
        centre = self._live_pin(5)
        down = self._gripper_down_quat(arm)
        elapsed = (self.count - self._state_started) * self.sim_dt
        if float(ring[0, 2]) < self.table_height - 0.05:
            return self._fail_planetary("ring left the table workspace")
        if state in ("hover", "descend", "close", "lift"):
            return self._pick_action(arm, gripper, ee, ring, centre, down, elapsed, dt)
        if state in ("transfer", "stage", "retract_torso", "approach", "search", "realign"):
            return self._ring_insert(arm, gripper, ee, ring, centre, elapsed, dt)
        if state == "mesh_hold":
            return self._ring_hold(arm, gripper, ee, ring, elapsed, dt)
        if state == "retry_pick":
            return self._release_action(arm, gripper, ee, ring, centre, down, elapsed, dt)
        if state in ("release", "retreat", "verify"):
            return self._ring_release(arm, gripper, ee, elapsed, dt)
        if state == "park":
            return self._ring_park(arm, gripper, side, elapsed)
        raise RuntimeError(f"Unknown ring state: {state}")

    def _ring_insert(self, arm, gripper, ee, ring, centre, elapsed, dt):
        state = self._planetary_state
        if float((ee[:, :3] - ring[:, :3] - self._tcp_offset(arm)).norm()) > 0.080:
            result = self._retry_pick(ee, "ring slipped during transport")
            return (
                result if result is not None else self._planetary_command(arm, gripper, ee[:, :3], ee[:, 3:7], 0.0, dt)
            )
        if state in ("transfer", "stage", "retract_torso") and not self._ring_torso_retracted:
            return self._ring_stage(arm, gripper, ee, ring, centre, elapsed, dt)
        xy, z, tilt = self._gear_errors(5)
        q = ring[:, 3:7]
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2].square() + q[:, 3].square()))
        desired_yaw = yaw
        if state == "approach":
            desired_yaw = self._ring_mesh_yaw(yaw, centre)
        elif state == "search":
            if elapsed - self._ring_depth_sample[0] >= 0.4:
                self._ring_touching = self._ring_depth_sample[1] - z < 0.0005
                self._ring_depth_sample = (elapsed, z)
            if self._ring_touching and z > 0.013:
                self._ring_search_offset += self._ring_search_direction * self.RING_SEARCH_SPEED_RAD_S * dt
                limit = self.RING_SEARCH_SWEEP_RAD * self._insert_attempt
                if abs(self._ring_search_offset) >= limit:
                    self._ring_search_offset = max(-limit, min(limit, self._ring_search_offset))
                    self._ring_search_direction *= -1
            requested = self._ring_yaw_anchor + self._ring_search_offset
            difference = torch.atan2(torch.sin(requested - yaw), torch.cos(requested - yaw))
            desired_yaw = yaw + difference.clamp(-0.025, 0.025)
        level = torch.zeros_like(q)
        level[:, 0], level[:, 3] = torch.cos(desired_yaw / 2), torch.sin(desired_yaw / 2)
        orientation, offset = self._ring_target_transform(ee, ring, level)

        if state in ("transfer", "realign"):
            height, speed = self.RING_TRANSFER_HEIGHT, 0.040 if state == "transfer" else 0.015
            if self._stable(xy < 0.002 and abs(z - height) < 0.004 and tilt < 0.04):
                self._transition("approach")
            elif elapsed > 10.0:
                return self._fail_planetary("ring transfer did not converge")
        elif state == "approach":
            height, speed = 0.038, 0.006
            phase_error = float(torch.atan2(torch.sin(desired_yaw - yaw), torch.cos(desired_yaw - yaw)).abs())
            if self._stable(xy < 0.0015 and abs(z - height) < 0.003 and tilt < 0.03 and phase_error < 0.02):
                self._ring_yaw_anchor = yaw.clone()
                self._ring_search_offset = 0.0
                self._ring_search_direction = 1
                self._ring_touching = False
                self._ring_depth_sample = (0.0, z)
                self._transition("search")
            elif elapsed > 8.0:
                return self._fail_planetary("ring approach did not converge")
        else:
            height, speed = max(self.RING_SEAT_HEIGHT - 0.0005, z - self.RING_PRELOAD_M), 0.003
            if self._seated(5) and xy < 0.0015 and tilt < 0.02:
                # Stop the preload and tooth sweep as soon as the ring seats.
                # Continuing to push while waiting to release drags the ring
                # sideways across the already mounted gears.
                self._release_position = ee[:, :3].clone()
                self._release_orientation = ee[:, 3:7].clone()
                self._ring_hold_position = ring[:, :3].clone()
                self._motion_target = self._release_position.clone()
                # Retain the learned gravity compensation; clearing it here
                # makes the wrist sag several millimetres into the assembly.
                self._transition("mesh_hold")
                return self._ring_hold(arm, gripper, ee, ring, 0.0, dt)
            elif elapsed > 16.0:
                if self._insert_attempt >= self.MAX_INSERT_ATTEMPTS:
                    return self._fail_planetary("ring meshing search exhausted")
                self._insert_attempt += 1
                self._transition("realign")
        target = centre + offset
        target[:, 2] += height
        if state == "realign" and z < self.RING_TRANSFER_HEIGHT - 0.005:
            target[:, :2] = ee[:, :2]
        return self._planetary_command(arm, gripper, target, orientation, 0.0, dt, speed, contact=state == "search")

    def _ring_target_transform(self, ee, ring, level):
        correction = quat_mul(level, quat_conjugate(ring[:, 3:7]))
        return quat_mul(correction, ee[:, 3:7]), quat_apply(correction, ee[:, :3] - ring[:, :3])

    def _ring_hold(self, arm, gripper, ee, ring, elapsed, dt):
        drift = float((ring[:, :3] - self._ring_hold_position).norm())
        xy, _, tilt = self._gear_errors(5)
        ready = self._seated(5) and xy < self.RING_HOLD_XY_TOLERANCE and tilt < 0.02 and drift < 0.0015
        if self._stable(ready, 0.5):
            self._ring_release_opening = float(self.scene["robot"].data.joint_pos[0, gripper.joint_ids].mean())
            self._transition("release")
        elif elapsed > 3.0:
            if self._insert_attempt >= self.MAX_INSERT_ATTEMPTS:
                return self._fail_planetary("ring did not stay seated when preload stopped")
            self._insert_attempt += 1
            self._transition("realign")
        return self._planetary_command(
            arm, gripper, self._release_position, self._release_orientation, 0.0, dt, 0.003, contact=True
        )

    def _ring_stage(self, arm, gripper, ee, ring, centre, elapsed, dt):
        """Hold the picked ring clear while moving the chest out of its path."""
        robot = self.scene["robot"]
        if self._planetary_state == "transfer":
            self._transition("stage")
            elapsed = 0.0
        if self._planetary_state == "stage":
            side = self.gear_to_pin_map["ring_gear"]["arm"]
            waypoint = centre.clone()
            waypoint[:, 0] += 0.12
            waypoint[:, 1] += 0.13 if side == "left" else -0.13
            waypoint[:, 2] += self.RING_TRANSFER_HEIGHT
            q = ring[:, 3:7]
            yaw = torch.atan2(
                2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2].square() + q[:, 3].square())
            )
            level = torch.zeros_like(q)
            level[:, 0], level[:, 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
            correction = quat_mul(level, quat_conjugate(q))
            orientation = quat_mul(correction, ee[:, 3:7])
            target = waypoint + quat_apply(correction, ee[:, :3] - ring[:, :3])
            if self._stable(float((waypoint - ring[:, :3]).norm()) < 0.003, 0.3):
                self._ring_torso_ids = robot.find_joints("torso_joint3")[0]
                self._ring_torso_start = robot.data.joint_pos[:, self._ring_torso_ids].clone()
                self._ring_torso_hold = ee.clone()
                self._ring_saved_torso_limits = robot.data.joint_pos_limits[:, self._ring_torso_ids].clone()
                # The environment locks the torso at reset. Open only the
                # bounded pitch interval needed for this driven movement.
                lower = self._ring_torso_start - 0.05
                upper = torch.full_like(lower, self.RING_TORSO_TARGET)
                robot.write_joint_position_limit_to_sim(
                    torch.stack((lower, upper), dim=-1), joint_ids=self._ring_torso_ids
                )
                self._transition("retract_torso")
            elif elapsed > 10.0:
                return self._fail_planetary("ring staging waypoint not reached")
            return self._planetary_command(arm, gripper, target, orientation, 0.0, dt, 0.040)

        alpha = min(elapsed / self.RING_TORSO_MOVE_S, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        torso = self._ring_torso_start * (1 - alpha) + self.RING_TORSO_TARGET * alpha
        position_error = float((ee[:, :3] - self._ring_torso_hold[:, :3]).norm())
        actual = float(robot.data.joint_pos[:, self._ring_torso_ids].min())
        # Require sustained clearance and a held world pose. Contact-solver
        # velocity buffers can stay nonzero under static gravitational load.
        if elapsed >= self.RING_TORSO_MOVE_S and self._stable(position_error < 0.005 and actual > -0.20, 0.3):
            self._ring_torso_retracted = True
            self._transition("transfer")
        elif elapsed > 6.0:
            return self._fail_planetary("torso retraction did not clear the assembly")
        action, ids = self._planetary_command(
            arm, gripper, self._ring_torso_hold[:, :3], self._ring_torso_hold[:, 3:7], 0.0, dt, 0.040
        )
        return torch.cat((action, torso), dim=-1), ids + self._ring_torso_ids

    def reset_actuator_settings(self):
        super().reset_actuator_settings()
        saved = getattr(self, "_ring_saved_torso_limits", None)
        if saved is not None:
            self.scene["robot"].write_joint_position_limit_to_sim(saved, joint_ids=self._ring_torso_ids)
            self._ring_saved_torso_limits = None

    def _ring_release(self, arm, gripper, ee, elapsed, dt):
        state = self._planetary_state
        target = self._release_position.clone()
        opening = self.GRIPPER_OPEN_POS
        if state == "release":
            opening = min(opening, self._ring_release_opening + 0.010 * elapsed)
            actual = float(self.scene["robot"].data.joint_pos[0, gripper.joint_ids].mean())
            if elapsed >= 0.8 and opening >= self.GRIPPER_OPEN_POS - 1e-6 and actual >= opening - 0.002:
                self._transition("retreat")
        else:
            target[:, 2] += 0.055
            if state == "retreat" and float((target - ee[:, :3]).norm()) < 0.005:
                self._transition("verify")
            elif state == "verify":
                if self._stable(all(self._seated(i) for i in range(1, 6)), 0.7):
                    self._park_start = self.scene["robot"].data.joint_pos[:, arm.joint_ids].clone()
                    self._transition("park")
                elif elapsed > 3.0:
                    return self._fail_planetary("assembly did not remain seated after ring release")
        if elapsed > 8.0:
            return self._fail_planetary(f"ring {state} timed out")
        return self._planetary_command(arm, gripper, target, self._release_orientation, opening, dt, 0.040)

    def _ring_park(self, arm, gripper, side, elapsed):
        ready = getattr(self, f"initial_pos_{side}").unsqueeze(0)
        alpha = min(elapsed / 2.0, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        action = self._park_start * (1 - alpha) + ready * alpha
        jaw = torch.full((1, len(gripper.joint_ids)), self.GRIPPER_OPEN_POS, device=self.device)
        if elapsed >= 2.5:
            if not all(self._seated(i) for i in range(1, 6)):
                return self._fail_planetary("a component moved while parking after ring placement")
            if self._stable(True, 0.5):
                self.ring_complete = True
                self._ring_active = False
        return torch.cat((action, jaw), dim=-1), list(arm.joint_ids) + list(gripper.joint_ids)
