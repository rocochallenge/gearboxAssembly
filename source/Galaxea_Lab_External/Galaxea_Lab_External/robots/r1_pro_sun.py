"""Feedback meshing of the central gear, with clearance for the seated planets.

The stock gripper cannot follow the sun gear all the way down between the
planets. Grasp its upper rim, mesh while the fingertips clear their tops, then
open and verify the final gravity-assisted seating. Contact uses bounded
position preload with frozen integral compensation; no physics assets are changed.
"""

import math

import torch

from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from .gearbox_geometry import centre_gear_seat_position


class R1ProSunMixin:
    SUN_TIMEOUT_S = 55.0
    SUN_PRELOAD_M = 0.0015
    # The 25 g gear needs little grip force. The stock 100 N effort can eject
    # it from the shallow upper-rim grasp when the teeth meet.
    SUN_GRIP_EFFORT_N = 8.0
    SUN_SEARCH_SPEED_RAD_S = math.radians(4.0)
    # Twelve teeth: a +/-15 degree sweep covers one complete tooth pitch.
    SUN_SEARCH_SWEEP_RAD = math.radians(15.0)
    SUN_APPROACH_HEIGHT_M = 0.040
    SUN_GRAVITY_SEATING = False

    def _sun_mesh_yaw(self, current_yaw, centre):
        """Nearest tooth phase that best matches the three observed planets.

        Equal external gears require opposite tooth phases at each contact.
        The 12-tooth asset's tip-face midpoint is 22.17 degrees modulo 30,
        giving a 0.66 degree phase datum (15 - 2*22.17 modulo 30).
        The circular mean is only a starting estimate; contact still verifies
        engagement because the three planets can have incompatible phases.
        """
        phases = []
        for i in range(1, 4):
            gear = self._held_object(i).data.root_state_w
            q = gear[:, 3:7]
            yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            delta = gear[:, :2] - centre[:, :2]
            direction = torch.atan2(delta[:, 1], delta[:, 0])
            phases.append(12 * (2 * direction - yaw + math.radians(0.66)))
        phases = torch.stack(phases)
        phase = torch.atan2(phases.sin().sum(dim=0), phases.cos().sum(dim=0)) / 12
        difference = 12 * (phase - current_yaw)
        return current_yaw + torch.atan2(difference.sin(), difference.cos()) / 12

    def get_action(self):
        if not getattr(self, "planetary_complete", False):
            return super().get_action()
        if not getattr(self, "sun_complete", False):
            if not getattr(self, "_sun_active", False):
                self._start_sun()
            return self._sun_action()
        return super().get_action()

    def _start_sun(self):
        self._sun_active = True
        self.sun_complete = False
        self._sun_started = self.count
        self._sun_disturbed_since = None
        self._planetary_gear = 4
        # The three-pin assignment deliberately has no entry for the sun.
        side = "left" if float(self.sun_planetary_gear_4.data.root_state_w[0, 1]) >= 0.0 else "right"
        self.gear_to_pin_map["sun_planetary_gear_4"] = {"arm": side}
        self._planetary_last_count = self.count
        self._pick_attempt = self._insert_attempt = 1
        self._motion_target = self._position_bias = self._orientation_bias = None
        self._grasp_yaw_delta = self._grasp_shift = 0.0
        self._planetary_opening = self.GRIPPER_OPEN_POS
        self.total_time_steps = int(self.EPISODE_LENGTH_S / self.sim_dt)
        self._transition("hover")

    def _pickup_height_offset(self):
        # Virtual TCP is 7 mm below the finger tips. A +11 mm target leaves
        # the tips 18 mm up a 24 mm gear: 6 mm of rim remains inside the jaws.
        return 0.011 if getattr(self, "_sun_active", False) else super()._pickup_height_offset()

    def _pickup_clearance(self):
        # Near the shoulder, the 100 mm waypoint is inside the arm's minimum
        # folding radius. Fifty millimetres clears the 30 mm carrier/pins and
        # 24 mm loose parts before the transfer moves toward the centre.
        return 0.05 if getattr(self, "_sun_active", False) else super()._pickup_clearance()

    def _pickup_lift_reached(self, target, ee):
        if getattr(self, "_sun_active", False):
            # At the inner edge of reach a few millimetres of lateral wrist
            # error is harmless once the object has demonstrably lifted clear.
            # The shared pickup stage separately verifies that object motion.
            return float((target - ee[:, :3]).norm()) < 0.015 and abs(float(target[0, 2] - ee[0, 2])) < 0.008
        return super()._pickup_lift_reached(target, ee)

    def _sun_fingertip_corners(self, arm):
        """Opened G1Z fingertip envelope in the end-effector link frame."""
        local_tcp = quat_apply(quat_conjugate(self._gripper_down_quat(arm)), -self._tcp_offset(arm))
        half_width = self.GRIPPER_OPEN_POS + 0.032647
        return (
            torch.tensor(
                [[x, y, 0.00695] for x in (-0.028, 0.028) for y in (-half_width, half_width)], device=self.device
            )
            + local_tcp
        )

    def _sun_mesh_clearance(self, arm, ee, offset, orientation, centre):
        """Lowest pad tip over the planets, including the subsequent opening.

        Pad dimensions include the URDF joint origins, as in the grasp planner.
        Use the observed in-hand transform rather than assuming the gear stayed
        at the commanded grasp height. Tilted pads need extra vertical clearance.
        """
        corners = self._sun_fingertip_corners(arm)
        relative_tip_z = float(offset[0, 2] + quat_apply(orientation.expand(4, -1), corners)[:, 2].min())
        actual_tip_z = float(ee[0, 2] + quat_apply(ee[:, 3:7].expand(4, -1), corners)[:, 2].min())
        planets = torch.cat([self._held_object(i).data.root_state_w for i in range(1, 4)])
        up = quat_apply(planets[:, 3:7], torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(3, -1))
        # 24 mm height and 31.5 mm outer radius from the existing gear mesh.
        top_z = float((planets[:, 2] + 0.024 * up[:, 2] + 0.0315 * up[:, :2].norm(dim=-1)).max())
        top_relative = top_z - float(centre[0, 2])
        floor = max(0.012, top_relative + 0.002 - relative_tip_z)
        return floor, top_relative, actual_tip_z - top_z

    def _live_pin(self, gear_id):
        if gear_id == 4:
            return self.planetary_carrier.data.root_state_w[:, :3]
        return super()._live_pin(gear_id)

    def _seated(self, gear_id):
        if gear_id == 4:
            carrier = self.planetary_carrier.data.root_state_w
            gear = self.sun_planetary_gear_4.data.root_state_w
            delta = gear[:, :3] - centre_gear_seat_position(carrier[:, :3], carrier[:, 3:7])
            xy, z = float(delta[:, :2].norm()), float(delta[0, 2])
            _, _, tilt = self._gear_errors(4)
            return xy < 0.005 and abs(z) < 0.005 and tilt < 0.05
        return super()._seated(gear_id)

    def _sun_action(self):
        if self._planetary_state == "failed":
            return None, None
        if (self.count - self._sun_started) * self.sim_dt > self.SUN_TIMEOUT_S:
            return self._fail_planetary("central gear time budget exhausted")
        if self._planetary_state == "search" and not getattr(self, "_sun_phase_aligned", False):
            self._transition("phase_align")
        if self._planetary_state in ("phase_align", "search", "mesh_hold"):
            self._use_sun_grip_effort()
        # Allow brief contact vibration, but never proceed after displacing a
        # mounted planet. This timer is separate from the state-transition gate.
        if all(self._seated(i) for i in range(1, 4)):
            self._sun_disturbed_since = None
        elif self._sun_disturbed_since is None:
            self._sun_disturbed_since = self.count
        elif (self.count - self._sun_disturbed_since) * self.sim_dt > 0.3:
            if self._planetary_state in ("search", "mesh_hold") and self._insert_attempt < self.MAX_INSERT_ATTEMPTS:
                self._insert_attempt += 1
                self._sun_disturbed_since = None
                self._transition("realign")
            elif self._planetary_state != "realign" or (self.count - self._state_started) * self.sim_dt > 2.0:
                return self._fail_planetary("previously seated planetary gear moved during central insertion")

        dt = max(self.sim_dt, (self.count - self._planetary_last_count) * self.sim_dt)
        self._planetary_last_count = self.count
        side = self.gear_to_pin_map["sun_planetary_gear_4"]["arm"]
        arm = getattr(self, f"{side}_arm_entity_cfg")
        gripper = getattr(self, f"{side}_gripper_entity_cfg")
        ee = self.scene["robot"].data.body_state_w[:, arm.body_ids[0], :7]
        gear = self.sun_planetary_gear_4.data.root_state_w
        centre = self._live_pin(4)
        down = self._gripper_down_quat(arm)
        elapsed = (self.count - self._state_started) * self.sim_dt
        state = self._planetary_state
        if state in ("search", "mesh_hold", "realign", "retry_pick") and self._seated(4):
            # The shallow, gentle grasp may let the gear settle under gravity.
            # This can also happen while retreating from a slipped grasp. Open
            # where the wrist is and verify after retreat; never regrasp a gear
            # that has already landed correctly between the planets.
            self._release_position = ee[:, :3].clone()
            self._release_orientation = ee[:, 3:7].clone()
            self._transition("release")
            state, elapsed = "release", 0.0
        if float(gear[0, 2]) < self.table_height - 0.05:
            return self._fail_planetary("central gear left the table workspace")
        if state in ("hover", "descend", "close", "lift"):
            return self._pick_action(arm, gripper, ee, gear, centre, down, elapsed, dt)
        if state in ("transfer", "approach", "phase_align", "search", "mesh_hold", "realign"):
            return self._sun_insert(arm, gripper, ee, gear, centre, down, elapsed, dt)
        if state in ("release", "retreat", "verify", "retry_pick"):
            # Once dropped between the planets, a low regrasp could hit them.
            # A failed post-release seat is reported without blindly regrasping.
            if state == "verify" and elapsed > 2.0 and not self._seated(4):
                return self._fail_planetary("central gear did not settle after meshing and release")
            return self._release_action(arm, gripper, ee, gear, centre, down, elapsed, dt)
        if state == "park":
            return self._sun_park(arm, gripper, side, elapsed)
        raise RuntimeError(f"Unknown central gear state: {state}")

    def _use_sun_grip_effort(self):
        if getattr(self, "_sun_saved_grip_effort", None) is not None:
            return
        robot = self.scene["robot"]
        side = self.gear_to_pin_map["sun_planetary_gear_4"]["arm"]
        ids = list(getattr(self, f"{side}_gripper_entity_cfg").joint_ids)
        limits = robot.data.joint_effort_limits[:, ids].clone()
        self._sun_saved_grip_effort = (ids, limits)
        robot.write_joint_effort_limit_to_sim(limits.clamp(max=self.SUN_GRIP_EFFORT_N), joint_ids=ids)

    def reset_actuator_settings(self):
        """Restore the original jaw effort before the next stage or reset."""
        saved = getattr(self, "_sun_saved_grip_effort", None)
        if saved is not None:
            ids, limits = saved
            self.scene["robot"].write_joint_effort_limit_to_sim(limits, joint_ids=ids)
            self._sun_saved_grip_effort = None

    def _sun_insert(self, arm, gripper, ee, gear, centre, down, elapsed, dt):
        state = self._planetary_state
        if float((ee[:, :3] - gear[:, :3] - self._tcp_offset(arm)).norm()) > 0.08:
            result = self._retry_pick(ee, "central gear slipped during transport")
            if result is not None:
                return result
            return self._planetary_command(arm, gripper, ee[:, :3], ee[:, 3:7], 0.0, dt)

        q = gear[:, 3:7]
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        desired_yaw = yaw
        xy, z, tilt = self._gear_errors(4)
        if state in ("approach", "phase_align"):
            desired_yaw = self._sun_mesh_yaw(yaw, centre)
        elif state == "search":
            if not hasattr(self, "_sun_depth_sample"):
                self._sun_depth_sample = (self.count, z)
                self._sun_touch_started = False
            sample_count, sample_z = self._sun_depth_sample
            if (self.count - sample_count) * self.sim_dt >= 0.4:
                if sample_z - z < 0.0005 and z < 0.038:
                    self._sun_touch_started = True
                self._sun_depth_sample = (self.count, z)
            if self._sun_touch_started:
                self._sun_search_offset += self._sun_search_direction * self.SUN_SEARCH_SPEED_RAD_S * dt
            sweep = self.SUN_SEARCH_SWEEP_RAD * self._insert_attempt
            if abs(self._sun_search_offset) >= sweep:
                self._sun_search_offset = max(-sweep, min(sweep, self._sun_search_offset))
                self._sun_search_direction *= -1
            requested = self._sun_yaw_anchor + self._sun_search_offset
            yaw_error = torch.atan2(torch.sin(requested - yaw), torch.cos(requested - yaw))
            # Do not accumulate an arbitrarily large twist against blocked teeth.
            desired_yaw = yaw + yaw_error.clamp(-0.04, 0.04)
        elif state == "mesh_hold":
            yaw_error = torch.atan2(torch.sin(self._sun_hold_yaw - yaw), torch.cos(self._sun_hold_yaw - yaw))
            desired_yaw = yaw + yaw_error.clamp(-0.04, 0.04)
        level = torch.zeros_like(q)
        level[:, 0], level[:, 3] = torch.cos(desired_yaw / 2), torch.sin(desired_yaw / 2)
        correction = quat_mul(level, quat_conjugate(q))
        orientation = quat_mul(correction, ee[:, 3:7])
        offset = quat_apply(correction, ee[:, :3] - gear[:, :3])

        if state in ("transfer", "realign"):
            height = 0.080 if state == "transfer" else 0.055
            speed = 0.06 if state == "transfer" else 0.015
            if self._stable(xy < 0.0015 and abs(z - height) < 0.004 and tilt < 0.04):
                self._transition("approach")
            elif elapsed > 10.0:
                return self._fail_planetary("could not align above the carrier centre")
        elif state in ("approach", "phase_align"):
            height, speed = self.SUN_APPROACH_HEIGHT_M, 0.008
            phase_error = float(torch.atan2(torch.sin(desired_yaw - yaw), torch.cos(desired_yaw - yaw)).abs())
            if self.SUN_GRAVITY_SEATING:
                ready = xy < 0.0008 and abs(z - height) < 0.002 and tilt < 0.02 and phase_error < 0.01
            else:
                ready = xy < 0.0015 and abs(z - height) < 0.003 and tilt < 0.04 and phase_error < 0.02
            if self._stable(ready, 0.3 if self.SUN_GRAVITY_SEATING else 0.2):
                if self.SUN_GRAVITY_SEATING:
                    self._release_position = ee[:, :3].clone()
                    self._release_orientation = ee[:, 3:7].clone()
                    self._transition("release")
                else:
                    self._sun_yaw_anchor = yaw.clone()
                    self._sun_search_offset = 0.0
                    self._sun_search_direction = 1
                    self._sun_phase_aligned = True
                    self._sun_depth_sample = (self.count, z)
                    self._sun_touch_started = False
                    self._transition("search")
            elif elapsed > 8.0:
                return self._fail_planetary("central gear approach did not converge")
        else:
            # Stop the fingers above the planets, after substantial tooth
            # overlap. The final travel happens under gravity after opening.
            floor, planet_top, finger_clearance = self._sun_mesh_clearance(arm, ee, offset, orientation, centre)
            if floor > planet_top - 0.006:
                return self._fail_planetary("grasp too low to mesh without touching the planetary gears")
            if finger_clearance < 0.001:
                if self._insert_attempt >= self.MAX_INSERT_ATTEMPTS:
                    return self._fail_planetary("finger clearance exhausted during central gear meshing")
                self._insert_attempt += 1
                self._transition("realign")
                target = ee[:, :3].clone()
                target[:, 2] += 0.025
                return self._planetary_command(arm, gripper, target, ee[:, 3:7], 0.0, dt, 0.015)
            mesh_height = max(floor, planet_top - 0.006)
            speed = 0.004
            height = max(mesh_height, z - self.SUN_PRELOAD_M)
            engaged = -0.002 <= z < mesh_height + 0.002 and tilt < 0.04
            ready = xy < 0.0015 and engaged and finger_clearance > 0.001
            if state == "search" and engaged:
                # Stop twisting once the teeth overlap. Let centring settle
                # before releasing, instead of continually exciting side loads.
                self._sun_hold_yaw = yaw.clone()
                self._transition("mesh_hold")
            elif state == "mesh_hold" and self._stable(ready, 0.4):
                self._release_position = ee[:, :3].clone()
                self._release_orientation = ee[:, 3:7].clone()
                self._transition("release")
            elif elapsed > (3.0 if state == "mesh_hold" else 16.0):
                if self._insert_attempt >= self.MAX_INSERT_ATTEMPTS:
                    return self._fail_planetary("central gear meshing search exhausted")
                self._insert_attempt += 1
                self._transition("realign")
        target = centre + offset
        target[:, 2] += height
        if state == "realign" and z < 0.045:
            # Lift clear before correcting XY, to avoid dragging a pad across
            # the planetary gears while unloading a failed contact attempt.
            target[:, :2] = ee[:, :2]
        return self._planetary_command(
            arm, gripper, target, orientation, 0.0, dt, speed, contact=state in ("search", "mesh_hold")
        )

    def _sun_park(self, arm, gripper, side, elapsed):
        ready = getattr(self, f"initial_pos_{side}").unsqueeze(0)
        alpha = min(elapsed / 2.0, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        action = self._park_start * (1 - alpha) + ready * alpha
        jaw = torch.full((1, len(gripper.joint_ids)), self.GRIPPER_OPEN_POS, device=self.device)
        if elapsed >= 2.5:
            if not all(self._seated(i) for i in range(1, 5)):
                return self._fail_planetary("one of the first four gears moved after retreat")
            self.sun_complete = True
            self._sun_active = False
            self.reset_actuator_settings()
            shift = self.count - int(self.count_step_10[0])
            for step in range(10, 15):
                getattr(self, f"count_step_{step}").add_(shift)
            self.total_time_steps = int(self.count_step_14[-1])
            self._phase_last_count = self.count - 1
        return torch.cat((action, jaw), dim=-1), list(arm.joint_ids) + list(gripper.joint_ids)
