"""Feedback-controlled pickup and insertion of R1Pro's three planetary gears.

The policy reads the same simulator object poses as the legacy rule policy.
It commands joints only: there are no attachments, object teleports, or changes
to the assembly score. Each transition has a deadline and each gear has a
bounded retry budget.
"""

import itertools
import math

import torch

from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_mul,
)


def bounded_dls_step(jacobian, task_step, lower, upper, damping=0.001):
    """Redistribute IK motion after fixing joints that would exceed a bound.

    Clipping an unconstrained solution alone leaves the remaining joints aimed
    at a motion the saturated joint cannot perform. Re-solving the residual
    with the free columns uses R1Pro's redundant degree of freedom instead.
    """
    fixed = torch.zeros_like(lower)
    free = torch.ones_like(lower, dtype=torch.bool)
    identity = torch.eye(jacobian.shape[1], device=jacobian.device, dtype=jacobian.dtype)
    for _ in range(jacobian.shape[-1] + 1):
        active_jacobian = jacobian * free.unsqueeze(1)
        residual = task_step - (jacobian @ fixed.unsqueeze(-1)).squeeze(-1)
        solve = torch.linalg.solve(
            active_jacobian @ active_jacobian.transpose(-1, -2) + damping**2 * identity,
            residual.unsqueeze(-1),
        )
        step = fixed + (active_jacobian.transpose(-1, -2) @ solve).squeeze(-1)
        violated = free & ((step < lower) | (step > upper))
        if not bool(violated.any()):
            break
        fixed = torch.where(violated, step.clamp(min=lower, max=upper), fixed)
        free &= ~violated
    return step.clamp(min=lower, max=upper)


class R1ProPlanetaryMixin:
    ASSEMBLY_NAME = "R1Pro"
    GRASP_SHIFT_AXIS_LOCAL = (1.0, 0.0, 0.0)
    PLANETARY_TIMEOUT_S = 120.0
    EPISODE_LENGTH_S = 160.0
    MAX_PICK_ATTEMPTS = 3
    MAX_INSERT_ATTEMPTS = 3
    # At the 20 Hz control rate this limits changes in joint targets to 2 rad/s.
    # Large redundant-IK corrections near a joint limit can shake a gear loose.
    PLANETARY_MAX_JOINT_STEP = 0.10
    PLANETARY_FREE_YAW = True
    PLANETARY_TRANSFER_CLEARANCE = 0.08

    def _use_free_planetary_yaw(self):
        return self.PLANETARY_FREE_YAW

    def _refine_joint_target(self, arm_entity_cfg, jacobian, joint_pos, joint_pos_des):
        if not hasattr(self, "_planetary_state") or (
            self.planetary_complete
            and not getattr(self, "_sun_active", False)
            and not getattr(self, "_ring_active", False)
        ):
            return joint_pos_des
        max_step = min(self.IK_MAX_JOINT_STEP, self.PLANETARY_MAX_JOINT_STEP)
        limits = self.scene["robot"].data.joint_pos_limits[:, arm_entity_cfg.joint_ids]
        lower, upper = limits[..., 0] + 0.002, limits[..., 1] - 0.002
        free_yaw = self._use_free_planetary_yaw() and self._planetary_state in ("transfer", "insert", "realign")
        if not free_yaw and bool(((joint_pos_des >= lower) & (joint_pos_des <= upper)).all()):
            return joint_pos + (joint_pos_des - joint_pos).clamp(-max_step, max_step)
        lower_step = (lower - joint_pos).clamp(-max_step, max_step)
        upper_step = (upper - joint_pos).clamp(-max_step, max_step)
        task_step = (jacobian @ (joint_pos_des - joint_pos).unsqueeze(-1)).squeeze(-1)
        if free_yaw:
            # Free tool yaw during travel can avoid the redundant arm's limits.
            # An adapter may restore it for tooth-phase alignment above a pin.
            # The remaining five constraints still level and centre the gear.
            base_quat = self.scene["robot"].data.root_state_w[:, 3:7]
            up = quat_apply(quat_conjugate(base_quat), torch.tensor([[0.0, 0.0, 1.0]], device=self.device))
            projection = torch.eye(3, device=self.device).unsqueeze(0) - up.unsqueeze(-1) @ up.unsqueeze(-2)
            jacobian = jacobian.clone()
            jacobian[:, 3:] = projection @ jacobian[:, 3:]
            task_step[:, 3:] = (projection @ task_step[:, 3:].unsqueeze(-1)).squeeze(-1)
        return joint_pos + bounded_dls_step(jacobian, task_step, lower_step, upper_step)

    def _start_planetary(self):
        if self.scene.num_envs != 1:
            raise ValueError(f"{self.ASSEMBLY_NAME}'s sequential rule policy requires num_envs=1")
        self._assign_planetary_pins()
        self.planetary_complete = False
        self.planetary_failure = None
        self._planetary_gear = 1
        self._pick_attempt = 1
        self._insert_attempt = 1
        self._planetary_last_count = self.count
        self._planetary_state = "settle"
        self._state_started = self.count
        self._stable_since = None
        self._motion_target = None
        self._position_bias = None
        self._grasp_yaw_delta = 0.0
        self._grasp_shift = 0.0
        self._planetary_opening = self.GRIPPER_OPEN_POS
        # The remainder is re-timed once all three gears have been verified.
        self.total_time_steps = int(self.EPISODE_LENGTH_S / self.sim_dt)

    def _assign_planetary_pins(self):
        """Reserve pins by arm reach before considering pickup travel distance.

        Greedily selecting the pin nearest each loose gear can give a left arm
        the negative-Y pin and leave a positive-Y pin for the right arm. R1Pro
        stalls at its shoulder/wrist limits on those cross-body insertions.
        """
        carrier = self.planetary_carrier.data.root_state_w[:, :7]
        pins = [carrier[:, :3] + quat_apply(carrier[:, 3:7], p.unsqueeze(0)) for p in self.pin_local_positions[:3]]
        costs = []
        for gear_id in range(1, 4):
            assignment = self.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]
            arm = getattr(self, f"{assignment['arm']}_arm_entity_cfg")
            ready = self.scene["robot"].data.body_state_w[:, arm.body_ids[0], :3]
            gear = self._held_object(gear_id).data.root_state_w[:, :3]
            costs.append([
                float((pin - ready).norm() + 0.05 * (pin - gear).norm()) + 0.0001 * (pin_id != assignment.get("pin"))
                for pin_id, pin in enumerate(pins)
            ])
        assignment = min(itertools.permutations(range(3)), key=lambda a: sum(costs[i][a[i]] for i in range(3)))
        for gear_id, pin_id in enumerate(assignment, 1):
            self.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"].update(
                pin=pin_id, pin_local_pos=self.pin_local_positions[pin_id], pin_world_pos=pins[pin_id]
            )
        print(f"[{self.ASSEMBLY_NAME} assembly] planetary pin assignment={assignment}")

    def _transition(self, state):
        if state == "hover" and not self._choose_grasp():
            self._fail_planetary("no clear top-down grasp")
            return
        self._planetary_state = state
        self._state_started = self.count
        self._stable_since = None
        print(f"[{self.ASSEMBLY_NAME} assembly] gear={self._planetary_gear} state={state} t={self.count * self.sim_dt:.2f}")

    def _choose_grasp(self):
        """Choose wrist yaw/jaw opening with clearance from neighbouring parts.

        Apply the URDF finger-joint origins to convert_r1_pro_urdf.py's pad
        boxes: pad one is centred at x=-0.000003, y=0.016897 + jaw position.
        Check the closing sweep, including a small grasp offset along the
        pads' length. A round gear needs no particular pickup yaw.
        """
        gear_name = f"sun_planetary_gear_{self._planetary_gear}"
        gear = self.obj_dict[gear_name].data.root_state_w[0, :3].tolist()
        side = self.gear_to_pin_map[gear_name]["arm"]
        sign = -1 if side == "left" else 1
        candidates = []
        obstacles = []
        for name, obj in self.obj_dict.items():
            if name == gear_name:
                continue
            pos = obj.data.root_state_w[0, :3].tolist()
            if abs(pos[2] - gear[2]) > 0.05:
                continue
            radius = {"ring_gear": 0.10, "planetary_carrier": 0.07, "planetary_reducer": 0.04}.get(name, 0.032)
            obstacles.append((pos[0] - gear[0], pos[1] - gear[1], radius))
        # Keep the wrist in the reachable yaw range established for R1Pro.
        for degrees in range(90, 181, 5):
            yaw = math.radians(sign * degrees)
            c, s = math.cos(yaw), math.sin(yaw)
            for opening in (self.GRIPPER_OPEN_POS, 0.045, 0.040, 0.036):
                for shift in (0.0, -0.008, 0.008):
                    clearance = float("inf")
                    for dx, dy, radius in obstacles:
                        x, y = c * dx + s * dy - shift, -s * dx + c * dy
                        for xmin, xmax, ymin, ymax in (
                            (-0.028003, 0.027997, 0.030, opening + 0.032647),
                            (-0.027997, 0.028003, -opening - 0.032647, -0.030),
                        ):
                            gap = math.hypot(max(xmin - x, 0.0, x - xmax), max(ymin - y, 0.0, y - ymax)) - radius
                            clearance = min(clearance, gap)
                    if clearance >= 0.006:
                        cost = abs(degrees - 135) + (self.GRIPPER_OPEN_POS - opening) * 2500 + abs(shift) * 1000
                        candidates.append((cost, yaw - math.radians(sign * 135), opening, shift, clearance))
        if not candidates:
            return False
        _, self._grasp_yaw_delta, self._planetary_opening, self._grasp_shift, clearance = min(candidates)
        print(
            f"[R1Pro assembly] grasp yaw offset={math.degrees(self._grasp_yaw_delta):.0f}deg "
            f"jaw={self._planetary_opening:.3f} shift={self._grasp_shift:.3f} clearance={clearance:.3f}m"
        )
        return True

    def _grasp_yaw_quat(self):
        finished = getattr(self, "planetary_complete", False) and not getattr(self, "_sun_active", False)
        delta = 0.0 if finished else getattr(self, "_grasp_yaw_delta", 0.0)
        return torch.tensor([[math.cos(delta / 2), 0.0, 0.0, math.sin(delta / 2)]], device=self.device)

    def _tcp_offset(self, arm_entity_cfg):
        return quat_apply(self._grasp_yaw_quat(), super()._tcp_offset(arm_entity_cfg))

    def _gripper_down_quat(self, arm_entity_cfg=None):
        return quat_mul(self._grasp_yaw_quat(), super()._gripper_down_quat(arm_entity_cfg))

    def _stable(self, condition, duration=0.2):
        if not condition:
            self._stable_since = None
            return False
        if self._stable_since is None:
            self._stable_since = self.count
        return (self.count - self._stable_since) * self.sim_dt >= duration

    def _live_pin(self, gear_id):
        carrier = self.planetary_carrier.data.root_state_w[:, :7]
        local = self.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]["pin_local_pos"]
        return carrier[:, :3] + quat_apply(carrier[:, 3:7], local.unsqueeze(0))

    def _gear_errors(self, gear_id):
        gear = self._held_object(gear_id).data.root_state_w
        delta = gear[:, :3] - self._live_pin(gear_id)
        up = quat_apply(gear[:, 3:7], torch.tensor([[0.0, 0.0, 1.0]], device=self.device))
        tilt = torch.acos(up[:, 2].clamp(-1.0, 1.0))
        return float(delta[:, :2].norm()), float(delta[0, 2]), float(tilt[0])

    def _seated(self, gear_id):
        xy, z, tilt = self._gear_errors(gear_id)
        # These are the environment's existing tolerances, with positive Z
        # required so an object below the carrier cannot pass verification.
        return xy < 0.002 and 0.0 <= z < 0.012 and tilt < 0.05

    def _fail_planetary(self, reason):
        self.planetary_failure = f"gear {self._planetary_gear}: {reason}"
        print(f"[{self.ASSEMBLY_NAME} assembly] failed: {self.planetary_failure}")
        self._transition("failed")
        self.total_time_steps = self.count
        return None, None

    def _planetary_command(self, arm, gripper, position, orientation, opening, dt, speed=0.08, contact=False):
        """Rate-limit Cartesian targets and hold the arm while actuating the jaw."""
        if self._motion_target is None:
            self._motion_target = self.scene["robot"].data.body_state_w[:, arm.body_ids[0], :3].clone()
            self._position_bias = torch.zeros_like(self._motion_target)
            self._orientation_bias = torch.zeros_like(self._motion_target)
        if getattr(self, "_position_bias", None) is None:
            self._position_bias = torch.zeros_like(self._motion_target)
        if getattr(self, "_orientation_bias", None) is None:
            self._orientation_bias = torch.zeros_like(self._motion_target)
        delta = position - self._motion_target
        scale = (speed * dt / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.0)
        self._motion_target += delta * scale
        actual = self.scene["robot"].data.body_state_w[:, arm.body_ids[0], :3]
        # DLS + position PD has a millimetre-scale static error under gravity.
        # Integrate only small tracking errors, with an anti-windup limit;
        # otherwise a 2 mm insertion gate can never converge even in free space.
        error = self._motion_target - actual
        if contact:
            self._position_bias[:, 2].clamp_(min=0.0)
        elif float(error.norm()) < 0.02:
            self._position_bias = (self._position_bias + 2.0 * dt * error).clamp(-0.015, 0.015)
        actual_quat = self.scene["robot"].data.body_state_w[:, arm.body_ids[0], 3:7]
        rotation_error = axis_angle_from_quat(quat_mul(orientation, quat_conjugate(actual_quat)))
        if not contact and float(rotation_error.norm()) < 0.15:
            self._orientation_bias = (self._orientation_bias + dt * rotation_error).clamp(-0.10, 0.10)
        bias_angle = self._orientation_bias.norm(dim=-1)
        bias_quat = quat_from_angle_axis(bias_angle, self._orientation_bias / bias_angle.unsqueeze(-1).clamp_min(1e-8))
        action, joint_ids = self.move_robot_to_position(
            arm,
            gripper,
            self.diff_ik_controller,
            self._motion_target + self._position_bias,
            quat_mul(bias_quat, orientation),
            None,
        )
        jaw = torch.full((action.shape[0], len(gripper.joint_ids)), opening, device=self.device)
        return torch.cat((action, jaw), dim=-1), list(joint_ids) + list(gripper.joint_ids)

    def _retry_pick(self, ee, reason):
        if self._pick_attempt >= self.MAX_PICK_ATTEMPTS:
            return self._fail_planetary(reason)
        self._pick_attempt += 1
        self._retreat_position = ee[:, :3].clone()
        self._retreat_position[:, 2] += 0.10
        self._transition("retry_pick")
        return None

    def _planetary_action(self):
        if not hasattr(self, "_planetary_state"):
            self._start_planetary()
        if self._planetary_state == "failed":
            return None, None
        if self.count * self.sim_dt >= self.PLANETARY_TIMEOUT_S:
            return self._fail_planetary("planetary time budget exhausted")

        dt = max(self.sim_dt, (self.count - self._planetary_last_count) * self.sim_dt)
        self._planetary_last_count = self.count
        gear_id = self._planetary_gear
        arm_name = self.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]["arm"]
        arm = getattr(self, f"{arm_name}_arm_entity_cfg")
        gripper = getattr(self, f"{arm_name}_gripper_entity_cfg")
        robot = self.scene["robot"]
        ee = robot.data.body_state_w[:, arm.body_ids[0], :7]
        gear = self._held_object(gear_id).data.root_state_w
        if float(gear[0, 2]) < self.table_height - 0.05:
            return self._fail_planetary("gear left the table workspace")
        pin = self._live_pin(gear_id)
        down = self._gripper_down_quat(arm)
        elapsed = (self.count - self._state_started) * self.sim_dt
        state = self._planetary_state
        if state in ("settle", "hover", "descend", "close", "lift"):
            handler = self._pick_action
        elif state in ("transfer", "insert", "realign"):
            handler = self._insertion_action
        elif state in ("release", "retreat", "verify", "retry_pick"):
            handler = self._release_action
        elif state == "park":
            handler = self._park_action
        else:
            raise RuntimeError(f"Unknown planetary assembly state: {state}")
        return handler(arm, gripper, ee, gear, pin, down, elapsed, dt)

    def _pick_action(self, arm, gripper, ee, gear, pin, down, elapsed, dt):
        state = self._planetary_state
        opening = getattr(self, "_planetary_opening", self.GRIPPER_OPEN_POS)
        orientation = down
        speed = 0.08
        target = ee[:, :3].clone()

        if state == "settle":
            if elapsed >= 0.5:
                self._transition("hover")
        elif state in ("hover", "descend"):
            target = gear[:, :3] + self._tcp_offset(arm)
            local_shift = torch.tensor([self.GRASP_SHIFT_AXIS_LOCAL], device=self.device)
            local_shift = local_shift * getattr(self, "_grasp_shift", 0.0)
            target += quat_apply(down, local_shift)
            # Root Z follows the actual table/gear height. The virtual TCP is
            # 7 mm beyond the pad tips, leaving 3 mm clearance on pickup.
            target[:, 2] += self._pickup_clearance() if state == "hover" else self._pickup_height_offset()
            speed = 0.10 if state == "hover" else 0.045
            reached = float((target - ee[:, :3]).norm()) < (0.008 if state == "hover" else 0.004)
            angle_tolerance = 0.02 if opening <= 0.040 else 0.05
            aligned = float(quat_error_magnitude(ee[:, 3:7], down)[0]) < angle_tolerance
            if self._stable(reached and aligned):
                if state == "hover":
                    self._transition("descend")
                else:
                    self._grasp_position = target.clone()
                    self._pick_height = float(gear[0, 2])
                    self._transition("close")
            elif elapsed > 10.0:
                result = self._retry_pick(ee, f"{state} pose unreachable")
                if result is not None:
                    return result
        elif state == "close":
            target = self._grasp_position
            opening = 0.0
            if elapsed >= 1.0:
                self._transition("lift")
        elif state == "lift":
            opening = 0.0
            target = self._pickup_lift_target(arm, ee, gear, elapsed)
            speed = 0.06
            reached = self._pickup_lift_reached(target, ee)
            lifted = float(gear[0, 2]) > self._pick_height + min(0.05, self._pickup_clearance() - 0.01)
            if self._stable(reached and lifted, 0.3):
                self._transition("transfer")
            elif elapsed > 4.0 or (reached and not lifted and elapsed > 2.0):
                result = self._retry_pick(ee, "gear did not lift with the gripper")
                if result is not None:
                    return result

        return self._planetary_command(arm, gripper, target, orientation, opening, dt, speed)

    def _pickup_height_offset(self):
        return -0.004

    def _pickup_clearance(self):
        return 0.10

    def _pickup_lift_target(self, arm, ee, gear, elapsed):
        target = self._grasp_position.clone()
        target[:, 2] += self._pickup_clearance()
        return target

    def _pickup_lift_reached(self, target, ee):
        return float((target - ee[:, :3]).norm()) < 0.005

    def _planetary_insertion_yaw(self, yaw, pin):
        """Desired gear yaw and whether it is ready to descend onto the pin."""
        return yaw, True

    def _planetary_target_transform(self, ee, gear, level, pin):
        """Wrist target from the observed gear-to-wrist grasp."""
        correction = quat_mul(level, quat_conjugate(gear[:, 3:7]))
        orientation = quat_mul(correction, ee[:, 3:7])
        return orientation, quat_apply(correction, ee[:, :3] - gear[:, :3])

    def _insertion_action(self, arm, gripper, ee, gear, pin, down, elapsed, dt):
        gear_id = self._planetary_gear
        state = self._planetary_state
        opening = getattr(self, "_planetary_opening", self.GRIPPER_OPEN_POS)
        orientation = down
        speed = 0.08
        target = ee[:, :3].clone()

        if state in ("transfer", "insert", "realign"):
            opening = 0.0
            if float((ee[:, :3] - gear[:, :3] - self._tcp_offset(arm)).norm()) > 0.08:
                result = self._retry_pick(ee, "gear slipped during transport")
                if result is not None:
                    return result
            else:
                # Rotate the measured gear-to-wrist transform so the gear's
                # axis becomes vertical, preserving its yaw. Re-observe it
                # every step to compensate for in-hand settling and slip.
                q = gear[:, 3:7]
                yaw = torch.atan2(
                    2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2].square() + q[:, 3].square())
                )
                yaw, phase_aligned = self._planetary_insertion_yaw(yaw, pin)
                level = torch.zeros_like(q)
                level[:, 0], level[:, 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
                orientation, offset = self._planetary_target_transform(ee, gear, level, pin)
                clearance = self.PLANETARY_TRANSFER_CLEARANCE if state == "transfer" else 0.045 if state == "realign" else 0.010
                target = pin + offset
                target[:, 2] += clearance
                speed = 0.08 if state == "transfer" else 0.015
                xy, z, tilt = self._gear_errors(gear_id)
                if state in ("transfer", "realign"):
                    if self._stable(xy < 0.0015 and abs(z - clearance) < 0.004 and tilt < 0.04 and phase_aligned):
                        self._transition("insert")
                    elif elapsed > 10.0:
                        return self._fail_planetary("could not align above the live pin")
                else:
                    # Never release just because the insertion timer elapsed.
                    ready = xy < 0.0015 and 0.0 <= z < 0.015 and tilt < 0.04 and phase_aligned
                    if self._stable(ready, 0.3):
                        self._release_position = ee[:, :3].clone()
                        self._release_orientation = orientation.clone()
                        self._transition("release")
                    elif elapsed > 7.0:
                        if self._insert_attempt >= self.MAX_INSERT_ATTEMPTS:
                            return self._fail_planetary("insertion did not converge")
                        self._insert_attempt += 1
                        self._transition("realign")

        return self._planetary_command(arm, gripper, target, orientation, opening, dt, speed)

    def _release_action(self, arm, gripper, ee, gear, pin, down, elapsed, dt):
        gear_id = self._planetary_gear
        state = self._planetary_state
        opening = getattr(self, "_planetary_opening", self.GRIPPER_OPEN_POS)
        orientation = down
        speed = 0.08
        target = ee[:, :3].clone()

        if state in ("release", "retreat", "verify"):
            # Once released, freeze XY. Following a falling gear would sweep
            # the fingers into the carrier and disturb already seated gears.
            target = self._release_position.clone()
            orientation = self._release_orientation
            if state != "release":
                target[:, 2] += 0.10
            speed = 0.06
            if state == "release" and elapsed >= 0.8:
                self._transition("retreat")
            elif state == "retreat" and float((target - ee[:, :3]).norm()) < 0.005:
                self._transition("verify")
            elif state == "verify":
                if self._stable(self._seated(gear_id), 0.5):
                    self._park_start = self.scene["robot"].data.joint_pos[:, arm.joint_ids].clone()
                    self._transition("park")
                elif elapsed > 2.0:
                    result = self._retry_pick(ee, "gear not seated after release")
                    if result is not None:
                        return result
            if elapsed > 8.0:
                return self._fail_planetary(f"{state} timed out")
        elif state == "retry_pick":
            target = self._retreat_position
            if float((target - ee[:, :3]).norm()) < 0.005 and elapsed > 0.8:
                self._transition("hover")
            elif elapsed > 6.0:
                return self._fail_planetary("could not retreat for a retry")

        return self._planetary_command(arm, gripper, target, orientation, opening, dt, speed)

    def _park_action(self, arm, gripper, ee, gear, pin, down, elapsed, dt):
        gear_id = self._planetary_gear
        arm_name = self.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]["arm"]
        elapsed = (self.count - self._state_started) * self.sim_dt
        opening = getattr(self, "_planetary_opening", self.GRIPPER_OPEN_POS)
        ready = getattr(self, f"initial_pos_{arm_name}").unsqueeze(0)
        alpha = min(elapsed / 2.0, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        action = self._park_start * (1 - alpha) + ready * alpha
        jaw = torch.full((1, len(gripper.joint_ids)), opening, device=self.device)
        if elapsed >= 2.5:
            missing = [i for i in range(1, gear_id + 1) if not self._seated(i)]
            if missing:
                return self._fail_planetary(f"previously seated gears moved: {missing}")
            if gear_id == 3:
                self.planetary_complete = True
                shift = self.count - int(self.count_step_8[0])
                for step in range(8, 15):
                    getattr(self, f"count_step_{step}").add_(shift)
                self.total_time_steps = int(self.count_step_14[-1])
                self._phase_last_count = self.count - 1
            else:
                self._planetary_gear += 1
                self._pick_attempt = self._insert_attempt = 1
                self._motion_target = None
                self._transition("hover")
        return torch.cat((action, jaw), dim=-1), list(arm.joint_ids) + list(gripper.joint_ids)

    def get_action(self):
        if self.TCP_offset_x is None:
            self._compute_tcp_offset()
        if not getattr(self, "planetary_complete", False):
            return self._planetary_action()
        return super().get_action()
