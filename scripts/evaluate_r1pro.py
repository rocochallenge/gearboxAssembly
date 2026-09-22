"""Reproducible R1Pro / R1 Lite planetary assembly evaluation.

After sourcing scripts/env.sh, run e.g.:
    python scripts/evaluate_r1pro.py --headless --seed 42 --output /tmp/r1pro-42.json
    python scripts/evaluate_r1pro.py --headless --seed 42 --policy legacy --output /tmp/r1pro-42-legacy.json
    python scripts/evaluate_r1pro.py --robot r1_lite --headless --first-four --seed 44 --fast_physics --output /tmp/r1lite-44.json

Uses production physics, randomization, actions and scoring tolerances. Camera
observations and dataset recording are disabled. The default endpoint is after
the first three gears; --first-four includes central-gear verification and
--full-assembly also measures retention at episode end.
The JSON result is authoritative (Kit shutdown can override process exit codes).
"""

import argparse
import contextlib
import itertools
import json
import math
import os
import time
from pathlib import Path
from unittest.mock import patch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--robot", choices=("r1_pro", "r1_lite"), default="r1_pro")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--policy", choices=("feedback", "planetary", "legacy"), default="feedback")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--fast_physics", action="store_true", help="Evaluate with 32/8 robot solver iterations")
parser.add_argument("--max-seconds", type=float, help="Stop a diagnostic run early (simulated seconds)")
endpoint = parser.add_mutually_exclusive_group()
endpoint.add_argument("--full-assembly", action="store_true")
endpoint.add_argument("--first-four", action="store_true", help="Stop after verifying the central gear")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.max_seconds is not None and args.max_seconds <= 0:
    parser.error("--max-seconds must be positive")
os.environ["ROCO_ROBOT_BUNDLE"] = args.robot
app = AppLauncher(args).app

import torch
from Galaxea_Lab_External.robots.r1_lite_rule_policy import R1LiteRulePolicy
from Galaxea_Lab_External.robots.r1_lite_feedback_policy import R1LiteFeedbackPolicy
from Galaxea_Lab_External.robots.r1_pro_planetary import R1ProPlanetaryMixin
from Galaxea_Lab_External.robots.gearbox_geometry import centre_gear_seat_position
from Galaxea_Lab_External.robots.r1_pro_rule_policy import R1ProRulePolicy
from Galaxea_Lab_External.robots.physics_profiles import use_fast_physics
from Galaxea_Lab_External.tasks.direct.galaxea_lab_external import galaxea_lab_external_env as module
from Galaxea_Lab_External.tasks.direct.galaxea_lab_external.galaxea_lab_external_env_cfg import GalaxeaLabExternalEnvCfg


class NoCamera:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass


class EvaluationEnv(module.GalaxeaLabExternalEnv):
    def _setup_scene(self):
        # Only the sensor constructor is replaced. All rigid bodies and their
        # material properties are built by the production scene setup.
        with patch.object(module, "Camera", NoCamera):
            super()._setup_scene()

    def _get_observations(self):
        return {"policy": {}}

    def _write_hdf5_episode(self, success):
        pass

    def _get_dones(self):
        # Snapshot before the production step automatically resets the scene.
        done = super()._get_dones()
        policy = self.rule_policy
        phase = (
            getattr(policy, "_planetary_gear", None),
            getattr(policy, "_planetary_state", None),
            bool(getattr(policy, "planetary_complete", False)),
            bool(getattr(policy, "sun_complete", False)),
        )
        first_three_done = (
            not args.full_assembly
            and not args.first_four
            and (
                bool(getattr(policy, "planetary_complete", False))
                or (args.policy == "legacy" and policy.count >= int(policy.count_step_7[-1]))
            )
        )
        first_four_done = args.first_four and (
            bool(getattr(policy, "sun_complete", False))
            if args.policy == "feedback"
            else policy.count >= int(policy.count_step_10[-1])
        )
        if (
            phase != getattr(self, "_last_phase", None)
            or policy.count % 100 == 0
            or first_three_done
            or first_four_done
            or any(bool(flag.item()) for flag in done)
        ):
            self.last_snapshot = snapshot(self)
            self._last_phase = phase
        return done

    def _get_rewards(self):
        # _get_dones already computed the production score for this step.
        return self.score_tensor.to(torch.float32)


class LegacyR1ProPolicy(R1ProRulePolicy):
    get_action = R1LiteRulePolicy.get_action


class PlanetaryOnlyR1ProPolicy(R1ProRulePolicy):
    """Compare the timed fourth-gear sequence with the same first-three policy."""

    get_action = R1ProPlanetaryMixin.get_action


class PlanetaryOnlyR1LitePolicy(R1LiteFeedbackPolicy):
    get_action = R1ProPlanetaryMixin.get_action


def snapshot(env):
    pins, _, gears, quats, *_ = env.get_key_points()
    matches = [
        [
            bool(
                env._pose_matches(
                    gears[i], quats[i], pin, env.PLANETARY_GEAR_XY_TOLERANCE, env.PLANETARY_GEAR_Z_TOLERANCE
                ).item()
            )
            for pin in pins
        ]
        for i in range(3)
    ]
    assignment = max(itertools.permutations(range(3)), key=lambda a: sum(matches[i][a[i]] for i in range(3)))
    seated = [matches[i][assignment[i]] for i in range(3)]
    policy = env.rule_policy
    centre = env.planetary_carrier.data.root_state_w[:, :3]
    seat = centre_gear_seat_position(centre, env.planetary_carrier.data.root_state_w[:, 3:7])
    centre_seated = bool(
        env._pose_matches(gears[3], quats[3], seat, env.CENTRE_GEAR_XY_TOLERANCE, env.CENTRE_GEAR_Z_TOLERANCE).item()
    )
    ee = env.robot.data.body_state_w[:, policy.right_arm_entity_cfg.body_ids[0], :7]
    gear_id = getattr(policy, "_planetary_gear", None)
    object_name = "ring_gear" if gear_id == 5 else f"sun_planetary_gear_{gear_id}"
    arm_name = policy.gear_to_pin_map.get(object_name, {}).get("arm", "right")
    arm = getattr(policy, f"{arm_name}_arm_entity_cfg")
    active_ee = env.robot.data.body_state_w[:, arm.body_ids[0], :7]
    phases = []
    for position, quaternion in zip(gears[:3], quats[:3]):
        q = quaternion[0]
        yaw = torch.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        delta = position[0, :2] - centre[0, :2]
        phases.append(12 * (yaw - 2 * torch.atan2(delta[1], delta[0])))
    phase_errors = [
        float(torch.rad2deg(torch.atan2(torch.sin(p - phases[0]), torch.cos(p - phases[0])) / 12))
        if seated[0] and seated[i] else None
        for i, p in enumerate(phases)
    ]
    return {
        "simulation_seconds": round(policy.count * policy.sim_dt, 3),
        "state": getattr(policy, "_planetary_state", "legacy"),
        "active_gear": getattr(policy, "_planetary_gear", None),
        "failure": getattr(policy, "planetary_failure", None),
        "verified_first_three": bool(getattr(policy, "planetary_complete", False)),
        "verified_fourth": bool(getattr(policy, "sun_complete", False)),
        "verified_fifth": bool(getattr(policy, "assembly_complete", False)),
        "ring_position": env.ring_gear.data.root_state_w[0, :3].tolist(),
        "ring_quaternion": env.ring_gear.data.root_state_w[0, 3:7].tolist(),
        "centre_seated": centre_seated,
        "centre_position": gears[3][0].tolist(),
        "centre_quaternion": quats[3][0].tolist(),
        "carrier_position": centre[0].tolist(),
        "centre_error_mm": ((gears[3] - centre)[0] * 1000).tolist(),
        "centre_seat_error_mm": ((gears[3] - seat)[0] * 1000).tolist(),
        "right_ee_pose": ee[0].tolist(),
        "active_arm": arm_name,
        "active_ee_pose": active_ee[0].tolist(),
        "active_arm_joints": env.robot.data.joint_pos[0, arm.joint_ids].tolist(),
        "pick_attempt": getattr(policy, "_pick_attempt", None),
        "insert_attempt": getattr(policy, "_insert_attempt", None),
        "seated": seated,
        "planetary_score": sum(seated),
        "assembly_score": int(env.score_tensor.item()),
        "gear_positions": [g[0].tolist() for g in gears[:3]],
        "gear_quaternions": [q[0].tolist() for q in quats[:3]],
        "mesh_phase_errors_deg": phase_errors,
        "right_arm_joints": env.robot.data.joint_pos[0, policy.right_arm_entity_cfg.joint_ids].tolist(),
        "pin_positions": [p[0].tolist() for p in pins],
        "pin_assignment": list(assignment),
        "xy_errors_mm": [float((gears[i][0, :2] - pins[assignment[i]][0, :2]).norm()) * 1000 for i in range(3)],
    }


args.output.parent.mkdir(parents=True, exist_ok=True)
log_path = args.output.with_suffix(".log")
trace_path = args.output.with_suffix(".jsonl")
cfg = GalaxeaLabExternalEnvCfg()
cfg.seed = args.seed
cfg.sim.device = args.device
cfg.record_data = False
cfg.keep_failed = None
cfg.num_rerenders_on_reset = 0
if args.fast_physics:
    use_fast_physics(cfg)
if args.policy == "legacy":
    cfg.rule_policy_class = LegacyR1ProPolicy if args.robot == "r1_pro" else R1LiteRulePolicy
    cfg.episode_length_s = 60.0
elif args.policy == "planetary":
    cfg.rule_policy_class = PlanetaryOnlyR1ProPolicy if args.robot == "r1_pro" else PlanetaryOnlyR1LitePolicy
    cfg.episode_length_s = cfg.rule_policy_class.EPISODE_LENGTH_S
else:
    cfg.rule_policy_class = R1ProRulePolicy if args.robot == "r1_pro" else R1LiteFeedbackPolicy
    cfg.episode_length_s = cfg.rule_policy_class.EPISODE_LENGTH_S

with log_path.open("w") as log, trace_path.open("w") as trace, torch.inference_mode():
    with contextlib.redirect_stdout(log):
        env = EvaluationEnv(cfg=cfg)
        env.reset(seed=args.seed)
    last_state = None
    started = time.perf_counter()
    duration = min(cfg.episode_length_s, args.max_seconds or cfg.episode_length_s)
    for step in range(math.ceil(duration / env.step_dt)):
        with contextlib.redirect_stdout(log):
            _, _, terminated, truncated, _ = env.step(torch.zeros((1, cfg.action_space), device=env.device))
        record = env.last_snapshot
        state = (record["active_gear"], record["state"])
        if state != last_state or step % 20 == 19:
            trace.write(json.dumps(record) + "\n")
            trace.flush()
        if state != last_state:
            print(
                f"t={record['simulation_seconds']:6.2f} gear={state[0]} state={state[1]} seated={record['seated']}",
                flush=True,
            )
            last_state = state
        ended = bool(terminated.item() or truncated.item())
        first_three_done = (
            record["verified_first_three"]
            if args.policy != "legacy"
            else (record["simulation_seconds"] >= float(env.rule_policy.count_step_7[-1]) * env.physics_dt)
        )
        first_four_done = args.first_four and (
            record["verified_fourth"]
            if args.policy == "feedback"
            else (record["simulation_seconds"] >= float(env.rule_policy.count_step_10[-1]) * env.physics_dt)
        )
        if ended or (first_three_done and not args.full_assembly and not args.first_four) or first_four_done:
            break
    if not ended:
        record = snapshot(env)
    # Preserve the final completion/cutoff even if no state name changed and
    # the next periodic trace sample was not reached.
    trace.write(json.dumps(record) + "\n")
    trace.flush()
    result = {
        "robot": args.robot,
        "policy_class": cfg.rule_policy_class.__name__,
        "initial_torso_position": list(cfg.initial_torso_pos),
        "physics_dt": env.physics_dt,
        "control_dt": env.step_dt,
        "record_data": False,
        "seed": args.seed,
        "wall_seconds": time.perf_counter() - started,
        "evaluation_limit_seconds": args.max_seconds,
        "fast_physics": args.fast_physics,
        "robot_solver_position_iterations": cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count,
        "robot_solver_velocity_iterations": cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count,
        "policy": args.policy,
        "full_assembly": args.full_assembly,
        "first_four_requested": args.first_four,
        "first_four_success": (
            record["planetary_score"] == 3
            and record["centre_seated"]
            and (args.policy != "feedback" or record["verified_fourth"])
        ),
        "first_three_success": record["planetary_score"] == 3 and (
            args.policy == "legacy" or record["verified_first_three"]
        ),
        "assembly_success": (
            record["assembly_score"] == env.SUCCESS_SCORE
            and (args.policy != "feedback" or record["verified_fifth"])
        ) if args.full_assembly else None,
        **record,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    with contextlib.redirect_stdout(log):
        env.close()
app.close()
