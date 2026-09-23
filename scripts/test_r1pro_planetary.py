"""Controller regression checks without a physics scene.

Run after sourcing scripts/env.sh:
    python scripts/test_r1pro_planetary.py --headless
Isaac Sim is launched only to make the production policy imports available.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace
import unittest

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import numpy as np
from isaaclab.utils.math import quat_apply
from pxr import Usd, UsdGeom
from Galaxea_Lab_External import GALAXEA_LAB_ASSETS_DIR
from Galaxea_Lab_External.robots.r1_pro_rule_policy import R1ProRulePolicy
from Galaxea_Lab_External.robots.r1_pro_planetary import bounded_dls_step
from Galaxea_Lab_External.robots.physics_profiles import use_fast_physics
from Galaxea_Lab_External.robots.robot_bundles import (
    GALAXEA_R1_PRO_BUNDLE,
    GALAXEA_R1_LITE_BUNDLE,
    GALAXEA_R1_BUNDLE,
)
from Galaxea_Lab_External.tasks.direct.galaxea_lab_external.galaxea_lab_external_env import GalaxeaLabExternalEnv


class SceneStub(dict):
    """Mapping-shaped scene fixture with the one environment used in production."""

    num_envs = 1


class PlanetaryControlTests(unittest.TestCase):
    def test_ik_uses_redundancy_when_a_joint_is_at_its_limit(self):
        jacobian = torch.tensor([[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        task_step = torch.tensor([[-0.2, 0.2]])
        lower = torch.tensor([[0.0, -1.0, -1.0]])
        upper = torch.ones(1, 3)
        step = bounded_dls_step(jacobian, task_step, lower, upper)
        self.assertTrue(bool(((step >= lower) & (step <= upper)).all()))
        torch.testing.assert_close((jacobian @ step.unsqueeze(-1)).squeeze(-1), task_step)
        # Simple clipping of the original solution leaves a 10 cm residual.
        clipped = torch.tensor([[0.0, -0.1, 0.2]])
        self.assertGreater(float((jacobian @ clipped.unsqueeze(-1)).squeeze(-1).sub(task_step).norm()), 0.09)

    def test_ik_remains_bounded_when_the_target_is_unreachable(self):
        jacobian = torch.eye(3).unsqueeze(0)
        lower, upper = -torch.ones(1, 3) * 0.1, torch.ones(1, 3) * 0.1
        step = bounded_dls_step(jacobian, torch.tensor([[1.0, -1.0, 2.0]]), lower, upper)
        self.assertTrue(bool(torch.isfinite(step).all()))
        torch.testing.assert_close(step, torch.tensor([[0.1, -0.1, 0.1]]))

    def test_planetary_ik_limits_large_joint_corrections(self):
        p = self.make_policy("transfer")
        p.scene["robot"].data.joint_pos_limits = torch.tensor([[[-2.0, 2.0]] * 8])
        actual = torch.zeros(1, 7)
        target = torch.tensor([[0.5, -0.4, 0.02, 0.0, 0.0, 0.0, 0.0]])
        result = p._refine_joint_target(p.left_arm_entity_cfg, torch.eye(6, 7).unsqueeze(0), actual, target)
        self.assertLessEqual(float(result.abs().max()), 0.100001)
        self.assertAlmostEqual(float(result[0, 2]), 0.02)
        p.planetary_complete = True
        torch.testing.assert_close(p._refine_joint_target(p.left_arm_entity_cfg, None, actual, target), target)

    def test_insertion_can_rotate_about_the_gear_axis_to_avoid_a_joint_limit(self):
        p = self.make_policy("transfer")
        limits = torch.tensor([[[-2.0, 2.0]] * 8])
        limits[:, 0, 1] = 0.002
        p.scene["robot"].data.joint_pos_limits = limits
        jacobian = torch.zeros(1, 6, 7)
        jacobian[:, 0, :2] = 1.0
        jacobian[:, 5, 1] = 1.0
        target = torch.tensor([[0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        result = p._refine_joint_target(p.left_arm_entity_cfg, jacobian, torch.zeros(1, 7), target)
        self.assertAlmostEqual(float(result[0, 0]), 0.0, places=5)
        self.assertGreater(float(result[0, 1]), 0.099)

    def make_phase_alignment_policy(self, gear_id=2):
        """Place a carried planet at the unloaded transfer waypoint.

        The first planet lies on the carrier's +X bearing with zero yaw.  The
        second pin is five degrees around the carrier, so equal external gears
        require ten degrees of phase advance modulo the 30-degree tooth pitch.
        """
        p = self.make_policy("transfer")
        p._planetary_gear = gear_id
        p._pick_attempt = 1
        radius = 0.05
        p.gear_to_pin_map["sun_planetary_gear_1"]["pin_local_pos"] = torch.tensor([radius, 0.0, 0.0])
        angle = torch.tensor(5.0 * torch.pi / 180.0)
        p.gear_to_pin_map["sun_planetary_gear_2"]["pin_local_pos"] = torch.tensor(
            [radius * angle.cos(), radius * angle.sin(), 0.0]
        )
        first = p._held_object(1).data.root_state_w
        first[:, :3] = p._live_pin(1)
        first[:, 3:7] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        carried = p._held_object(gear_id).data.root_state_w
        carried[:, :3] = p._live_pin(gear_id)
        carried[:, 2] += p.PLANETARY_TRANSFER_CLEARANCE
        carried[:, 3:7] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        return p

    def test_planet_phase_is_not_constrained_during_free_space_transfer(self):
        p = self.make_phase_alignment_policy()
        gear = p._held_object(2).data.root_state_w
        gear[:, 0] += 0.010
        yaw = torch.tensor([0.0])
        requested, ready = p._planetary_insertion_yaw(yaw, p._live_pin(2))
        torch.testing.assert_close(requested, yaw)
        self.assertFalse(ready)
        self.assertTrue(p._use_free_planetary_yaw())
        self.assertIsNone(getattr(p, "_planetary_phase_token", None))

    def test_planet_phase_gate_rejects_low_tilted_or_nontransfer_payloads(self):
        for case in ("below_clearance", "tilted", "nontransfer"):
            with self.subTest(case=case):
                p = self.make_phase_alignment_policy()
                gear = p._held_object(2).data.root_state_w
                if case == "below_clearance":
                    gear[:, 2] -= 0.005
                elif case == "tilted":
                    angle = torch.tensor(0.05 / 2)
                    gear[:, 3:7] = torch.tensor([[angle.cos(), angle.sin(), 0.0, 0.0]])
                else:
                    p._planetary_state = "realign"

                yaw = torch.tensor([0.123])
                requested, ready = p._planetary_insertion_yaw(yaw, p._live_pin(2))
                torch.testing.assert_close(requested, yaw)
                self.assertFalse(ready)
                self.assertIsNone(getattr(p, "_planetary_phase_token", None))
                self.assertTrue(p._use_free_planetary_yaw())

    def test_planet_phase_latches_only_at_the_unloaded_pin_waypoint(self):
        p = self.make_phase_alignment_policy()
        requested, ready = p._planetary_insertion_yaw(torch.tensor([0.0]), p._live_pin(2))
        # Ten degrees are required; one control update is bounded to 0.05 rad.
        torch.testing.assert_close(requested, torch.tensor([0.05]))
        self.assertFalse(ready)
        self.assertEqual(p._planetary_phase_token, (2, 1))
        self.assertFalse(p._use_free_planetary_yaw())

        # Once latched, fixed-yaw authority remains through insertion rather
        # than disappearing when the payload leaves the transfer waypoint.
        p._planetary_state = "insert"
        p._held_object(2).data.root_state_w[:, 2] -= 0.060
        self.assertFalse(p._use_free_planetary_yaw())
        requested, ready = p._planetary_insertion_yaw(torch.tensor([0.170]), p._live_pin(2))
        self.assertTrue(ready)
        self.assertAlmostEqual(float(requested[0]), 10.0 * torch.pi / 180.0, places=5)

    def test_new_pick_attempt_restores_free_yaw_until_phase_is_reacquired(self):
        p = self.make_phase_alignment_policy()
        p._planetary_insertion_yaw(torch.tensor([0.0]), p._live_pin(2))
        self.assertFalse(p._use_free_planetary_yaw())

        p._pick_attempt = 2
        p._planetary_state = "transfer"
        self.assertTrue(p._use_free_planetary_yaw())
        p._held_object(2).data.root_state_w[:, 0] += 0.010
        p._planetary_insertion_yaw(torch.tensor([0.0]), p._live_pin(2))
        self.assertEqual(p._planetary_phase_token, (2, 1))
        self.assertTrue(p._use_free_planetary_yaw())

        p._held_object(2).data.root_state_w[:, 0] -= 0.010
        p._planetary_insertion_yaw(torch.tensor([0.0]), p._live_pin(2))
        self.assertEqual(p._planetary_phase_token, (2, 2))
        self.assertFalse(p._use_free_planetary_yaw())

    def test_phase_constraint_does_not_change_gear_one_sun_or_ring_yaw(self):
        p = self.make_phase_alignment_policy(gear_id=1)
        p._planetary_phase_token = (2, 1)
        yaw = torch.tensor([0.123])
        requested, ready = p._planetary_insertion_yaw(yaw, p._live_pin(1))
        torch.testing.assert_close(requested, yaw)
        self.assertTrue(ready)
        self.assertTrue(p._use_free_planetary_yaw())

        for gear_id, active in ((4, "_sun_active"), (5, "_ring_active")):
            with self.subTest(gear_id=gear_id):
                p._planetary_gear = gear_id
                setattr(p, active, True)
                self.assertTrue(p._use_free_planetary_yaw())
                setattr(p, active, False)

    def test_starting_a_new_planetary_sequence_clears_the_phase_gate(self):
        p = self.make_phase_alignment_policy()
        p._planetary_phase_token = (2, 1)
        p._assign_planetary_pins = lambda: None
        p._start_planetary()
        self.assertIsNone(p._planetary_phase_token)
        self.assertTrue(p._use_free_planetary_yaw())

    def test_blocked_planetary_gear_does_not_make_the_wrist_target_keep_descending(self):
        p = self.make_policy("transfer")
        gear = p._held_object(1).data.root_state_w
        pin = p._live_pin(1)
        ee = p.scene["robot"].data.body_state_w[:, 0, :7]
        requests = []
        p._planetary_command = lambda arm, gripper, target, *args: requests.append(target.clone())

        def command(height, gap):
            gear[:, 2] = pin[:, 2] + height
            ee[:, :3] = gear[:, :3] + p._tcp_offset(None)
            ee[:, 2] = gear[:, 2] + gap
            p._insertion_action(p.left_arm_entity_cfg, p.left_gripper_entity_cfg,
                                ee, gear, pin, ee[:, 3:7], 0.0, 0.05)

        command(0.080, 0.280)
        p._planetary_state = "insert"
        command(0.030, 0.280)
        first_target = requests[-1]
        command(0.030, 0.270)
        torch.testing.assert_close(requests[-1], first_target)

        # A lift above the peg unloads contact and permits recalibration.
        p._planetary_state = "realign"
        command(0.055, 0.270)
        self.assertAlmostEqual(float(requests[-1][0, 2] - pin[0, 2]), 0.315, places=5)
        command(0.040, 0.260)
        self.assertAlmostEqual(float(requests[-1][0, 2] - pin[0, 2]), 0.315, places=5)

    def make_policy(self, state="insert"):
        p = R1ProRulePolicy.__new__(R1ProRulePolicy)
        p.device = "cpu"
        p.table_height = 0.9
        p.sim_dt = 0.01
        p.sim = SimpleNamespace(device="cpu")
        p.count = 500
        p._planetary_last_count = 495
        p._planetary_state = state
        p._state_started = 500
        p._stable_since = None
        p._planetary_gear = 1
        p._pick_attempt = p._insert_attempt = 1
        p.planetary_complete = False
        p.planetary_failure = None
        p.total_time_steps = 16000
        p.diff_ik_controller = None
        p.left_arm_entity_cfg = SimpleNamespace(joint_ids=list(range(7)), body_ids=[0])
        p.left_gripper_entity_cfg = SimpleNamespace(joint_ids=[7], body_ids=[1])
        p._gripper_down_quat = lambda arm: torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        p._tcp_offset = lambda arm: torch.tensor([[-0.02, -0.02, 0.28]])
        p.planetary_carrier = SimpleNamespace(
            data=SimpleNamespace(root_state_w=torch.tensor([[0.45, 0.0, 0.90, 1.0, 0.0, 0.0, 0.0] + [0.0] * 6]))
        )
        p.gear_to_pin_map = {}
        for gear_id, local in enumerate(((0, -0.054, 0), (0.0471, 0.0268, 0), (-0.0471, 0.0268, 0)), 1):
            p.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"] = {"arm": "left", "pin_local_pos": torch.tensor(local)}
            pos = p._live_pin(gear_id)
            pos[:, 2] += 0.01
            root = torch.cat((pos, torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.zeros(1, 6)), dim=-1)
            setattr(p, f"sun_planetary_gear_{gear_id}", SimpleNamespace(data=SimpleNamespace(root_state_w=root)))
        ee = p.sun_planetary_gear_1.data.root_state_w[:, :7].clone()
        ee[:, :3] += p._tcp_offset(None)
        p.scene = SceneStub({
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    body_state_w=ee.unsqueeze(1),
                    joint_pos=torch.zeros(1, 8),
                    joint_effort_limits=torch.full((1, 8), 100.0),
                    root_state_w=torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] + [0.0] * 6]),
                )
            )
        })
        robot = p.scene["robot"]

        def write_effort_limits(limits, joint_ids):
            robot.data.joint_effort_limits[:, joint_ids] = limits

        robot.write_joint_effort_limit_to_sim = write_effort_limits
        p._motion_target = ee[:, :3].clone()
        p.move_robot_to_position = lambda *a: (torch.zeros(1, 7), list(range(7)))
        p.obj_dict = {f"sun_planetary_gear_{i}": p._held_object(i) for i in range(1, 4)}
        return p

    def advance(self, p, ticks):
        for _ in range(ticks):
            action = p._planetary_action()
            p.count += 5
        return action

    def make_arm_limited_retreat(self, lift=0.100, lateral=0.006):
        # Seed 25004: the pad cleared the gear by 100 mm, but wrist limits
        # left a 6 mm lateral residual against the old 5 mm pose gate.
        p = self.make_policy("retreat")
        robot = p.scene["robot"]
        ee = robot.data.body_state_w[:, 0, :7]
        p._release_position = ee[:, :3].clone()
        p._release_orientation = ee[:, 3:7].clone()
        ee[:, 0] += lateral
        ee[:, 2] += lift
        robot.data.joint_pos_limits = torch.tensor([[[-2.0, 2.0]] * 8])
        robot.data.joint_pos[:, 6] = 1.998
        p._state_started = p.count - 300
        return p

    def test_clear_arm_limited_retreat_reaches_post_release_verification(self):
        for lift, lateral in ((0.100, 0.006), (0.086, 0.011)):
            with self.subTest(lift=lift, lateral=lateral):
                p = self.make_arm_limited_retreat(lift, lateral)
                self.advance(p, 1)
                self.assertEqual(p._planetary_state, "verify")
                self.assertFalse(p.planetary_complete)
                self.advance(p, 10)
                self.assertEqual(p._planetary_state, "park")

    def test_arm_limit_cannot_excuse_incomplete_or_displaced_retreat(self):
        for lift, lateral, seated in ((0.05, 0.006, True), (0.10, 0.015, True), (0.10, 0.006, False)):
            with self.subTest(lift=lift, lateral=lateral, seated=seated):
                p = self.make_arm_limited_retreat(lift, lateral)
                if not seated:
                    p.sun_planetary_gear_1.data.root_state_w[:, 0] += 0.004
                self.advance(p, 1)
                self.assertEqual(p._planetary_state, "retreat")
                self.assertFalse(p.planetary_complete)

    def test_seated_planet_is_verified_instead_of_regrasped_after_slip(self):
        p = self.make_policy("retry_pick")
        ee = p.scene["robot"].data.body_state_w[:, 0, :7]
        p._retreat_position = ee[:, :3] + torch.tensor([[0.0, 0.0, 0.10]])
        self.advance(p, 1)
        self.assertEqual(p._planetary_state, "release")
        torch.testing.assert_close(p._release_position, ee[:, :3])
        self.assertFalse(p.planetary_complete)

    def test_unseated_planet_still_uses_pick_retry(self):
        p = self.make_policy("retry_pick")
        ee = p.scene["robot"].data.body_state_w[:, 0, :7]
        p._retreat_position = ee[:, :3] + torch.tensor([[0.0, 0.0, 0.10]])
        p.sun_planetary_gear_1.data.root_state_w[:, 0] += 0.004
        self.advance(p, 1)
        self.assertEqual(p._planetary_state, "retry_pick")
        self.assertFalse(p.planetary_complete)

    def test_planetary_jaw_opening_waits_for_the_fully_open_pad_envelope(self):
        for roll in (0.0, 0.17):
            with self.subTest(roll=roll):
                p = self.make_policy("release")
                p._planetary_gear = 3
                robot = p.scene["robot"]
                ee = robot.data.body_state_w[:, 0, :7]
                angle = torch.tensor(roll / 2)
                ee[:, 3:7] = torch.tensor([[angle.cos(), angle.sin(), 0.0, 0.0]])
                tip = torch.tensor([[-0.0295, -0.087647, -0.27155]])
                # Closed pads can clear while the outward sweep of tilted
                # fingers would still strike a mounted planetary gear.
                ee[:, 2] = 0.934 - quat_apply(ee[:, 3:7], tip)[0, 2] + 0.001
                p._release_position = ee[:, :3].clone()
                p._release_orientation = ee[:, 3:7].clone()
                robot.data.joint_pos[:, 7] = 0.030
                action, _ = self.advance(p, 1)
                self.assertAlmostEqual(float(action[0, 7]), 0.033, places=5)
                robot.data.joint_pos[:, 7] = 0.033
                action, _ = self.advance(p, 1)
                self.assertAlmostEqual(float(action[0, 7]), 0.033, places=5)

                p._planetary_state = "retreat"
                ee[:, 2] += 0.05
                action, _ = self.advance(p, 1)
                self.assertAlmostEqual(float(action[0, 7]), p.GRIPPER_OPEN_POS, places=5)

    def test_first_two_planets_keep_full_release_opening(self):
        for gear_id in (1, 2):
            with self.subTest(gear_id=gear_id):
                p = self.make_policy("release")
                p._planetary_gear = gear_id
                robot = p.scene["robot"]
                ee = robot.data.body_state_w[:, 0, :7]
                p._release_position = ee[:, :3].clone()
                p._release_orientation = ee[:, 3:7].clone()
                robot.data.joint_pos[:, 7] = 0.030
                action, _ = self.advance(p, 1)
                self.assertAlmostEqual(float(action[0, 7]), p.GRIPPER_OPEN_POS, places=5)

    def test_shifted_pin_does_not_trigger_timed_release(self):
        p = self.make_policy()
        p.planetary_carrier.data.root_state_w[:, 1] += 0.003
        action, _ = self.advance(p, 12)
        self.assertEqual(p._planetary_state, "insert")
        self.assertEqual(float(action[0, -1]), 0.0)
        self.assertAlmostEqual(float(p._motion_target[0, 1]), -0.071, places=5)

    def test_pin_rotation_is_followed(self):
        p = self.make_policy()
        p.planetary_carrier.data.root_state_w[:, 3:7] = torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]])
        torch.testing.assert_close(p._live_pin(1), torch.tensor([[0.504, 0.0, 0.90]]))

    def test_release_requires_sustained_alignment(self):
        p = self.make_policy()
        self.advance(p, 3)
        self.assertEqual(p._planetary_state, "insert")
        p.sun_planetary_gear_1.data.root_state_w[:, 0] += 0.003
        self.advance(p, 1)
        p.sun_planetary_gear_1.data.root_state_w[:, 0] -= 0.003
        self.advance(p, 4)
        self.assertEqual(p._planetary_state, "insert")
        action, _ = self.advance(p, 5)
        self.assertEqual(p._planetary_state, "release")
        self.assertGreater(float(action[0, -1]), 0.0)
        self.assertLessEqual(float(action[0, -1]), p.GRIPPER_OPEN_POS)

    def test_tilted_or_high_gear_does_not_release(self):
        for high in (False, True):
            p = self.make_policy()
            gear = p.sun_planetary_gear_1.data.root_state_w
            if high:
                gear[:, 2] += 0.02
            else:
                gear[:, 3:7] = torch.tensor([[0.99875, 0.049979, 0.0, 0.0]])
            self.advance(p, 12)
            self.assertEqual(p._planetary_state, "insert")

    def test_unseated_release_retries_instead_of_advancing(self):
        p = self.make_policy("verify")
        ee = p.scene["robot"].data.body_state_w[:, 0, :7]
        p._release_position = ee[:, :3].clone()
        p._release_orientation = ee[:, 3:7].clone()
        p.sun_planetary_gear_1.data.root_state_w[:, 0] += 0.004
        self.advance(p, 43)
        self.assertEqual(p._planetary_state, "retry_pick")
        self.assertEqual(p._planetary_gear, 1)
        self.assertFalse(p.planetary_complete)

    def test_insertion_retries_are_bounded(self):
        p = self.make_policy()
        p.sun_planetary_gear_1.data.root_state_w[:, 0] += 0.003
        p._insert_attempt = p.MAX_INSERT_ATTEMPTS
        p._state_started = p.count - 705
        self.advance(p, 1)
        self.assertEqual(p._planetary_state, "failed")
        self.assertIn("insertion did not converge", p.planetary_failure)

    def test_empty_grasp_does_not_advance_to_transfer(self):
        p = self.make_policy("lift")
        p._pick_height = 0.90
        p._grasp_position = p.scene["robot"].data.body_state_w[:, 0, :3].clone()
        p._grasp_position[:, 2] -= p._pickup_clearance()
        p._state_started = p.count - 205
        self.advance(p, 1)
        self.assertEqual(p._planetary_state, "retry_pick")

    def test_released_gear_is_not_chased(self):
        p = self.make_policy("release")
        ee = p.scene["robot"].data.body_state_w[:, 0, :7]
        p._release_position = ee[:, :3].clone()
        p._release_orientation = ee[:, 3:7].clone()
        p.sun_planetary_gear_1.data.root_state_w[:, 0] += 0.01
        self.advance(p, 4)
        torch.testing.assert_close(p._motion_target[:, :2], p._release_position[:, :2])

    def test_legacy_remainder_is_retimed_only_after_all_three_are_seated(self):
        p = self.make_policy("park")
        p._planetary_gear = 3
        p.count = 3000
        p._state_started = 2700
        p._park_start = torch.zeros(1, 7)
        p.initial_pos_left = torch.zeros(7)
        for step in range(8, 15):
            setattr(p, f"count_step_{step}", torch.tensor([2000 + (step - 8) * 100, 2100 + (step - 8) * 100]))
        self.advance(p, 1)
        self.assertTrue(p.planetary_complete)
        self.assertEqual(int(p.count_step_8[0]), 3000)
        self.assertEqual(p.total_time_steps, 3700)
        self.assertEqual(p._phase_last_count, 2999)

    def test_displaced_earlier_gear_blocks_the_remainder(self):
        p = self.make_policy("park")
        p._planetary_gear = 3
        p._state_started = p.count - 300
        p._park_start = torch.zeros(1, 7)
        p.initial_pos_left = torch.zeros(7)
        p.sun_planetary_gear_1.data.root_state_w[:, 0] += 0.003
        self.advance(p, 1)
        self.assertFalse(p.planetary_complete)
        self.assertIn("previously seated gears moved", p.planetary_failure)

    def test_crowded_grasp_rotates_away_from_neighbour(self):
        p = self.make_policy()
        # Seed 43: the nominal -135 degree jaw sweep hits gear two.
        p._held_object(1).data.root_state_w[:, :3] = torch.tensor([[0.582, 0.0855, 0.901]])
        p._held_object(2).data.root_state_w[:, :3] = torch.tensor([[0.6485, 0.0236, 0.901]])
        p._held_object(3).data.root_state_w[:, :3] = torch.tensor([[0.6485, -0.0989, 0.901]])
        p.obj_dict["planetary_carrier"] = p.planetary_carrier
        p.obj_dict["ring_gear"] = SimpleNamespace(
            data=SimpleNamespace(root_state_w=torch.tensor([[0.4973, 0.2335, 0.901, 1.0, 0.0, 0.0, 0.0] + [0.0] * 6]))
        )
        self.assertTrue(p._choose_grasp())
        self.assertAlmostEqual(p._grasp_yaw_delta, 45 * torch.pi / 180)
        self.assertAlmostEqual(p._planetary_opening, 0.055)
        self.assertAlmostEqual(p._grasp_shift, -0.008)

    def make_pin_calibration_policy(self):
        p = self.make_policy()
        p.pin_local_positions = [
            p.gear_to_pin_map[f"sun_planetary_gear_{i}"]["pin_local_pos"] for i in range(1, 4)
        ]
        p.right_arm_entity_cfg = SimpleNamespace(body_ids=[1])
        p.scene["robot"].data.body_state_w = torch.tensor(
            [[[0.48, 0.45, 1.25, 1.0, 0.0, 0.0, 0.0], [0.48, -0.45, 1.25, 1.0, 0.0, 0.0, 0.0]]]
        )
        p.gear_to_pin_map["sun_planetary_gear_3"]["arm"] = "right"
        for i, pos in enumerate(
            ((0.582, 0.0855, 0.901), (0.6485, 0.0236, 0.901), (0.6485, -0.0989, 0.901)), 1
        ):
            p._held_object(i).data.root_state_w[:, :3] = torch.tensor([pos])
        return p

    def assert_mesh_axis_mapping(self, p):
        mesh_axes = torch.tensor(
            [
                [-0.0000000058710575, -0.0539999945163727, 0.0],
                [0.0467653796672821, 0.0270000021457672, 0.0],
                [-0.0467653832435608, 0.0269999965429306, 0.0],
            ]
        )
        carrier = p.planetary_carrier.data.root_state_w[:, :7]
        for gear_id in (1, 2, 3):
            mapping = p.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]
            expected_local = mesh_axes[mapping["pin"]]
            expected_world = carrier[:, :3] + quat_apply(carrier[:, 3:7], expected_local.unsqueeze(0))
            torch.testing.assert_close(mapping["pin_local_pos"], expected_local, atol=1e-8, rtol=0)
            torch.testing.assert_close(mapping["pin_world_pos"], expected_world, atol=1e-7, rtol=0)

    def test_pro_assignment_uses_measured_mesh_axes(self):
        p = self.make_pin_calibration_policy()
        p._assign_planetary_pins()
        self.assert_mesh_axis_mapping(p)

    def test_pro_mesh_axes_follow_carrier_rotation_and_translation(self):
        p = self.make_pin_calibration_policy()
        angle = torch.tensor(torch.pi / 4)
        p.planetary_carrier.data.root_state_w[:, :3] = torch.tensor([[0.411, -0.027, 0.913]])
        p.planetary_carrier.data.root_state_w[:, 3:7] = torch.tensor(
            [[angle.cos(), 0.0, 0.0, angle.sin()]]
        )
        p._assign_planetary_pins()
        self.assert_mesh_axis_mapping(p)

    def test_pin_calibration_preserves_sun_ring_and_scorer_constants(self):
        p = self.make_pin_calibration_policy()
        before = (
            p.SUN_APPROACH_HEIGHT_M,
            p.RING_TRANSFER_TIMEOUT_S,
            p.RING_HOLD_XY_TOLERANCE,
            GalaxeaLabExternalEnv.PLANETARY_GEAR_XY_TOLERANCE,
            GalaxeaLabExternalEnv.CENTRE_GEAR_XY_TOLERANCE,
            GalaxeaLabExternalEnv.RING_GEAR_XY_TOLERANCE,
        )
        self.assertEqual(before[:4], (0.040, 15.0, 0.002, 0.002))
        p._assign_planetary_pins()
        after = (
            p.SUN_APPROACH_HEIGHT_M,
            p.RING_TRANSFER_TIMEOUT_S,
            p.RING_HOLD_XY_TOLERANCE,
            GalaxeaLabExternalEnv.PLANETARY_GEAR_XY_TOLERANCE,
            GalaxeaLabExternalEnv.CENTRE_GEAR_XY_TOLERANCE,
            GalaxeaLabExternalEnv.RING_GEAR_XY_TOLERANCE,
        )
        self.assertEqual(after, before)

    def test_pin_assignment_reserves_the_near_side_for_each_arm(self):
        p = self.make_policy()
        p.pin_local_positions = [p.gear_to_pin_map[f"sun_planetary_gear_{i}"]["pin_local_pos"] for i in range(1, 4)]
        p.right_arm_entity_cfg = SimpleNamespace(body_ids=[1])
        p.scene["robot"].data.body_state_w = torch.tensor(
            [[[0.48, 0.45, 1.25, 1.0, 0.0, 0.0, 0.0], [0.48, -0.45, 1.25, 1.0, 0.0, 0.0, 0.0]]]
        )
        p.gear_to_pin_map["sun_planetary_gear_3"]["arm"] = "right"
        for i, pos in enumerate(((0.582, 0.0855, 0.901), (0.6485, 0.0236, 0.901), (0.6485, -0.0989, 0.901)), 1):
            p._held_object(i).data.root_state_w[:, :3] = torch.tensor([pos])
        p._assign_planetary_pins()
        self.assertEqual(p.gear_to_pin_map["sun_planetary_gear_3"]["pin"], 0)
        self.assertEqual({p.gear_to_pin_map[f"sun_planetary_gear_{i}"]["pin"] for i in (1, 2)}, {1, 2})
        # Keep an already reachable assignment stable when the travel-distance
        # difference is less than a millimetre (the seed 42 layout).
        for i, (pos, pin) in enumerate(zip(((0.49392, 0.29712, 0.901), (0.51022, 0.39521, 0.901)), (1, 2)), 1):
            p._held_object(i).data.root_state_w[:, :3] = torch.tensor([pos])
            p.gear_to_pin_map[f"sun_planetary_gear_{i}"]["pin"] = pin
        p._assign_planetary_pins()
        self.assertEqual(tuple(p.gear_to_pin_map[f"sun_planetary_gear_{i}"]["pin"] for i in range(1, 4)), (1, 2, 0))

    def test_grasp_shift_moves_the_pick_target_along_the_pad(self):
        p = self.make_policy("hover")
        p._grasp_shift = -0.008
        p._motion_target = p._held_object(1).data.root_state_w[:, :3] + p._tcp_offset(None)
        p._motion_target[:, 2] += p._pickup_clearance()
        original = p._motion_target.clone()
        self.advance(p, 2)
        self.assertLess(float(p._motion_target[0, 0]), float(original[0, 0]) - 0.007)
        torch.testing.assert_close(p._motion_target[:, 1:], original[:, 1:])

    def test_off_table_gear_ends_with_failure(self):
        p = self.make_policy("hover")
        p._held_object(1).data.root_state_w[:, 2] = 0.03
        self.advance(p, 1)
        self.assertIn("left the table workspace", p.planetary_failure)

    def test_verified_seating_enters_park(self):
        p = self.make_policy("verify")
        ee = p.scene["robot"].data.body_state_w[:, 0, :7]
        p._release_position = ee[:, :3].clone()
        p._release_orientation = ee[:, 3:7].clone()
        self.advance(p, 11)
        self.assertEqual(p._planetary_state, "park")

    def test_grasp_yaw_rotates_tcp_and_restores_legacy_orientation(self):
        p = self.make_policy()
        del p._tcp_offset
        del p._gripper_down_quat
        p.TCP_offset_x = -0.02086
        p.tcp_offsets = {"left": torch.tensor([[-0.02086, -0.02086, 0.2785]])}
        p._grasp_yaw_delta = 35 * torch.pi / 180
        angle = torch.tensor(35 * torch.pi / 180)
        base = p.tcp_offsets["left"]
        expected = base.clone()
        expected[:, 0] = base[:, 0] * angle.cos() - base[:, 1] * angle.sin()
        expected[:, 1] = base[:, 0] * angle.sin() + base[:, 1] * angle.cos()
        torch.testing.assert_close(p._tcp_offset(p.left_arm_entity_cfg), expected)
        p.planetary_complete = True
        torch.testing.assert_close(p._tcp_offset(p.left_arm_entity_cfg), base)
        torch.testing.assert_close(p._gripper_down_quat(p.left_arm_entity_cfg), torch.tensor([p.GRIPPER_DOWN_QUAT]))


class SunControlTests(PlanetaryControlTests):
    def test_actual_mesh_support_surface_is_nine_mm_above_carrier_origin(self):
        def intersections(name):
            stage = Usd.Stage.Open(str(Path(GALAXEA_LAB_ASSETS_DIR) / "Gearbox" / f"{name}_3x_scale.usd"))
            mesh = UsdGeom.Mesh(next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)))
            self.assertTrue(all(n == 3 for n in mesh.GetFaceVertexCountsAttr().Get()))
            points = np.asarray(mesh.GetPointsAttr().Get())
            vertices = points[np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)]
            a, b = vertices[:, 1, :2] - vertices[:, 0, :2], vertices[:, 2, :2] - vertices[:, 0, :2]
            det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
            valid = abs(det) > 1e-5
            vertices, a, b, det = vertices[valid], a[valid], b[valid], det[valid]
            # This point is inside a sun tooth and above the carrier plate,
            # outside its central 25 mm radius hole (asset coordinates are mm).
            c = np.array([30.0, -4.0]) - vertices[:, 0, :2]
            u = (c[:, 0] * b[:, 1] - c[:, 1] * b[:, 0]) / det
            v = (a[:, 0] * c[:, 1] - a[:, 1] * c[:, 0]) / det
            inside = (u >= -1e-5) & (v >= -1e-5) & (u + v <= 1.00001)
            z = (
                vertices[:, 0, 2]
                + u * (vertices[:, 1, 2] - vertices[:, 0, 2])
                + v * (vertices[:, 2, 2] - vertices[:, 0, 2])
            )
            return z[inside] * 0.001

        self.assertAlmostEqual(float(intersections("sun_planetary_gear").min()), 0.0)
        self.assertAlmostEqual(float(intersections("planetary_carrier").max()), 0.009)

    def centre_score(self, height):
        p = self.make_sun_policy()
        p.sun_planetary_gear_4.data.root_state_w[:, :3] = p._live_pin(4)
        p.sun_planetary_gear_4.data.root_state_w[:, 2] += height
        env = SimpleNamespace(device="cpu", rule_policy=p, sim=SimpleNamespace(get_physics_dt=lambda: p.sim_dt))
        for name in (
            "ASSEMBLY_ANGLE_TOLERANCE",
            "PLANETARY_GEAR_XY_TOLERANCE",
            "PLANETARY_GEAR_Z_TOLERANCE",
            "CENTRE_GEAR_XY_TOLERANCE",
            "CENTRE_GEAR_Z_TOLERANCE",
            "RING_GEAR_XY_TOLERANCE",
            "RING_GEAR_Z_TOLERANCE",
        ):
            setattr(env, name, getattr(GalaxeaLabExternalEnv, name))
        env._pose_matches = lambda *a: GalaxeaLabExternalEnv._pose_matches(env, *a)
        carrier = p.planetary_carrier.data.root_state_w
        env.get_key_points = lambda: (
            [p._live_pin(i) for i in range(1, 4)],
            None,
            [p._held_object(i).data.root_state_w[:, :3] for i in range(1, 5)],
            [p._held_object(i).data.root_state_w[:, 3:7] for i in range(1, 5)],
            carrier[:, :3],
            carrier[:, 3:7],
            carrier[:, :3],
            carrier[:, 3:7],
            None,
            None,
        )
        return int(GalaxeaLabExternalEnv.evaluate_score(env)[0].item())

    def test_scoring_accepts_sun_on_the_actual_support_surface(self):
        self.assertEqual(self.centre_score(0.009), 5)

    def test_scoring_rejects_sun_penetrating_the_carrier_plate(self):
        self.assertEqual(self.centre_score(0.0), 4)

    def make_sun_policy(self, state="search"):
        p = self.make_policy(state)
        p.planetary_complete = p._sun_active = True
        p.sun_complete = False
        p._sun_started = p.count
        p._sun_disturbed_since = None
        p._planetary_gear = 4
        p.gear_to_pin_map["sun_planetary_gear_4"] = {"arm": "left"}
        root = p.planetary_carrier.data.root_state_w.clone()
        root[:, 2] += 0.040
        p.sun_planetary_gear_4 = SimpleNamespace(data=SimpleNamespace(root_state_w=root))
        ee = root[:, :7].clone()
        ee[:, :3] += p._tcp_offset(None)
        ee[:, 2] += 0.011
        p.scene["robot"].data.body_state_w = ee.unsqueeze(1)
        p._motion_target = ee[:, :3].clone()
        p._sun_yaw_anchor = torch.zeros(1)
        p._sun_search_offset = 0.0
        p._sun_search_direction = 1
        p._sun_phase_aligned = True
        p._position_bias = p._orientation_bias = None
        return p

    def advance_sun(self, p, ticks):
        for _ in range(ticks):
            action = p._sun_action()
            p.count += 5
        return action

    def test_sun_contact_slip_does_not_ratchet_the_wrist_target(self):
        p = self.make_sun_policy("transfer")
        ee = p.scene["robot"].data.body_state_w[:, 0, :7]
        gear = p.sun_planetary_gear_4.data.root_state_w
        level = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        orientation, _ = p._sun_target_transform(ee, gear, level)
        p._planetary_state = "search"
        # Contact rolls the wrist while the gear stays blocked by the teeth.
        # The correction must restore the pre-contact grasp, not adopt the slip.
        ee[:, 3:7] = torch.tensor([[0.99875, 0.049979, 0.0, 0.0]])
        corrected, offset = p._sun_target_transform(ee, gear, level)
        torch.testing.assert_close(corrected, orientation)
        self.assertTrue(bool(torch.isfinite(offset).all()))
        # A subsequent pickup establishes a new grasp before contact.
        p._planetary_state = "transfer"
        refreshed, _ = p._sun_target_transform(ee, gear, level)
        torch.testing.assert_close(refreshed, ee[:, 3:7])

    def test_sun_yaw_slip_keeps_correction_toward_the_requested_phase(self):
        for state in ("search", "approach", "phase_align"):
            with self.subTest(state=state):
                p = self.make_sun_policy("transfer")
                ee = p.scene["robot"].data.body_state_w[:, 0, :7]
                gear = p.sun_planetary_gear_4.data.root_state_w
                ee[:, 3:7] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
                gear[:, 3:7] = ee[:, 3:7]
                p._sun_target_transform(ee, gear, ee[:, 3:7].clone())
                p._planetary_state = state
                # Teeth turn the payload inside the shallow grasp. The
                # requested correction remains clockwise after unloading.
                angle = torch.tensor(0.35)
                gear[:, 3:7] = torch.tensor([[angle.div(2).cos(), 0.0, 0.0, angle.div(2).sin()]])
                requested = angle - 0.04
                level = torch.tensor([[requested.div(2).cos(), 0.0, 0.0, requested.div(2).sin()]])
                orientation, _ = p._sun_target_transform(ee, gear, level)
                wrist_yaw = 2 * torch.atan2(orientation[0, 3], orientation[0, 0])
                self.assertLess(float(wrist_yaw), 0.0)
                self.assertAlmostEqual(float(wrist_yaw), -0.04, places=5)

    def test_sun_grasp_is_remeasured_once_after_unloading_each_retry(self):
        p = self.make_sun_policy("transfer")
        ee = p.scene["robot"].data.body_state_w[:, 0, :7]
        gear = p.sun_planetary_gear_4.data.root_state_w
        level = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        original, _ = p._sun_target_transform(ee, gear, level)
        p._planetary_state = "realign"
        ee[:, 3:7] = torch.tensor([[0.9998, 0.0199987, 0.0, 0.0]])
        gear[:, 2] = p._live_pin(4)[:, 2] + 0.049
        target, _ = p._sun_target_transform(ee, gear, level)
        torch.testing.assert_close(target, original)

        gear[:, 2] += 0.003
        refreshed, _ = p._sun_target_transform(ee, gear, level)
        torch.testing.assert_close(refreshed, ee[:, 3:7])
        ee[:, 3:7] = torch.tensor([[0.9992, 0.0399893, 0.0, 0.0]])
        for state in ("realign", "approach", "search"):
            p._planetary_state = state
            target, _ = p._sun_target_transform(ee, gear, level)
            torch.testing.assert_close(target, refreshed)

        p._planetary_state = "realign"
        p._insert_attempt += 1
        p._state_started += 1
        target, _ = p._sun_target_transform(ee, gear, level)
        torch.testing.assert_close(target, ee[:, 3:7])

    def test_seated_sun_can_verify_after_an_arm_limited_clear_retreat(self):
        p = self.make_sun_policy("retreat")
        gear = p.sun_planetary_gear_4.data.root_state_w
        gear[:, :3] = p._live_pin(4)
        gear[:, 2] += 0.009
        robot = p.scene["robot"]
        ee = robot.data.body_state_w[:, 0, :7]
        p._release_position = ee[:, :3].clone()
        p._release_orientation = ee[:, 3:7].clone()
        ee[:, 0] += 0.009
        ee[:, 2] += 0.100
        robot.data.joint_pos_limits = torch.tensor([[[-2.0, 2.0]] * 8])
        robot.data.joint_pos[:, 6] = 1.998
        p._state_started = p.count - 300
        self.advance_sun(p, 1)
        self.assertEqual(p._planetary_state, "verify")
        self.assertFalse(p.sun_complete)

    def make_sun_retreat_with_lateral_residual(self):
        p = self.make_sun_policy("retreat")
        p.sun_planetary_gear_4.data.root_state_w[:, :3] = p._live_pin(4)
        p.sun_planetary_gear_4.data.root_state_w[:, 2] += 0.009
        robot = p.scene["robot"]
        ee = robot.data.body_state_w[:, 0, :7]
        p._release_position = ee[:, :3].clone()
        p._release_orientation = ee[:, 3:7].clone()
        ee[:, 0] += 0.016
        ee[:, 2] += 0.100
        robot.data.joint_pos_limits = torch.tensor([[[-2.0, 2.0]] * 8])
        robot.data.joint_pos[:, 6] = 1.998
        p._state_started = p.count - 300
        return p

    def test_sun_retreat_uses_measured_finger_clearance_at_a_joint_limit(self):
        # Fresh Blackwell seed 25001 lifted 100 mm, with 112 mm finger
        # clearance, but a saturated wrist left 16 mm lateral error.
        p = self.make_sun_retreat_with_lateral_residual()
        self.advance_sun(p, 1)
        self.assertEqual(p._planetary_state, "verify")
        self.assertFalse(p.sun_complete)
        self.advance_sun(p, 10)
        self.assertEqual(p._planetary_state, "park")

    def test_sun_clearance_gate_requires_clearance_lift_limit_and_retained_seats(self):
        for case in ("low_clearance", "short_lift", "large_lateral", "not_limited", "unseated", "too_early"):
            with self.subTest(case=case):
                p = self.make_sun_retreat_with_lateral_residual()
                robot = p.scene["robot"]
                ee = robot.data.body_state_w[:, 0, :7]
                if case == "low_clearance":
                    # Preserve 100 mm relative lift while the absolute pad
                    # envelope is only 15 mm above the mounted gears.
                    ee[:, 2] -= 0.110
                    p._release_position[:, 2] -= 0.110
                elif case == "short_lift":
                    p._release_position[:, 2] += 0.020
                elif case == "large_lateral":
                    ee[:, 0] += 0.015
                elif case == "not_limited":
                    robot.data.joint_pos[:, 6] = 1.9
                elif case == "unseated":
                    p.sun_planetary_gear_2.data.root_state_w[:, 0] += 0.004
                else:
                    p._state_started = p.count - 200
                self.advance_sun(p, 1)
                self.assertEqual(p._planetary_state, "retreat")
                self.assertFalse(p.sun_complete)

    def test_central_approach_enters_search_above_the_planetary_teeth(self):
        p = self.make_sun_policy("approach")
        # The 40 mm waypoint is above the teeth: subsequent descent must
        # use the bounded contact controller before tooth engagement.
        self.advance_sun(p, 5)
        self.assertIn(p._planetary_state, ("search", "mesh_hold"))
        self.assertFalse(p.sun_complete)

    def test_blocked_sun_does_not_release_on_rotation_timer(self):
        p = self.make_sun_policy()
        action, _ = self.advance_sun(p, 130)
        self.assertEqual(p._planetary_state, "search")
        self.assertEqual(float(action[0, -1]), 0.0)
        self.assertFalse(p.sun_complete)
        self.assertLessEqual(abs(p._sun_search_offset), p.SUN_SEARCH_SWEEP_RAD)

    def test_contact_does_not_integrate_vertical_preload(self):
        p = self.make_sun_policy()
        p._position_bias = torch.tensor([[0.0, 0.0, -0.015]])
        self.advance_sun(p, 10)
        self.assertEqual(float(p._position_bias[0, 2]), 0.0)
        ee = p.scene["robot"].data.body_state_w[:, 0, :3]
        self.assertGreaterEqual(float(p._motion_target[0, 2] - ee[0, 2]), -p.SUN_PRELOAD_M - 1e-6)

    def test_sun_preload_does_not_follow_a_wrist_sliding_down_a_blocked_gear(self):
        p = self.make_sun_policy()
        p._sun_touch_started = True
        ee = p.scene["robot"].data.body_state_w[:, 0, :7]
        contact_z = float(ee[0, 2])
        for _ in range(6):
            target = ee[:, :3].clone()
            target[:, 2] -= p.SUN_PRELOAD_M
            p._planetary_command(
                p.left_arm_entity_cfg, p.left_gripper_entity_cfg,
                target, ee[:, 3:7], 0.0, 0.05, contact=True,
            )
            # Contact deflects the wrist while the teeth keep the gear blocked.
            ee[:, 2] -= 0.002
        self.assertGreaterEqual(float(p._motion_target[0, 2]), contact_z - p.SUN_PRELOAD_M - 1e-6)

        # Unloading for realignment must allow the next contact at a new height.
        p._planetary_state = "realign"
        ee[:, 2] += 0.05
        p._planetary_command(
            p.left_arm_entity_cfg, p.left_gripper_entity_cfg,
            ee[:, :3].clone(), ee[:, 3:7], 0.0, 0.05,
        )
        p._motion_target = ee[:, :3].clone()
        p._planetary_state = "search"
        target = ee[:, :3].clone()
        target[:, 2] -= p.SUN_PRELOAD_M
        p._planetary_command(
            p.left_arm_entity_cfg, p.left_gripper_entity_cfg,
            target, ee[:, 3:7], 0.0, 0.05, contact=True,
        )
        torch.testing.assert_close(p._motion_target, target)

    def test_rotation_starts_when_descent_stalls_at_the_teeth(self):
        p = self.make_sun_policy()
        self.advance_sun(p, 12)
        self.assertFalse(p._sun_touch_started)
        self.assertEqual(p._sun_search_offset, 0.0)
        p.sun_planetary_gear_4.data.root_state_w[:, 2] -= 0.008
        p.scene["robot"].data.body_state_w[:, 0, 2] -= 0.008
        self.advance_sun(p, 20)
        self.assertTrue(p._sun_touch_started)
        self.assertGreater(p._sun_search_offset, 0.0)

    def test_contact_preserves_free_space_bias_without_windup(self):
        p = self.make_sun_policy()
        p._position_bias = torch.tensor([[0.001, 0.002, 0.004]])
        self.advance_sun(p, 20)
        torch.testing.assert_close(p._position_bias, torch.tensor([[0.001, 0.002, 0.004]]))

    def test_meshed_sun_requires_stability_before_release(self):
        p = self.make_sun_policy()
        p.sun_planetary_gear_4.data.root_state_w[:, 2] -= 0.02
        p.scene["robot"].data.body_state_w[:, 0, 2] -= 0.02
        p._motion_target[:, 2] -= 0.02
        self.advance_sun(p, 4)
        self.assertEqual(p._planetary_state, "mesh_hold")
        angle = p._sun_search_offset
        self.advance_sun(p, 8)
        self.assertEqual(p._planetary_state, "release")
        self.assertEqual(p._sun_search_offset, angle)
        self.assertFalse(p.sun_complete)
        self.assertFalse(p._seated(4))

    def test_sun_matches_production_centre_tolerances(self):
        p = self.make_sun_policy()
        root = p.sun_planetary_gear_4.data.root_state_w
        root[:, :3] = p._live_pin(4)
        root[:, 2] += 0.009
        self.assertTrue(p._seated(4))
        root[:, 2] += 0.006
        self.assertFalse(p._seated(4))

    def test_gear_that_settles_in_grip_can_release_at_the_seated_tolerance(self):
        p = self.make_sun_policy()
        p.sun_planetary_gear_4.data.root_state_w[:, :3] = p._live_pin(4)
        p.sun_planetary_gear_4.data.root_state_w[:, 0] += 0.003
        p.sun_planetary_gear_4.data.root_state_w[:, 2] += 0.009
        self.advance_sun(p, 12)
        self.assertTrue(p._seated(4))
        self.assertEqual(p._planetary_state, "release")
        self.assertFalse(p.sun_complete)

    def test_sun_grip_effort_is_bounded_and_restored_without_changing_arm_limits(self):
        p = self.make_sun_policy()
        robot = p.scene["robot"]
        p._use_sun_grip_effort()
        p._use_sun_grip_effort()
        self.assertEqual(float(robot.data.joint_effort_limits[0, 7]), 8.0)
        self.assertTrue(bool((robot.data.joint_effort_limits[0, :7] == 100.0).all()))
        p.reset_actuator_settings()
        p.reset_actuator_settings()
        self.assertTrue(bool((robot.data.joint_effort_limits == 100.0).all()))

    def test_slipped_gear_that_lands_seated_does_not_trigger_a_regrasp(self):
        p = self.make_sun_policy("retry_pick")
        p.sun_planetary_gear_4.data.root_state_w[:, :3] = p._live_pin(4)
        p.sun_planetary_gear_4.data.root_state_w[:, 2] += 0.009
        p._choose_grasp = lambda: self.fail("must not regrasp a seated central gear")
        self.advance_sun(p, 1)
        self.assertEqual(p._planetary_state, "release")
        self.assertFalse(p.sun_complete)

    def test_sun_protects_previously_seated_planets(self):
        p = self.make_sun_policy()
        p.sun_planetary_gear_2.data.root_state_w[:, 0] += 0.01
        self.advance_sun(p, 9)
        self.assertEqual(p._planetary_state, "realign")
        self.advance_sun(p, 45)
        self.assertEqual(p._planetary_state, "failed")
        self.assertIn("planetary gear moved", p.planetary_failure)

    def test_failed_sun_seating_does_not_attempt_low_regrasp(self):
        p = self.make_sun_policy("verify")
        p._state_started = p.count - 210
        self.advance_sun(p, 1)
        self.assertEqual(p._planetary_state, "failed")
        self.assertIn("did not settle", p.planetary_failure)

    def test_sun_handoff_waits_for_retreat_and_four_seated_gears(self):
        p = self.make_sun_policy("park")
        p._use_sun_grip_effort()
        p.sun_planetary_gear_4.data.root_state_w[:, :3] = p._live_pin(4)
        p.sun_planetary_gear_4.data.root_state_w[:, 2] += 0.009
        p._state_started = p.count - 260
        p._park_start = torch.zeros(1, 7)
        p.initial_pos_left = torch.zeros(7)
        for step in range(10, 15):
            setattr(p, f"count_step_{step}", torch.tensor([1000 + (step - 10) * 100, 1100 + (step - 10) * 100]))
        self.advance_sun(p, 1)
        self.assertTrue(p.sun_complete)
        self.assertFalse(p._sun_active)
        self.assertTrue(bool((p.scene["robot"].data.joint_effort_limits == 100.0).all()))
        self.assertEqual(int(p.count_step_10[0]), 500)
        self.assertEqual(p.total_time_steps, 1000)

    def test_sun_grasp_is_higher_than_planet_grasp(self):
        p = self.make_sun_policy()
        self.assertAlmostEqual(p._pickup_height_offset(), 0.011)
        self.assertAlmostEqual(p._pickup_clearance(), 0.05)
        p._sun_active = False
        self.assertAlmostEqual(p._pickup_height_offset(), -0.004)
        self.assertAlmostEqual(p._pickup_clearance(), 0.10)

    def test_sun_assignment_does_not_require_a_fourth_pin(self):
        p = self.make_sun_policy()
        del p.gear_to_pin_map["sun_planetary_gear_4"]
        p.sun_planetary_gear_4.data.root_state_w[:, 1] = -0.15
        p._choose_grasp = lambda: True
        p._start_sun()
        self.assertEqual(p.gear_to_pin_map["sun_planetary_gear_4"]["arm"], "right")
        self.assertTrue(p._sun_active)
        self.assertFalse(p.sun_complete)

    def test_low_sun_grasp_cannot_push_fingers_into_planets(self):
        p = self.make_sun_policy()
        p.scene["robot"].data.body_state_w[:, 0, 2] -= 0.018
        self.advance_sun(p, 1)
        self.assertEqual(p._planetary_state, "failed")
        self.assertIn("grasp too low", p.planetary_failure)

    def test_sun_unloads_before_fingers_sweep_into_the_planets(self):
        p = self.make_sun_policy()
        p.sun_planetary_gear_4.data.root_state_w[:, 2] -= 0.019
        p.scene["robot"].data.body_state_w[:, 0, 2] -= 0.026
        self.advance_sun(p, 1)
        self.assertEqual(p._planetary_state, "realign")
        self.assertEqual(p._insert_attempt, 2)

    def test_sun_lift_accepts_lateral_error_only_after_the_gear_lifts(self):
        p = self.make_sun_policy("lift")
        ee = p.scene["robot"].data.body_state_w[:, 0, :7]
        p._grasp_position = ee[:, :3].clone()
        p._grasp_position[:, 0] -= 0.007
        p._grasp_position[:, 2] -= 0.05
        p._pick_height = float(p.sun_planetary_gear_4.data.root_state_w[0, 2])
        self.advance_sun(p, 8)
        self.assertEqual(p._planetary_state, "lift")
        p._pick_height -= 0.05
        self.advance_sun(p, 8)
        self.assertEqual(p._planetary_state, "transfer")


class PhysicsProfileTests(unittest.TestCase):
    def test_fast_profile_does_not_mutate_shared_robot_defaults(self):
        for bundle in (GALAXEA_R1_PRO_BUNDLE, GALAXEA_R1_LITE_BUNDLE):
            with self.subTest(robot=bundle.name):
                shared = bundle.articulation_cfg
                before = shared.to_dict()
                cfg = SimpleNamespace(robot_bundle=bundle, robot_cfg=shared)
                use_fast_physics(cfg)
                self.assertEqual(shared.to_dict(), before)
                self.assertIsNot(cfg.robot_cfg, shared)
                for props in (cfg.robot_cfg.spawn.rigid_props, cfg.robot_cfg.spawn.articulation_props):
                    self.assertEqual(props.solver_position_iteration_count, 32)
                    self.assertEqual(props.solver_velocity_iteration_count, 8)
                    # Restore these two fields to check every other robot
                    # setting (collisions, gains, initial pose) is preserved.
                    props.solver_position_iteration_count = 128
                    props.solver_velocity_iteration_count = 128
                self.assertEqual(cfg.robot_cfg.to_dict(), before)
                # Changing the private config afterward must not leak to another run.
                cfg.robot_cfg.spawn.rigid_props.max_contact_impulse = 123
                self.assertEqual(shared.to_dict(), before)

    def test_fast_profile_rejects_an_unvalidated_robot(self):
        cfg = SimpleNamespace(
            robot_bundle=GALAXEA_R1_BUNDLE,
            robot_cfg=GALAXEA_R1_BUNDLE.articulation_cfg,
        )
        with self.assertRaisesRegex(ValueError, "--fast_physics"):
            use_fast_physics(cfg)


suite = unittest.TestSuite([
    unittest.defaultTestLoader.loadTestsFromTestCase(PlanetaryControlTests),
    # Inherit the fixture, without rerunning the planetary checks twice.
    unittest.TestSuite(SunControlTests(name) for name in SunControlTests.__dict__ if name.startswith("test_")),
    unittest.defaultTestLoader.loadTestsFromTestCase(PhysicsProfileTests),
])
result = unittest.TextTestRunner(verbosity=2).run(suite)
print(f"PLANETARY_TESTS_PASSED={result.wasSuccessful()}", flush=True)
# Kit's fast shutdown exits the process; assert before shutting it down.
if not result.wasSuccessful():
    raise AssertionError("R1Pro planetary regression checks failed")
app.close()
