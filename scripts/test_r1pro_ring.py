"""R1Pro ring dispatch, grasp geometry, and completion regression checks.

Run after sourcing scripts/env.sh:
    python scripts/test_r1pro_ring.py --headless
"""

import argparse
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from isaaclab.utils.math import quat_apply, quat_mul
from Galaxea_Lab_External.robots.r1_lite_rule_policy import R1LiteRulePolicy
from Galaxea_Lab_External.robots.r1_pro_rule_policy import R1ProRulePolicy
from Galaxea_Lab_External.tasks.direct.galaxea_lab_external.galaxea_lab_external_env import GalaxeaLabExternalEnv


class ProRingTests(unittest.TestCase):
    def policy(self):
        p = R1ProRulePolicy.__new__(R1ProRulePolicy)
        p.planetary_complete = True
        p.sun_complete = True
        p._sun_active = False
        p._ring_active = False
        p.TCP_offset_x = 0.0
        return p

    def test_score_five_while_ring_is_held_does_not_complete_pro(self):
        p = self.policy()
        p.count = 100
        p.total_time_steps = 16000
        env = SimpleNamespace(
            rule_policy=p,
            sim=SimpleNamespace(get_physics_dt=lambda: 0.01),
            evaluate_score=lambda: (torch.tensor([5]), None),
            SUCCESS_SCORE=5,
            episode_length_buf=torch.tensor([20]),
            max_episode_length=3200,
        )
        done, timeout = GalaxeaLabExternalEnv._get_dones(env)
        self.assertFalse(bool(done.item()))
        self.assertFalse(bool(timeout.item()))
        p.ring_complete = True
        done, _ = GalaxeaLabExternalEnv._get_dones(env)
        self.assertTrue(bool(done.item()))

    def test_fourth_completion_dispatches_feedback_ring(self):
        p = self.policy()
        started = []
        p._start_ring = lambda: started.append(True)
        p._ring_action = lambda: ("feedback", [7])
        with patch.object(R1LiteRulePolicy, "get_action", return_value=("legacy", [])):
            self.assertEqual(p.get_action(), ("feedback", [7]))
        self.assertEqual(started, [True])

    def test_completed_pro_assembly_does_not_resume_timed_stages(self):
        p = self.policy()
        p.ring_complete = True
        with patch.object(R1LiteRulePolicy, "get_action", return_value=("legacy", [])):
            self.assertEqual(p.get_action(), (None, None))

    def test_pro_boss_grasp_clears_ring_lid(self):
        p = self.policy()
        p._ring_active = True
        # Converted G1Z pad tips lie 6.95 mm above the calibrated virtual TCP.
        tip_height = p._pickup_height_offset() + 0.00695
        self.assertGreater(tip_height, 0.02745 + 0.002)
        self.assertLess(tip_height, 0.04745 - 0.010)

    def test_ring_ik_keeps_seven_joint_commands_bounded(self):
        p = self.policy()
        p._ring_active = True
        p._planetary_state = "search"
        arm = SimpleNamespace(joint_ids=list(range(7)))
        p.scene = {
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    joint_pos_limits=torch.tensor([[[-2.0, 2.0]] * 7]),
                )
            )
        }
        action = p._refine_joint_target(arm, torch.eye(6, 7).unsqueeze(0), torch.zeros(1, 7), torch.full((1, 7), 0.5))
        self.assertEqual(action.shape, (1, 7))
        self.assertTrue(bool(torch.isfinite(action).all()))
        self.assertLessEqual(float(action.abs().max()), p.PLANETARY_MAX_JOINT_STEP + 1e-6)

    def test_ring_contact_does_not_accumulate_wrist_tilt(self):
        p = self.policy()
        p._ring_active = True
        p._planetary_state = "approach"
        p._ring_torso_retracted = True
        p._gear_errors = lambda gear_id: (0.0001, 0.038, 0.001)
        p._ring_mesh_yaw = lambda yaw, centre: yaw
        p._stable = lambda *args: False
        p._tcp_offset = lambda arm: torch.tensor([[0.0, 0.0, 0.3]])
        commands = []
        p._planetary_command = lambda *args, **kwargs: commands.append(args)
        ring = torch.tensor([[0.45, 0.0, 0.948, 1.0, 0.0, 0.0, 0.0]])
        orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        offset = p._tcp_offset(None)
        ee = torch.cat((ring[:, :3] + offset, orientation), dim=-1)
        p._ring_insert(None, None, ee, ring, ring[:, :3], 0.0, 0.05)
        # Contact holds the ring level while the wrist pitches about its grasp.
        # Recomputing the grasp from this deflection would retain the new tilt.
        tilt = torch.tensor([[0.9950042, 0.0, 0.0998334, 0.0]])
        ee = torch.cat((ring[:, :3] + quat_apply(tilt, offset), quat_mul(tilt, orientation)), dim=-1)
        p._ring_insert(None, None, ee, ring, ring[:, :3], 1.0, 0.05)
        torch.testing.assert_close(commands[-1][3], orientation, atol=1e-6, rtol=0)

    def test_small_seated_settling_does_not_lift_the_ring_again(self):
        for xy, should_release in ((0.0016, True), (0.0021, False)):
            with self.subTest(xy=xy):
                p = self.policy()
                p._planetary_state = "mesh_hold"
                p._stable_since = None
                p.sim_dt = 0.01
                p._release_position = torch.tensor([[0.45, 0.0, 1.22]])
                p._release_orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
                p._ring_hold_position = torch.tensor([[0.4512, 0.0, 0.9195]])
                p._gear_errors = lambda gear_id: (xy, 0.009, 0.004)
                p._seated = lambda gear_id: True
                p._transition = lambda state: setattr(p, "_planetary_state", state)
                p._planetary_command = lambda *args, **kwargs: None
                p.scene = {"robot": SimpleNamespace(data=SimpleNamespace(joint_pos=torch.tensor([[0.033]])))}
                gripper = SimpleNamespace(joint_ids=[0])
                ring = torch.tensor([[0.45 + xy, 0.0, 0.919, 1.0, 0.0, 0.0, 0.0]])
                p.count = 100
                p._ring_hold(None, gripper, None, ring, 0.0, 0.05)
                self.assertEqual(p._planetary_state, "mesh_hold")
                p.count = 155
                p._ring_hold(None, gripper, None, ring, 0.55, 0.05)
                self.assertEqual(p._planetary_state == "release", should_release)
                self.assertFalse(p.assembly_complete)

    def test_parking_requires_retention_and_keeps_pro_arm_shape(self):
        for disturbed in (False, True):
            with self.subTest(disturbed=disturbed):
                p = self.policy()
                p.device = "cpu"
                p._ring_active = True
                p.initial_pos_left = torch.zeros(7)
                p._park_start = torch.ones(1, 7)
                p._stable_since = None
                p.sim_dt = 0.01
                p._seated = lambda gear_id: not (disturbed and gear_id == 4)
                failures = []
                p._fail_planetary = failures.append
                arm, gripper = SimpleNamespace(joint_ids=list(range(7))), SimpleNamespace(joint_ids=[7])
                p.count = 250
                result = p._ring_park(arm, gripper, "left", 2.5)
                self.assertFalse(p.assembly_complete)
                p.count = 305
                p._ring_park(arm, gripper, "left", 3.05)
                self.assertEqual(p.assembly_complete, not disturbed)
                self.assertEqual(bool(failures), disturbed)
                if not disturbed:
                    self.assertEqual(result[0].shape, (1, 8))
                    self.assertEqual(result[1], list(range(8)))


result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProRingTests))
print(f"PRO_RING_TESTS_PASSED={result.wasSuccessful()}", flush=True)
if not result.wasSuccessful():
    raise AssertionError("R1Pro ring regression checks failed")
app.close()
