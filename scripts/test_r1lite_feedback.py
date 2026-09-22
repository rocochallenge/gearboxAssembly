"""R1 Lite feedback geometry and six-axis IK regression checks.

Run after sourcing scripts/env.sh:
    python scripts/test_r1lite_feedback.py --headless
"""

import argparse
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import h5py
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from isaaclab.utils.math import quat_apply
from Galaxea_Lab_External import GALAXEA_LAB_ASSETS_DIR
from Galaxea_Lab_External.robots.r1_lite_feedback_policy import R1LiteFeedbackPolicy
from Galaxea_Lab_External.robots.r1_lite_ring import R1LiteRingMixin
from Galaxea_Lab_External.robots.r1_pro_planetary import R1ProPlanetaryMixin
from Galaxea_Lab_External.robots.galaxea_robots import GALAXEA_R1_LITE_CFG
from Galaxea_Lab_External.robots.robot_bundles import GALAXEA_R1_LITE_BUNDLE, GALAXEA_R1_PRO_BUNDLE
from Galaxea_Lab_External.tasks.direct.galaxea_lab_external.galaxea_lab_external_env import GalaxeaLabExternalEnv


class LiteFeedbackTests(unittest.TestCase):
    def test_lite_bundle_selects_feedback_without_changing_pro(self):
        self.assertIs(GALAXEA_R1_LITE_BUNDLE.rule_policy_class, R1LiteFeedbackPolicy)
        self.assertEqual(GALAXEA_R1_PRO_BUNDLE.rule_policy_class.__name__, "R1ProRulePolicy")

    def policy(self):
        p = R1LiteFeedbackPolicy.__new__(R1LiteFeedbackPolicy)
        p.device = "cpu"
        p.sim = SimpleNamespace(device="cpu")
        p.planetary_complete = False
        p._planetary_state = "insert"
        p._grasp_yaw_delta = 0.0
        p.left_arm_entity_cfg = SimpleNamespace(joint_ids=list(range(6)), body_ids=[0])
        p.TCP_offset_x = 0.0
        down = p._gripper_down_quat()
        tcp = torch.tensor([[0.08165 + 0.03689 + p.FINGERTIP_EXTENSION, 0.0, 0.0]])
        p.tcp_offsets = {"left": -quat_apply(down, tcp)}
        return p

    def test_lite_gripper_axis_points_down_and_grasp_shift_is_horizontal(self):
        p = self.policy()
        orientation = p._gripper_down_quat()
        direction = quat_apply(orientation, torch.tensor([p.GRIPPER_AXIS_LOCAL]))
        shift = quat_apply(orientation, torch.tensor([p.GRASP_SHIFT_AXIS_LOCAL]))
        torch.testing.assert_close(direction, torch.tensor([[0.0, 0.0, -1.0]]), atol=1e-6, rtol=0)
        self.assertLess(abs(float(shift[0, 2])), 1e-6)

    def test_fingertip_envelope_matches_vendor_mesh_and_joint_origin(self):
        p = self.policy()
        path = Path(GALAXEA_LAB_ASSETS_DIR) / "Robots/R1_Lite/meshes/left_gripper_finger_link1.STL"
        data = path.read_bytes()
        count = int.from_bytes(data[80:84], "little")
        dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
        vertices = np.frombuffer(data, dtype=dtype, count=count, offset=84)["vertices"].reshape(-1, 3)
        corners = p._sun_fingertip_corners(p.left_arm_entity_cfg)
        expected_tip = 0.08165 + 0.03689 + float(vertices[:, 0].max())
        torch.testing.assert_close(corners[:, 0], torch.full((4,), expected_tip), atol=1e-6, rtol=0)
        distal = vertices[vertices[:, 0] > vertices[:, 0].max() - 0.025]
        self.assertGreaterEqual(p.FINGER_PAD_OUTER_Y + 1e-6, 0.013453 + float(distal[:, 1].max()))
        self.assertGreaterEqual(p.FINGER_PAD_HALF_Z + 1e-6, float(np.abs(distal[:, 2] + 0.00012059).max()))

    def test_sun_pickup_sets_tip_at_upper_rim(self):
        p = self.policy()
        p._sun_active = True
        tip_height = p._pickup_height_offset() + p.FINGERTIP_EXTENSION - p.FINGER_TIP_X
        self.assertAlmostEqual(tip_height, 0.018)

    def test_sun_does_not_lift_before_the_jaws_reach_the_rim(self):
        p = self.policy()
        p._sun_active = True
        p._planetary_state = "close"
        p._grasp_position = torch.tensor([[0.58, -0.21, 1.08]])
        p.scene = {"robot": SimpleNamespace(data=SimpleNamespace(
            joint_pos=torch.tensor([[0.041]]), joint_vel=torch.tensor([[-0.008]]),
        ))}
        p._transition = lambda state: setattr(p, "_planetary_state", state)
        p._stable = lambda condition, duration=0.2: condition
        p._planetary_command = lambda *args, **kwargs: None
        ee = torch.cat((p._grasp_position, p._gripper_down_quat()), dim=-1)
        p._pick_action(p.left_arm_entity_cfg, SimpleNamespace(joint_ids=[0]), ee, None, None,
                       p._gripper_down_quat(), 2.0, 0.05)
        self.assertEqual(p._planetary_state, "close")

    def test_sun_closing_target_stays_close_to_the_measured_jaws(self):
        p = self.policy()
        p._sun_active = True
        p.scene = {"robot": SimpleNamespace(data=SimpleNamespace(joint_pos=torch.tensor([[0.031]])))}
        with patch.object(R1ProPlanetaryMixin, "_planetary_command") as command:
            p._planetary_command(p.left_arm_entity_cfg, SimpleNamespace(joint_ids=[0]),
                                 torch.zeros(1, 3), p._gripper_down_quat(), 0.0, 0.05)
        opening = command.call_args.args[4]
        self.assertGreaterEqual(opening, 0.03059)
        self.assertLess(opening, 0.031)

    def test_sun_releases_only_from_a_low_aligned_waypoint(self):
        cases = (
            (0.0002, 0.025, 0.0, 0.0, "release"),
            (0.0002, 0.050, 0.0, 0.0, "approach"),  # High drops can tumble.
            (0.0012, 0.025, 0.0, 0.0, "approach"),
            (0.0002, 0.025, 0.04, 0.0, "approach"),
            (0.0002, 0.025, 0.0, 0.03, "approach"),
        )
        for xy_error, height, tilt, phase_error, expected_state in cases:
            with self.subTest(xy=xy_error, height=height, tilt=tilt, phase=phase_error):
                p = self.policy()
                p._sun_active = True
                p._planetary_state = "approach"
                centre = torch.tensor([[0.5, 0.0, 0.901]])
                gear = torch.tensor([[0.5 + xy_error, 0.0, 0.901 + height, 1.0, 0.0, 0.0, 0.0]])
                down = p._gripper_down_quat()
                offset = p._tcp_offset(p.left_arm_entity_cfg).clone()
                offset[:, 2] += p._pickup_height_offset()
                ee = torch.cat((gear[:, :3] + offset, down), dim=-1)
                p._gear_errors = lambda gear_id: (xy_error, height, tilt)
                p._sun_mesh_yaw = lambda yaw, position: yaw + phase_error
                p._stable = lambda condition, duration=0.2: condition
                p._transition = lambda state: setattr(p, "_planetary_state", state)
                p._planetary_command = lambda *args, **kwargs: None
                p._sun_insert(p.left_arm_entity_cfg, SimpleNamespace(joint_ids=[6]),
                              ee, gear, centre, down, 1.0, 0.05)
                self.assertEqual(p._planetary_state, expected_state)
                if expected_state == "release":
                    torch.testing.assert_close(p._release_position, ee[:, :3])

    def test_ring_grasp_clears_lid_and_engages_the_boss(self):
        path = Path(GALAXEA_LAB_ASSETS_DIR) / "Gearbox/ring_gear_3x_scale.stl"
        data = path.read_bytes()
        count = int.from_bytes(data[80:84], "little")
        dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
        vertices = np.frombuffer(data, dtype=dtype, count=count, offset=84)["vertices"].reshape(-1, 3) * 0.001
        radius = np.linalg.norm(vertices[:, :2], axis=1)
        lid_top = float(vertices[radius > 0.036, 2].max())
        boss_top = float(vertices[:, 2].max())
        tip = R1LiteRingMixin.RING_GRASP_TIP_HEIGHT
        self.assertGreater(tip - lid_top, 0.002)
        self.assertGreater(boss_top - tip, 0.010)

    def test_public_policy_selects_ring_grasp_and_hover_geometry(self):
        p = self.policy()
        p._ring_active = True
        p._planetary_gear = 5
        p.obj_dict = {"ring_gear": SimpleNamespace(data=SimpleNamespace(
            root_state_w=torch.tensor([[0.6, -0.15, 0.902]]),
        ))}
        self.assertTrue(p._choose_grasp())
        tip_height = p._pickup_height_offset() + p.FINGERTIP_EXTENSION - p.FINGER_TIP_X
        self.assertGreater(tip_height, 0.02945)
        hover_tip = p._pickup_clearance() + p.FINGERTIP_EXTENSION - p.FINGER_TIP_X
        self.assertGreater(hover_tip, 0.055)

    def test_five_points_do_not_finish_while_the_ring_is_held(self):
        policy = SimpleNamespace(count=100, total_time_steps=16000, assembly_complete=False)
        env = SimpleNamespace(
            rule_policy=policy, sim=SimpleNamespace(get_physics_dt=lambda: 0.01),
            evaluate_score=lambda: (torch.tensor([5]), None), SUCCESS_SCORE=5,
            episode_length_buf=torch.tensor([20]), max_episode_length=3200,
        )
        # Use the real done gate: the old scorer ended the episode as soon as
        # the held ring crossed its positional tolerance.
        done, timeout = GalaxeaLabExternalEnv._get_dones(env)
        self.assertFalse(bool(done.item()))
        self.assertFalse(bool(timeout.item()))
        policy.assembly_complete = True
        done, _ = GalaxeaLabExternalEnv._get_dones(env)
        self.assertTrue(bool(done.item()))
        policy.assembly_complete = False
        policy.count = policy.total_time_steps
        done, _ = GalaxeaLabExternalEnv._get_dones(env)
        self.assertTrue(bool(done.item()))

    def test_rim_grasp_allows_tooth_overlap_before_opening_touches_planets(self):
        p = self.policy()
        p._sun_active = True
        planet = SimpleNamespace(data=SimpleNamespace(
            root_state_w=torch.tensor([[0.5, 0.0, 0.910, 1.0, 0.0, 0.0, 0.0]])
        ))
        p._held_object = lambda gear_id: planet
        centre = torch.tensor([[0.5, 0.0, 0.901]])
        offset = p._tcp_offset(p.left_arm_entity_cfg).clone()
        offset[:, 2] += p._pickup_height_offset()
        down = p._gripper_down_quat()
        ee = torch.cat((centre + offset + torch.tensor([[0.0, 0.0, 0.040]]), down), dim=-1)
        floor, top, clearance = p._sun_mesh_clearance(p.left_arm_entity_cfg, ee, offset, down, centre)
        self.assertLessEqual(floor, top - 0.006)
        self.assertGreater(clearance, 0.002)

    def test_sun_grip_preserves_lite_limits_and_never_raises_lower_limits(self):
        p = self.policy()
        p.left_gripper_entity_cfg = SimpleNamespace(joint_ids=[6])
        p.gear_to_pin_map = {"sun_planetary_gear_4": {"arm": "left"}}
        calls = []
        limits = torch.tensor([[87.0] * 6 + [60.0]])
        p.scene = {"robot": SimpleNamespace(
            data=SimpleNamespace(joint_effort_limits=limits),
            write_joint_effort_limit_to_sim=lambda value, joint_ids: calls.append((value.clone(), joint_ids)),
        )}
        p._use_sun_grip_effort()
        self.assertEqual(calls[0][1], [6])
        torch.testing.assert_close(calls[0][0], torch.tensor([[60.0]]))
        p.reset_actuator_settings()
        torch.testing.assert_close(calls[1][0], torch.tensor([[60.0]]))
        self.assertIsNone(p._sun_saved_grip_effort)

    def test_unobstructed_lite_grasp_uses_neutral_yaw(self):
        p = self.policy()
        p._planetary_gear = 1
        p.obj_dict = {"sun_planetary_gear_1": SimpleNamespace(
            data=SimpleNamespace(root_state_w=torch.tensor([[0.5, 0.3, 0.901]]))
        )}
        self.assertTrue(p._choose_grasp())
        self.assertEqual(p._grasp_yaw_delta, 0.0)
        self.assertEqual(p._grasp_shift, 0.0)
        self.assertLessEqual(p._planetary_opening, 0.05)

    def test_six_joint_ik_stays_within_step_and_joint_limits(self):
        p = self.policy()
        limits = torch.tensor([[[-2.0, 2.0]] * 6])
        limits[:, 0, 1] = 0.02
        p.scene = {"robot": SimpleNamespace(data=SimpleNamespace(
            joint_pos_limits=limits,
            root_state_w=torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
        ))}
        result = p._refine_joint_target(
            p.left_arm_entity_cfg, torch.eye(6).unsqueeze(0), torch.zeros(1, 6), torch.full((1, 6), 0.5)
        )
        self.assertEqual(result.shape, (1, 6))
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertLessEqual(float(result.abs().max()), p.PLANETARY_MAX_JOINT_STEP + 1e-6)
        self.assertLessEqual(float(result[0, 0]), 0.018001)

    def test_ring_transfer_keeps_the_joint_step_limit_after_sun_completion(self):
        p = self.policy()
        p.planetary_complete = p.sun_complete = p._ring_active = True
        p._planetary_state = "transfer"
        p.scene = {"robot": SimpleNamespace(data=SimpleNamespace(
            joint_pos_limits=torch.tensor([[[-2.0, 2.0]] * 6]),
        ))}
        result = p._refine_joint_target(
            p.left_arm_entity_cfg, torch.eye(6).unsqueeze(0), torch.zeros(1, 6), torch.full((1, 6), 0.5)
        )
        self.assertLessEqual(float(result.abs().max()), p.PLANETARY_MAX_JOINT_STEP + 1e-6)

    def test_ring_completion_requires_retention_after_parking(self):
        p = self.policy()
        p._ring_active = True
        p.initial_pos_left = torch.zeros(6)
        p._park_start = torch.ones(1, 6)
        p._stable_since = None
        p.sim_dt = 0.01
        p._seated = lambda gear_id: True
        gripper = SimpleNamespace(joint_ids=[6])
        p.count = 250
        p._ring_park(p.left_arm_entity_cfg, gripper, "left", 2.5)
        self.assertFalse(p.assembly_complete)
        p.count = 305
        p._ring_park(p.left_arm_entity_cfg, gripper, "left", 3.05)
        self.assertTrue(p.assembly_complete)

    def test_ring_stages_before_unlocking_torso_and_restores_limits_on_reset(self):
        p = self.policy()
        p._planetary_state = "transfer"
        p._ring_torso_retracted = False
        p.gear_to_pin_map = {"ring_gear": {"arm": "left"}}
        positions = torch.zeros(1, 8)
        positions[:, 7] = -0.39
        limits = torch.zeros(1, 8, 2)
        limits[:, 7, :] = -0.39
        writes = []
        p.scene = {"robot": SimpleNamespace(
            data=SimpleNamespace(joint_pos=positions, joint_pos_limits=limits),
            find_joints=lambda name: ([7], [name]),
            write_joint_position_limit_to_sim=lambda values, joint_ids: writes.append((values.clone(), joint_ids)),
        )}
        p._stable = lambda condition, duration=0.2: condition
        p._transition = lambda state: setattr(p, "_planetary_state", state)
        p._planetary_command = lambda *args, **kwargs: (torch.zeros(1, 7), list(range(7)))
        centre = torch.tensor([[0.45, 0.0, 0.91]])
        ring = torch.tensor([[0.65, 0.18, 0.96, 1.0, 0.0, 0.0, 0.0]])
        ee = torch.cat((ring[:, :3] + torch.tensor([[0.0, 0.0, 0.19]]), p._gripper_down_quat()), dim=-1)
        gripper = SimpleNamespace(joint_ids=[6])
        p._ring_stage(p.left_arm_entity_cfg, gripper, ee, ring, centre, 0.0, 0.05)
        self.assertEqual(p._planetary_state, "stage")
        self.assertEqual(writes, [])

        ring[:, :3] = centre + torch.tensor([[0.12, 0.13, 0.05]])
        ee[:, :3] = ring[:, :3] + torch.tensor([[0.0, 0.0, 0.19]])
        p._ring_stage(p.left_arm_entity_cfg, gripper, ee, ring, centre, 2.0, 0.05)
        self.assertEqual(p._planetary_state, "retract_torso")
        self.assertEqual(writes[0][1], [7])
        torch.testing.assert_close(writes[0][0], torch.tensor([[[-0.44, -0.10]]]))
        action, ids = p._ring_stage(p.left_arm_entity_cfg, gripper, ee, ring, centre, 1.5, 0.05)
        self.assertEqual(ids, list(range(8)))
        self.assertAlmostEqual(float(action[0, -1]), -0.245, places=5)
        self.assertFalse(p._ring_torso_retracted)
        positions[:, 7] = -0.13
        p._ring_stage(p.left_arm_entity_cfg, gripper, ee, ring, centre, 3.3, 0.05)
        self.assertTrue(p._ring_torso_retracted)
        p.reset_actuator_settings()
        torch.testing.assert_close(writes[-1][0], torch.tensor([[[-0.39, -0.39]]]))

    def test_ring_release_requires_sustained_seating_without_more_preload(self):
        p = self.policy()
        p._planetary_state = "mesh_hold"
        p._release_position = torch.tensor([[0.45, 0.0, 1.11]])
        p._release_orientation = p._gripper_down_quat()
        p._ring_hold_position = torch.tensor([[0.45, 0.0, 0.919]])
        p._stable_since = None
        p.sim_dt = 0.01
        p._gear_errors = lambda gear_id: (0.0003, 0.009, 0.001)
        p._seated = lambda gear_id: True
        p._transition = lambda state: setattr(p, "_planetary_state", state)
        p.scene = {"robot": SimpleNamespace(data=SimpleNamespace(joint_pos=torch.tensor([[0.032]])))}
        commands = []
        p._planetary_command = lambda *args, **kwargs: commands.append((args, kwargs))
        ring = torch.tensor([[0.45, 0.0, 0.919, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.2]])
        gripper = SimpleNamespace(joint_ids=[0])
        for count in (100, 140):
            p.count = count
            p._ring_hold(p.left_arm_entity_cfg, gripper, None, ring, (count - 100) * .01, .05)
            self.assertEqual(p._planetary_state, "mesh_hold")
        ring[:, 0] += 0.003
        p.count = 155
        p._ring_hold(p.left_arm_entity_cfg, gripper, None, ring, .55, .05)
        self.assertEqual(p._planetary_state, "mesh_hold")
        ring[:, 0] -= 0.003
        for count in (160, 215):
            p.count = count
            p._ring_hold(p.left_arm_entity_cfg, gripper, None, ring, (count - 100) * .01, .05)
        self.assertEqual(p._planetary_state, "release")
        for args, kwargs in commands:
            torch.testing.assert_close(args[2], p._release_position)
            self.assertTrue(kwargs["contact"])

    def test_ring_contact_hold_preserves_gravity_compensation(self):
        p = self.policy()
        p._planetary_state = "search"
        p._ring_torso_retracted = True
        p._ring_depth_sample = (0.0, 0.009)
        p._ring_touching = False
        p._ring_yaw_anchor = torch.zeros(1)
        p._ring_search_offset = 0.0
        p._gear_errors = lambda gear_id: (0.0002, 0.009, 0.001)
        p._seated = lambda gear_id: True
        p._transition = lambda state: setattr(p, "_planetary_state", state)
        p._ring_hold = lambda *args: "holding"
        p._position_bias = torch.tensor([[0.0, 0.0, 0.0044]])
        p._orientation_bias = torch.tensor([[0.001, 0.0, 0.0]])
        ring = torch.tensor([[0.45, 0.0, 0.919, 1.0, 0.0, 0.0, 0.0]])
        ee = torch.cat((ring[:, :3] + p._tcp_offset(p.left_arm_entity_cfg), p._gripper_down_quat()), dim=-1)
        result = p._ring_insert(p.left_arm_entity_cfg, None, ee, ring, ring[:, :3], 0.1, 0.05)
        self.assertEqual(result, "holding")
        self.assertEqual(p._planetary_state, "mesh_hold")
        torch.testing.assert_close(p._motion_target, ee[:, :3])
        torch.testing.assert_close(p._position_bias, torch.tensor([[0.0, 0.0, 0.0044]]))
        torch.testing.assert_close(p._orientation_bias, torch.tensor([[0.001, 0.0, 0.0]]))

    def recording_env(self):
        obs = {name: torch.zeros((1, 240, 320, 3), dtype=torch.uint8)
               for name in ("head_rgb", "left_hand_rgb", "right_hand_rgb")}
        obs.update({name: torch.zeros((1, 240, 320, 1))
                    for name in ("head_depth", "left_hand_depth", "right_hand_depth")})
        actions = {}
        for side in ("left", "right"):
            for kind in ("pos", "vel"):
                obs[f"{side}_arm_joint_{kind}"] = torch.zeros((1, 6))
                obs[f"{side}_gripper_joint_{kind}"] = torch.zeros(1)
            actions[f"{side}_arm_action"] = torch.zeros((1, 6))
            actions[f"{side}_gripper_action"] = torch.zeros(1)
        obs["torso_joint_pos"] = torch.tensor([[0.15, 0.0, -0.13]])
        obs["torso_joint_vel"] = torch.zeros((1, 3))
        actions["torso_action"] = torch.tensor([[0.15, 0.0, -0.10]])
        data = {f"/observations/{name}": [] for name in obs}
        data.update({f"/actions/{name}": [] for name in actions})
        data.update({"/score": [], "/current_time": []})
        return SimpleNamespace(
            obs=obs, act=actions, data_dict=data, score=0, score_tensor=torch.tensor([0]),
            rule_policy=SimpleNamespace(count=11100), sim=SimpleNamespace(get_physics_dt=lambda: 0.01),
            physics_dt=0.01, step_dt=0.05, _torso_joint_idx=[0, 1, 2],
            robot=SimpleNamespace(joint_names=["torso_joint1", "torso_joint2", "torso_joint3"]),
            cfg=SimpleNamespace(robot_bundle=GALAXEA_R1_LITE_BUNDLE, robot_cfg=GALAXEA_R1_LITE_CFG),
        )

    def test_recording_preserves_torso_targets_and_six_joint_arm_arrays(self):
        env = self.recording_env()
        env.score = 5
        env.score_tensor.fill_(5)
        GalaxeaLabExternalEnv._record_data(env)
        # Physics updates the same state/target buffers in place. Earlier
        # frames must retain their values, including when recording on CPU.
        env.obs["torso_joint_pos"][0, 2] = -0.12
        env.act["torso_action"][0, 2] = -0.09
        GalaxeaLabExternalEnv._record_data(env)
        with tempfile.TemporaryDirectory() as directory:
            env.save_hdf5_file_name = str(Path(directory) / "assembly.hdf5")
            GalaxeaLabExternalEnv._write_hdf5_episode(env, success=True)
            with h5py.File(env.save_hdf5_file_name) as recording:
                self.assertEqual(recording["actions/left_arm_action"].shape, (2, 6))
                self.assertEqual(recording["actions/right_arm_action"].shape, (2, 6))
                np.testing.assert_allclose(recording["observations/torso_joint_pos"][:], [[0.15, 0.0, -0.13], [0.15, 0.0, -0.12]])
                np.testing.assert_allclose(recording["actions/torso_action"][:], [[0.15, 0.0, -0.10], [0.15, 0.0, -0.09]])
                self.assertEqual(list(recording.attrs["torso_joint_names"]), env.robot.joint_names)

    def test_recording_uses_current_score_when_reward_override_skips_scalar_cache(self):
        env = self.recording_env()
        expected = [0, 3, 5]
        for score in expected:
            # Evaluation overrides reuse the score from _get_dones without
            # running the base _get_rewards that refreshes env.score.
            env.score_tensor.fill_(score)
            env.rule_policy.count += 5
            GalaxeaLabExternalEnv._record_data(env)
        self.assertEqual(env.score, 0)
        with tempfile.TemporaryDirectory() as directory:
            env.save_hdf5_file_name = str(Path(directory) / "assembly.hdf5")
            GalaxeaLabExternalEnv._write_hdf5_episode(env, success=True)
            with h5py.File(env.save_hdf5_file_name) as recording:
                np.testing.assert_array_equal(recording["score"][:], expected)
        # Recorded values must also survive the next episode's reset.
        env.score_tensor.zero_()
        self.assertEqual(env.data_dict["/score"], expected)

    def test_transfer_ik_retains_the_yaw_constraint_for_lite(self):
        p = self.policy()
        p._planetary_state = "transfer"
        p.scene = {"robot": SimpleNamespace(data=SimpleNamespace(
            joint_pos_limits=torch.tensor([[[-2.0, 2.0]] * 6]),
            root_state_w=torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
        ))}
        desired = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.05]])
        result = p._refine_joint_target(p.left_arm_entity_cfg, torch.eye(6).unsqueeze(0), torch.zeros(1, 6), desired)
        torch.testing.assert_close(result, desired)

    def test_planetary_transfer_aligns_teeth_before_lowering_onto_the_pin(self):
        """The seed-43 failure had three seated gears with incompatible phases."""
        p = self.policy()
        p._planetary_state = "transfer"
        p._planetary_gear = 3
        p._planetary_opening = p.GRIPPER_OPEN_POS
        p.diff_ik_controller = None
        centre = torch.tensor([[0.5, 0.0, 0.901]])
        pin = centre + torch.tensor([[0.0, -0.054, 0.0]])
        first = torch.tensor([[0.4529, 0.0268, 0.911, math.cos(0.05), 0.0, 0.0, math.sin(0.05)]])
        gear = torch.cat((pin + torch.tensor([[0.0, 0.0, p.PLANETARY_TRANSFER_CLEARANCE]]),
                          torch.tensor([[math.cos(-0.135), 0.0, 0.0, math.sin(-0.135)]])), dim=-1)
        p.planetary_carrier = SimpleNamespace(data=SimpleNamespace(root_state_w=centre))
        p._held_object = lambda gear_id: SimpleNamespace(data=SimpleNamespace(root_state_w=first))
        p._gear_errors = lambda gear_id: (0.0, p.PLANETARY_TRANSFER_CLEARANCE, 0.0)
        p._stable = lambda condition, duration=0.2: condition
        p._transition = lambda state: setattr(p, "_planetary_state", state)
        commands = []
        p._planetary_command = lambda *args, **kwargs: commands.append(args)
        down = p._gripper_down_quat(p.left_arm_entity_cfg)
        ee = torch.cat((gear[:, :3] + p._tcp_offset(p.left_arm_entity_cfg), down), dim=-1)

        p._insertion_action(p.left_arm_entity_cfg, SimpleNamespace(joint_ids=[6]),
                            ee, gear, pin, down, 1.0, 0.05)

        # A position-only gate would start lowering while the teeth disagree.
        self.assertEqual(p._planetary_state, "transfer")
        from isaaclab.utils.math import quat_conjugate, quat_mul
        correction = quat_mul(commands[0][3], quat_conjugate(ee[:, 3:7]))
        predicted = quat_mul(correction, gear[:, 3:7])[0]
        yaw = math.atan2(2 * (predicted[0] * predicted[3] + predicted[1] * predicted[2]),
                         1 - 2 * (predicted[2] ** 2 + predicted[3] ** 2))
        self.assertLess(yaw, -0.29)

    def test_retry_unloads_vertically_without_requesting_pro_hover_height(self):
        p = self.policy()
        p._pick_attempt = 1
        p._planetary_gear = 1
        p.count, p.sim_dt = 100, 0.01
        ee = torch.tensor([[0.574, 0.043, 1.07, 0.7071068, 0.0, 0.7071068, 0.0]])
        before = ee.clone()
        p._retry_pick(ee, "missed grasp")
        self.assertEqual(p._planetary_state, "retry_pick")
        self.assertEqual(p._pick_attempt, 2)
        torch.testing.assert_close(p._retreat_position, torch.tensor([[0.574, 0.043, 1.105]]))
        torch.testing.assert_close(ee, before)

    def test_no_grasp_when_both_finger_sweeps_are_blocked(self):
        p = self.policy()
        p._planetary_gear = 1
        p.obj_dict = {
            name: SimpleNamespace(data=SimpleNamespace(root_state_w=torch.tensor([[0.5, 0.3, 0.901]])))
            for name in ("sun_planetary_gear_1", "ring_gear")
        }
        self.assertFalse(p._choose_grasp())

    def test_far_edge_pickup_is_reachable_with_configured_torso(self):
        """Regression for seed 43: x=0.65 near the midline was unreachable."""
        path = Path(GALAXEA_LAB_ASSETS_DIR) / "Robots/R1_Lite/urdf/mmp_revB_invconfig_upright_a1x.urdf"
        joints = {j.find("child").get("link"): j for j in ET.parse(path).getroot().findall("joint")}
        initial = GALAXEA_R1_LITE_CFG.init_state.joint_pos
        p = self.policy()
        target = np.array([0.65, 0.0, 0.902 + 0.08165 + 0.03689 + p.FINGERTIP_EXTENSION + p._pickup_clearance()])
        rotation = Rotation.from_euler("y", math.pi / 2).as_matrix()
        for side in ("left", "right"):
            with self.subTest(side=side):
                chain, lower, upper = [], [], []
                child = f"{side}_arm_link6"
                while child in joints:
                    joint = joints[child]
                    chain.insert(0, joint)
                    child = joint.find("parent").get("link")
                transforms = []
                for joint in chain:
                    name = joint.get("name")
                    origin = joint.find("origin")
                    transform = np.eye(4)
                    transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                    transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
                    axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ") if joint.find("axis") is not None else np.zeros(3)
                    index = int(name[-1]) - 1 if name.startswith(f"{side}_arm_joint") else None
                    if index is not None:
                        lower.append(float(joint.find("limit").get("lower")) + 0.003)
                        upper.append(float(joint.find("limit").get("upper")) - 0.003)
                    transforms.append((transform, axis, index, initial.get(name, 0.0), joint.get("type")))

                def residual(q):
                    pose = np.eye(4)
                    for transform, axis, index, fixed, kind in transforms:
                        pose = pose @ transform
                        value = q[index] if index is not None else fixed
                        if kind in ("revolute", "continuous"):
                            pose[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec(axis * value).as_matrix()
                    return np.r_[pose[:3, 3] - target, 0.2 * Rotation.from_matrix(rotation @ pose[:3, :3].T).as_rotvec()]

                ready = np.array([initial[f"{side}_arm_joint{i}"] for i in range(1, 7)])
                fit = least_squares(residual, ready, bounds=(lower, upper), max_nfev=150)
                self.assertLess(np.linalg.norm(fit.fun[:3]), 0.001)
                self.assertLess(np.linalg.norm(fit.fun[3:]) / 0.2, 0.01)


result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LiteFeedbackTests))
print(f"LITE_FEEDBACK_TESTS_PASSED={result.wasSuccessful()}", flush=True)
if not result.wasSuccessful():
    raise AssertionError("R1 Lite feedback regression checks failed")
app.close()
