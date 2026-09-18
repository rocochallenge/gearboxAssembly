"""R1 Lite feedback geometry and six-axis IK regression checks.

Run after sourcing scripts/env.sh:
    python scripts/test_r1lite_feedback.py --headless
"""

import argparse
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from isaaclab.utils.math import quat_apply
from Galaxea_Lab_External import GALAXEA_LAB_ASSETS_DIR
from Galaxea_Lab_External.robots.r1_lite_feedback_policy import R1LiteFeedbackPolicy
from Galaxea_Lab_External.robots.galaxea_robots import GALAXEA_R1_LITE_CFG
from Galaxea_Lab_External.robots.robot_bundles import GALAXEA_R1_LITE_BUNDLE, GALAXEA_R1_PRO_BUNDLE


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
