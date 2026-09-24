# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert the Galaxea R1Pro (2026, G1Z gripper) URDF into a USD usable by Isaac Lab.

Vendor source: https://github.com/userguide-galaxea/URDF ``R1Pro/urdf_r1pro_g1z_2026``,
staged under ``source/Galaxea_Lab_External/assets/Robots/R1_Pro/``.

The committed URDF is never modified. A temp copy is written next to it (so relative
mesh paths resolve) with three patches applied, then fed to
``isaaclab.sim.converters.UrdfConverter``:

1. ``package://r1pro_urdf/meshes/`` mesh refs -> ``../meshes/``. Isaac Sim's URDF
   importer cannot resolve ``package://`` outside a ROS environment.

2. The three ``wheel_motor_joint*`` joints are ``continuous``. The importer maps
   them to revolute joints with unbounded limits, which PhysX then rejects every
   time Isaac Lab writes joint limits (``setLimitParams() only supports limit
   angles in range [-2Pi, 2Pi]``). They are zero-locked by the actuator config
   anyway, so we make them ``revolute`` with ``[-2pi, 2pi]`` limits.

3. The finger links' ``<collision>`` meshes are replaced by one flat pad box each
   (measured from the vendor STLs, extended down to the tine tip). The importer's convex hull of the full
   finger mesh spans from the slider block (which crosses the gripper centreline)
   down to the tine tip, so at pad height the two hulls overlap by ~5 cm and
   squeeze anything between the pads back out. With boxes the inner faces sit
   where the visual pads are (y = 0 in the gripper frame when closed).

The gripper finger joints are handled the same way as for R1_Lite, in the
*committed* URDF rather than here: the vendor file ships them with
``lower="0" upper="0" effort="0"`` and no ``<mimic>``; ``r1pro_2026.urdf`` in this
repo carries ``finger_joint1`` in ``[0, 0.065]``, ``finger_joint2`` in ``[-0.065, 0]``
with ``<mimic joint="..._finger_joint1" multiplier="-1.0">``, and
``convert_mimic_joints_to_normal_joints=True`` below lets the importer parse the
mimic. Only finger joint 1 is actuated (``.*_gripper_finger_joint1``); q = 0 is
closed, +q opens, exactly like R1_Lite. After conversion this script rewrites the
mimic joints' compliance to rigid (see ``_harden_mimic_joints``): the importer emits
them as a soft 25 Hz spring, under which finger 2 lags finger 1 by centimetres.

Run with the project's Isaac Sim env active:

    conda deactivate 2>/dev/null || true
    source scripts/env.sh
    python scripts/convert_r1_pro_urdf.py

Output: ``source/Galaxea_Lab_External/assets/Robots/R1_Pro/r1_pro.usd`` plus the
``configuration/`` sublayers (``make_instanceable=True`` bundle, same layout as R1_Lite).
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert the R1Pro URDF into USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# default to headless when no display flag is given
if not args_cli.headless and not getattr(args_cli, "livestream", 0):
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
import pathlib
import tempfile
import xml.etree.ElementTree as ET

from pxr import Usd

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
ROBOT_DIR = PROJECT_ROOT / "source" / "Galaxea_Lab_External" / "assets" / "Robots" / "R1_Pro"
URDF_DIR = ROBOT_DIR / "urdf"
URDF_NAME = "r1pro_2026.urdf"
USD_DIR = ROBOT_DIR
USD_NAME = "r1_pro.usd"

PACKAGE_PREFIX = "package://r1pro_urdf/meshes/"
RELATIVE_PREFIX = "../meshes/"

# Finger collision boxes in each finger link's own frame, from the vendor STL extents
# (finger1: pad x in [-0.046, 0.010], y in [0.0256, 0.0571], z in [-0.105, -0.02]; the
# 1.5 cm tine below it is narrower). One flat pad per finger, extended down to the tine
# tip (z = -0.1201): a 2.4 cm gear on the table is then pinched across its full
# thickness by flat faces. With a separate small tine box the gear was held on the
# tooth flanks only and popped out upward as the fingers kept closing. Finger2 is
# mirrored in x and y. Each entry: (center xyz, size xyz).
FINGER1_BOXES = (
    ((-0.018, 0.0414, -0.0700), (0.056, 0.0315, 0.1001)),   # pad, down to the tip
)
FINGER2_BOXES = tuple(((-cx, -cy, cz), size) for (cx, cy, cz), size in FINGER1_BOXES)


def _box_finger_collisions(root: ET.Element) -> int:
    """Replace the finger links' mesh collisions with pad + tine boxes. Returns #links patched."""
    patched = 0
    for link in root.findall("link"):
        name = link.get("name")
        if not name.endswith(("_gripper_finger_link1", "_gripper_finger_link2")):
            continue
        boxes = FINGER1_BOXES if name.endswith("1") else FINGER2_BOXES
        for col in link.findall("collision"):
            link.remove(col)
        for center, size in boxes:
            col = ET.SubElement(link, "collision")
            origin = ET.SubElement(col, "origin")
            origin.set("xyz", " ".join(f"{v:.4f}" for v in center))
            origin.set("rpy", "0 0 0")
            geom = ET.SubElement(col, "geometry")
            box = ET.SubElement(geom, "box")
            box.set("size", " ".join(f"{v:.4f}" for v in size))
        patched += 1
    return patched


def _harden_mimic_joints(physics_usd: pathlib.Path) -> int:
    """Make the importer's PhysX mimic joints rigid. Returns #joints patched.

    For the ``<mimic>`` on the prismatic finger joints the Isaac Sim 5.1 importer applies
    ``PhysxMimicJointAPI`` with ``naturalFrequency = 25`` and ``dampingRatio = 0.005``,
    i.e. a soft, almost undamped spring between the two finger coordinates: finger 2 then
    lags finger 1 by centimetres and oscillates. ``naturalFrequency = 0`` makes the mimic a
    hard constraint (PhysX ``PxArticulationMimicJoint`` semantics), which is what a URDF
    mimic means.
    """
    stage = Usd.Stage.Open(str(physics_usd))
    patched = 0
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if name.startswith("physxMimicJoint:") and name.endswith(":naturalFrequency"):
                attr.Set(0.0)
                damping = prim.GetAttribute(name.replace(":naturalFrequency", ":dampingRatio"))
                if damping:
                    damping.Set(0.0)
                patched += 1
    if patched:
        stage.GetRootLayer().Save()
    return patched


def _bound_continuous_joints(root: ET.Element) -> int:
    """Turn ``continuous`` joints into ``revolute`` with +-2pi limits. Returns #patched."""
    patched = 0
    for joint in root.findall("joint"):
        if joint.get("type") != "continuous":
            continue
        joint.set("type", "revolute")
        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        limit.set("lower", f"{-2 * math.pi}")
        limit.set("upper", f"{2 * math.pi}")
        if limit.get("effort") is None:
            limit.set("effort", "50")
        if limit.get("velocity") is None:
            limit.set("velocity", "20")
        patched += 1
    return patched


def _rewrite_mesh_paths(root: ET.Element) -> int:
    """Rewrite package:// mesh refs to paths relative to the temp URDF. Returns #rewritten."""
    rewritten = 0
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        if fn.startswith(PACKAGE_PREFIX):
            mesh.set("filename", RELATIVE_PREFIX + fn[len(PACKAGE_PREFIX):])
            rewritten += 1
    return rewritten


def main() -> None:
    src_urdf = URDF_DIR / URDF_NAME
    if not src_urdf.is_file():
        raise FileNotFoundError(f"Source URDF not found: {src_urdf}")

    print("-" * 80)
    print(f"Source URDF:  {src_urdf}")
    print(f"Output USD:   {USD_DIR / USD_NAME}")
    print("-" * 80)

    tree = ET.parse(src_urdf)
    root = tree.getroot()

    n_mesh = _rewrite_mesh_paths(root)
    if n_mesh == 0:
        raise RuntimeError(
            f"Expected to find {PACKAGE_PREFIX!r} mesh refs in the URDF; the source layout may have changed."
        )
    n_wheels = _bound_continuous_joints(root)
    n_boxes = _box_finger_collisions(root)
    if n_boxes != 4:
        raise RuntimeError(f"Expected to patch 4 finger links' collisions, patched {n_boxes}")
    if len(root.findall("joint/mimic")) != 2:
        raise RuntimeError("Expected the committed URDF to carry a <mimic> on both *_gripper_finger_joint2")
    print(f"Rewrote {n_mesh} mesh refs; bounded {n_wheels} continuous joints; boxed {n_boxes} finger link collisions.")

    # Write the temp URDF in the same directory so '../meshes/' resolves.
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".urdf",
        prefix="r1_pro_converted_",
        dir=URDF_DIR,
        delete=False,
    ) as fh:
        tmp_urdf = pathlib.Path(fh.name)
        tree.write(fh, encoding="utf-8", xml_declaration=True)

    try:
        cfg = UrdfConverterCfg(
            asset_path=str(tmp_urdf),
            usd_dir=str(USD_DIR),
            usd_file_name=USD_NAME,
            fix_base=True,
            merge_fixed_joints=False,
            force_usd_conversion=True,
            make_instanceable=True,
            self_collision=False,
            collider_type="convex_hull",
            # IsaacLab forwards this to URDF importer's `parse_mimic` (the field name
            # is misleading): True → parse <mimic> tags (finger joint 2 mimics joint 1,
            # see the committed URDF), False → ignore <mimic> entirely.
            convert_mimic_joints_to_normal_joints=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                drive_type="force",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=1050.0,
                    damping=100.0,
                ),
            ),
        )
        converter = UrdfConverter(cfg)
        print(f"Generated USD: {converter.usd_path}")
        physics_layer = USD_DIR / "configuration" / f"{pathlib.Path(USD_NAME).stem}_physics.usd"
        n_mimic = _harden_mimic_joints(physics_layer)
        if n_mimic != 2:
            raise RuntimeError(f"Expected to harden 2 mimic joints in {physics_layer}, found {n_mimic}")
        print(f"Hardened {n_mimic} PhysX mimic joints in {physics_layer.name}.")
    finally:
        try:
            os.unlink(tmp_urdf)
        except FileNotFoundError:
            pass

    print("-" * 80)


if __name__ == "__main__":
    main()
    simulation_app.close()
