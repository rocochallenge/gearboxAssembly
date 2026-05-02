# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert the Galaxea R1_Lite URDF into a USD usable by Isaac Lab.

The vendor URDF references meshes via ``package://mobiman/urdf/R1_Lite/meshes/*.STL``.
Isaac Sim's URDF importer cannot resolve ``package://`` outside a ROS environment,
so this script rewrites those refs to ``../meshes/*.STL`` in a temp copy of the URDF
(in the same directory as the original, so the relative path resolves) and runs
``isaaclab.sim.converters.UrdfConverter`` against the temp copy. The committed URDF
is never modified. The temp copy is removed before exit.

Run with the project's Isaac Sim env active:

    conda deactivate 2>/dev/null || true
    source scripts/env.sh
    python scripts/convert_r1_lite_urdf.py

Output: ``source/Galaxea_Lab_External/assets/Robots/R1_Lite/r1_lite.usd``.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert the R1_Lite URDF into USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# default to headless when no display flag is given
if not args_cli.headless and not getattr(args_cli, "livestream", 0):
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import pathlib
import tempfile

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
URDF_DIR = PROJECT_ROOT / "source" / "Galaxea_Lab_External" / "assets" / "Robots" / "R1_Lite" / "urdf"
URDF_NAME = "mmp_revB_invconfig_upright_a1x.urdf"
USD_DIR = PROJECT_ROOT / "source" / "Galaxea_Lab_External" / "assets" / "Robots" / "R1_Lite"
USD_NAME = "r1_lite.usd"

PACKAGE_PREFIX = "package://mobiman/urdf/R1_Lite/meshes/"
RELATIVE_PREFIX = "../meshes/"


def main() -> None:
    src_urdf = URDF_DIR / URDF_NAME
    if not src_urdf.is_file():
        raise FileNotFoundError(f"Source URDF not found: {src_urdf}")

    print("-" * 80)
    print(f"Source URDF:  {src_urdf}")
    print(f"Output USD:   {USD_DIR / USD_NAME}")
    print("-" * 80)

    # Read URDF and rewrite mesh refs.
    text = src_urdf.read_text(encoding="utf-8")
    if PACKAGE_PREFIX not in text:
        raise RuntimeError(
            f"Expected to find {PACKAGE_PREFIX!r} in the URDF; the source layout may have changed."
        )
    rewritten = text.replace(PACKAGE_PREFIX, RELATIVE_PREFIX)

    # Write the temp URDF in the same directory so '../meshes/' resolves.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".urdf",
        prefix="r1_lite_converted_",
        dir=URDF_DIR,
        delete=False,
    ) as fh:
        tmp_urdf = pathlib.Path(fh.name)
        fh.write(rewritten)

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
            # is misleading): True → parse <mimic> tags and apply PhysxMimicJointAPI,
            # False (default) → ignore <mimic> entirely, joint2 becomes independent.
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
    finally:
        try:
            os.unlink(tmp_urdf)
        except FileNotFoundError:
            pass

    print("-" * 80)


if __name__ == "__main__":
    main()
    simulation_app.close()
