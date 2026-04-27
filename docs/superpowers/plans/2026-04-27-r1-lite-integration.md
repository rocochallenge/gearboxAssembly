# R1_Lite Robot Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the new Galaxea R1_Lite URDF into a USD usable by Isaac Lab 2.3.0, and add a single-pointer "robot bundle" mechanism so the user can switch the gearbox-assembly tasks between the existing R1 and the new R1_Lite by editing one symbol. R1 remains the default.

**Architecture:** A `RobotBundle` frozen dataclass aggregates the per-robot ArticulationCfg, three CameraCfgs, joint dof-name strings, and a rule-policy class reference. Two bundle instances (`GALAXEA_R1_BUNDLE`, `GALAXEA_R1_LITE_BUNDLE`) live in a new `robots/robot_bundles.py`; a module-level `ACTIVE_ROBOT_BUNDLE` pointer (default = R1) is what each env_cfg reads from. The R1_Lite USD is produced by a one-shot project script (`scripts/convert_r1_lite_urdf.py`) that rewrites `package://` mesh refs in memory and runs `isaaclab.sim.converters.UrdfConverter`. The forked R1_Lite rule policies (`r1_lite_rule_policy.py`, `r1_lite_recovery_rule_policy.py`) exist with mechanical joint-name renames so the env loads; their R1-tuned geometry constants are documented as future tune-me work.

**Tech Stack:** Python 3.11, Isaac Sim 5.1 binary (sourced via `scripts/env.sh`), Isaac Lab 2.3.0 (`isaaclab.sim.converters.UrdfConverter`, `isaaclab.assets.ArticulationCfg`, `isaaclab.sensors.CameraCfg`), uv-managed `.venv`, Git LFS.

**Branch:** All work continues on `dev` (already 4 commits ahead of `main` after the design-doc commit).

**Spec reference:** `docs/superpowers/specs/2026-04-27-r1-lite-integration-design.md`

---

## Pre-flight notes for the implementer

- **No Co-Authored-By trailer.** Repo convention: every commit message uses the body alone, no Claude co-author footer.
- **Isaac Sim must be active.** Tasks 4 and 14 launch Isaac Sim. Run `conda deactivate 2>/dev/null || true; source scripts/env.sh` from the repo root before those tasks. Tasks that only edit Python source can run with the .venv inactive.
- **Working directory.** All commands assume `cd /scratch/SM/AAAI_challenge/gearboxAssembly` (or absolute paths).
- **TDD note.** This codebase has no Python test suite — the validation is "the import succeeds / the simulator launches without error / the visual scene matches expectations". Each task that changes Python ends with an import smoke test. Task 14 covers end-to-end validation.
- **Order matters for Tasks 1 and 2.** `.gitattributes` MUST be committed before staging the STL files, otherwise the STL blobs land in regular git and need `git lfs migrate` to fix.

---

### Task 1: Add `*.stl` / `*.STL` to repo `.gitattributes` for LFS

**Files:**
- Modify: `.gitattributes`

- [ ] **Step 1: Read the current `.gitattributes`**

```bash
cat /scratch/SM/AAAI_challenge/gearboxAssembly/.gitattributes
```

Expected: 12 lines, ending with `*.hdf5 filter=lfs diff=lfs merge=lfs -text`.

- [ ] **Step 2: Append the two STL rules**

Use the Edit tool to append two lines after the existing `*.hdf5` line. The final two lines added:

```
*.stl filter=lfs diff=lfs merge=lfs -text
*.STL filter=lfs diff=lfs merge=lfs -text
```

- [ ] **Step 3: Verify the rule applies to a sample R1_Lite STL (without staging it yet)**

```bash
git -C /scratch/SM/AAAI_challenge/gearboxAssembly check-attr --all -- source/Galaxea_Lab_External/assets/Robots/R1_Lite/meshes/base_link.STL
```

Expected output (4 lines):
```
source/Galaxea_Lab_External/assets/Robots/R1_Lite/meshes/base_link.STL: diff: lfs
source/Galaxea_Lab_External/assets/Robots/R1_Lite/meshes/base_link.STL: merge: lfs
source/Galaxea_Lab_External/assets/Robots/R1_Lite/meshes/base_link.STL: text: unset
source/Galaxea_Lab_External/assets/Robots/R1_Lite/meshes/base_link.STL: filter: lfs
```

If `filter: lfs` is missing, `.gitattributes` was not edited correctly — fix and re-check.

- [ ] **Step 4: Commit**

```bash
git -C /scratch/SM/AAAI_challenge/gearboxAssembly add .gitattributes
git -C /scratch/SM/AAAI_challenge/gearboxAssembly commit -m "Track *.stl and *.STL in Git LFS.

The R1_Lite robot ships ~18 MB of STL meshes; route them through LFS to
match how the existing USD/OBJ assets are handled."
```

---

### Task 2: Stage and commit the R1_Lite raw assets

**Files:**
- New (already on disk, untracked): `source/Galaxea_Lab_External/assets/Robots/R1_Lite/`

The R1_Lite URDF, meshes, config, and launch files are already on disk under that path (added by the user before this work started). They need to be `git add`-ed AFTER the LFS rule is committed.

- [ ] **Step 1: Verify the tree is still untracked**

```bash
git -C /scratch/SM/AAAI_challenge/gearboxAssembly status --short
```

Expected: `?? source/Galaxea_Lab_External/assets/Robots/R1_Lite/` (and nothing else).

- [ ] **Step 2: Stage the entire R1_Lite tree**

```bash
git -C /scratch/SM/AAAI_challenge/gearboxAssembly add source/Galaxea_Lab_External/assets/Robots/R1_Lite/
```

- [ ] **Step 3: Verify STL files routed through LFS**

```bash
git -C /scratch/SM/AAAI_challenge/gearboxAssembly lfs ls-files | grep R1_Lite | head -5
```

Expected: 5 STL pointer lines, each beginning with a 12-char SHA followed by `*` and the path.

If this prints nothing, the STL files were staged as regular blobs (LFS rule not applied). Roll back with `git restore --staged source/Galaxea_Lab_External/assets/Robots/R1_Lite/`, double-check Task 1 was committed, and retry.

- [ ] **Step 4: Commit**

```bash
git -C /scratch/SM/AAAI_challenge/gearboxAssembly commit -m "Add R1_Lite robot URDF and meshes.

Vendor drop from Galaxea: SolidWorks-exported URDF
(mmp_revB_invconfig_upright_a1x.urdf, 34 links / 33 joints) plus 33 STL
meshes, the joint-name yaml, ROS launch files, and the package.xml.
STLs are LFS-tracked. Mesh refs in the URDF use package://mobiman/...
and will be rewritten in-memory by the upcoming conversion script."
```

---

### Task 3: Add the URDF→USD conversion script

**Files:**
- Create: `scripts/convert_r1_lite_urdf.py`

The script reads the committed URDF, rewrites `package://mobiman/urdf/R1_Lite/meshes/` to `../meshes/` in memory, writes a temporary URDF copy alongside the original (so `../meshes/` resolves), then runs `isaaclab.sim.converters.UrdfConverter` to produce `assets/Robots/R1_Lite/r1_lite.usd`.

- [ ] **Step 1: Write the script**

Create `/scratch/SM/AAAI_challenge/gearboxAssembly/scripts/convert_r1_lite_urdf.py` with this exact content:

```python
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
```

- [ ] **Step 2: Verify the script imports cleanly (without launching Isaac Sim)**

```bash
python -c "import ast; ast.parse(open('/scratch/SM/AAAI_challenge/gearboxAssembly/scripts/convert_r1_lite_urdf.py').read())"
```

Expected: silent (exit 0). Any output indicates a syntax error.

- [ ] **Step 3: Commit**

```bash
git -C /scratch/SM/AAAI_challenge/gearboxAssembly add scripts/convert_r1_lite_urdf.py
git -C /scratch/SM/AAAI_challenge/gearboxAssembly commit -m "Add scripts/convert_r1_lite_urdf.py.

One-shot URDF -> USD conversion for R1_Lite. Reads the vendor URDF,
rewrites package://mobiman/... mesh refs to relative ../meshes/ in a
temp copy (committed URDF stays untouched), and invokes
isaaclab.sim.converters.UrdfConverter with fix_base=True so the
mobile base is welded for tabletop tasks. Output:
assets/Robots/R1_Lite/r1_lite.usd."
```

---

### Task 4: Run the conversion script and commit the USD

**Files:**
- New (generated): `source/Galaxea_Lab_External/assets/Robots/R1_Lite/r1_lite.usd`

This task requires Isaac Sim 5.1 active.

- [ ] **Step 1: Activate the project env**

```bash
cd /scratch/SM/AAAI_challenge/gearboxAssembly
conda deactivate 2>/dev/null || true
source scripts/env.sh
```

Expected: banner printing `[env.sh] venv:`, `[env.sh] ISAAC_SIM:`, etc. If `scripts/env.sh` aborts ("A conda env is active"), heed its message.

- [ ] **Step 2: Run the conversion**

```bash
python scripts/convert_r1_lite_urdf.py
```

Expected output (the last line that matters):
```
Generated USD: /scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/assets/Robots/R1_Lite/r1_lite.usd
```

The Isaac Sim startup will print many `[Info]` lines and possibly inotify warnings — those are non-fatal. The script exits 0 on success.

- [ ] **Step 3: Verify the USD exists and looks plausible**

```bash
ls -la /scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/assets/Robots/R1_Lite/r1_lite.usd
file /scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/assets/Robots/R1_Lite/r1_lite.usd
```

Expected: file size > 1 MB; `file` reports binary data (not ASCII). The PXR-USDC magic bytes are not human-readable.

- [ ] **Step 4: Verify the temp URDF copy was cleaned up**

```bash
ls /scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/assets/Robots/R1_Lite/urdf/r1_lite_converted_*.urdf 2>&1
```

Expected: `ls: cannot access ...: No such file or directory`. If a temp URDF was left behind (e.g., conversion crashed mid-flight), delete it manually before staging.

- [ ] **Step 5: Verify LFS routing**

```bash
git -C /scratch/SM/AAAI_challenge/gearboxAssembly check-attr --all -- source/Galaxea_Lab_External/assets/Robots/R1_Lite/r1_lite.usd
```

Expected: 4 lines, last one `filter: lfs`.

- [ ] **Step 6: Stage and commit**

```bash
git -C /scratch/SM/AAAI_challenge/gearboxAssembly add source/Galaxea_Lab_External/assets/Robots/R1_Lite/r1_lite.usd
git -C /scratch/SM/AAAI_challenge/gearboxAssembly lfs ls-files | grep r1_lite.usd
```

The `lfs ls-files` output should show `r1_lite.usd` with a SHA prefix.

```bash
git -C /scratch/SM/AAAI_challenge/gearboxAssembly commit -m "Add converted r1_lite.usd.

Output of scripts/convert_r1_lite_urdf.py against the vendor URDF.
fix_base=True so the mobile base is welded; wheels stay articulated
as DOFs. LFS-tracked via the *.usd rule in .gitattributes."
```

---

### Task 5: Add `GALAXEA_R1_LITE_CFG` and R1_Lite camera cfgs

**Files:**
- Modify: `source/Galaxea_Lab_External/Galaxea_Lab_External/robots/galaxea_robots.py`

Three new module-level constants are appended after the existing `GALAXEA_HAND_CAMERA_CFG`:

- [ ] **Step 1: Read the file end to know what to append after**

```bash
tail -20 /scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/robots/galaxea_robots.py
```

Expected: the file ends after the closing `)` of `GALAXEA_HAND_CAMERA_CFG`.

- [ ] **Step 2: Append the new constants**

Use the Edit tool to append the following block to the end of `galaxea_robots.py` (after the closing `)` of `GALAXEA_HAND_CAMERA_CFG`):

```python


##
# R1_Lite (vendor URDF: mmp_revB_invconfig_upright_a1x). Coexists with R1.
##

GALAXEA_R1_LITE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{GALAXEA_LAB_ASSETS_DIR}/Robots/R1_Lite/r1_lite.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            linear_damping=0.1,
            angular_damping=0.1,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=192,
            max_contact_impulse=1e3,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=192,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.05, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "left_arm_joint1": -20.0 / 180.0 * math.pi,
            "left_arm_joint2": 100.6 / 180.0 * math.pi,
            "left_arm_joint3": -24.0 / 180.0 * math.pi,
            "left_arm_joint4": 17.8 / 180.0 * math.pi,
            "left_arm_joint5": 38.7 / 180.0 * math.pi,
            "left_arm_joint6": 20.1 / 180.0 * math.pi,
            "left_gripper_finger_joint1": 0.04,
            "right_arm_joint1": -20.0 / 180.0 * math.pi,
            "right_arm_joint2": 100.8 / 180.0 * math.pi,
            "right_arm_joint3": -22.0 / 180.0 * math.pi,
            "right_arm_joint4": -40 / 180.0 * math.pi,
            "right_arm_joint5": -67.6 / 180.0 * math.pi,
            "right_arm_joint6": 18.1 / 180.0 * math.pi,
            "right_gripper_finger_joint1": 0.04,
            "torso_joint1": 0.0,
            "torso_joint2": 0.0,
            "torso_joint3": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "r1_lite_arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_joint[1-5]"],
            stiffness=1050.0,
            damping=100.0,
            friction=0.0,
            armature=0.1,
            effort_limit_sim=87,
            velocity_limit_sim=10,
        ),
        "r1_lite_eefs": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm_joint6"],
            stiffness=1050.0,
            damping=100.0,
            friction=0.0,
            armature=0.1,
            effort_limit_sim=87,
            velocity_limit_sim=10,
        ),
        "r1_lite_grippers": ImplicitActuatorCfg(
            joint_names_expr=[".*_gripper_finger_joint[12]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=0.07,
            stiffness=25000.0,
            damping=1000.0,
            friction=0.2,
            armature=0.2,
        ),
        "r1_lite_torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_joint[1-3]"],
            stiffness=1050.0,
            damping=100.0,
            friction=0.0,
            armature=0.0,
            effort_limit_sim=87,
            velocity_limit_sim=124.6,
        ),
        "r1_lite_wheels": ImplicitActuatorCfg(
            joint_names_expr=["(steer|wheel)_motor_joint[1-3]"],
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=0.0,
            velocity_limit_sim=0.0,
        ),
    },
)

GALAXEA_R1_LITE_HEAD_CAMERA_CFG = CameraCfg(
    prim_path="/World/envs/env_.*/Robot/camera_head_left_link/head_cam",
    update_period=0.0,
    height=240,
    width=320,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=2.12,
        focus_distance=100.0,
        horizontal_aperture=6.055,
        clipping_range=(0.01, 100),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        convention="opengl",
    ),
)

GALAXEA_R1_LITE_HAND_CAMERA_CFG = CameraCfg(
    prim_path="/World/envs/env_.*/Robot/left_D405_link/left_hand_cam",
    update_period=0.0,
    height=240,
    width=320,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=2.12,
        focus_distance=100.0,
        horizontal_aperture=6.055,
        clipping_range=(0.01, 100),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        convention="opengl",
    ),
)
```

- [ ] **Step 3: Verify the file parses**

```bash
python -c "import ast; ast.parse(open('/scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/robots/galaxea_robots.py').read())"
```

Expected: silent exit 0.

- [ ] **Step 4: Commit**

```bash
git -C /scratch/SM/AAAI_challenge/gearboxAssembly add source/Galaxea_Lab_External/Galaxea_Lab_External/robots/galaxea_robots.py
git -C /scratch/SM/AAAI_challenge/gearboxAssembly commit -m "Add GALAXEA_R1_LITE_CFG and R1_Lite camera cfgs.

ArticulationCfg pointing at r1_lite.usd, mirroring the CHALLENGE PD
gains for arms/eefs/torso/grippers (the structure is comparable: 6-DOF
arms, 3-DOF torso, 2-finger prismatic grippers). The new wheels/steers
actuator group has stiffness=0 and effort=0 so the mobile base joints
stay put under fix_base=True. Initial joint_pos mirrors CHALLENGE.

Two CameraCfg constants for the R1_Lite link frames (camera_head_left
for stereo head, left_D405 / right_D405 for wrist cams) carrying the
same optical params as the existing R1 camera cfgs."
```

---

### Task 6: Fork `r1_lite_rule_policy.py`

**Files:**
- Create: `source/Galaxea_Lab_External/Galaxea_Lab_External/robots/r1_lite_rule_policy.py`

Mechanical fork of `galaxea_rule_policy.py`. Two transformations:
1. Class rename `GalaxeaRulePolicy` → `R1LiteRulePolicy`.
2. Joint-name literal rename `_gripper_axis1` → `_gripper_finger_joint1`.

Plus a docstring noting the geometry constants are still R1-tuned.

- [ ] **Step 1: Copy the source file**

```bash
cp /scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/robots/galaxea_rule_policy.py /scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/robots/r1_lite_rule_policy.py
```

- [ ] **Step 2: Replace the class name**

Use the Edit tool with `replace_all=true` on `r1_lite_rule_policy.py`:
- `old_string`: `GalaxeaRulePolicy`
- `new_string`: `R1LiteRulePolicy`

- [ ] **Step 3: Replace the gripper joint suffix**

Use the Edit tool with `replace_all=true` on `r1_lite_rule_policy.py`:
- `old_string`: `_gripper_axis1`
- `new_string`: `_gripper_finger_joint1`

- [ ] **Step 4: Grep for any remaining R1-only frame names that might be referenced as strings**

```bash
grep -nE "zed_link|left_realsense_link|right_realsense_link" /scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/robots/r1_lite_rule_policy.py
```

If any matches, perform analogous renames per file:
- `zed_link` → `camera_head_left_link`
- `left_realsense_link` → `left_D405_link`
- `right_realsense_link` → `right_D405_link`

If the grep returns nothing, no further edits are needed.

- [ ] **Step 5: Add a tune-me docstring at the top**

Use the Edit tool to insert a docstring AFTER the existing module-level header (typically the SPDX block) and BEFORE the first import. Insert this text on its own paragraph:

```python
"""Forked from galaxea_rule_policy.py for the R1_Lite robot.

Joint-name literals were mechanically renamed (_gripper_axis1 ->
_gripper_finger_joint1) so the env loads against R1_Lite's
articulation. The geometry constants below (target poses, approach
offsets, mounting plans) are still tuned for R1 dimensions; they will
produce wrong motions on R1_Lite until re-tuned. Use --no_action when
running rule_based_agent against R1_Lite to inspect the scene without
firing this policy.
"""
```

If the existing file already starts with a module docstring on line 6 (after the SPDX block), replace that docstring with the new one. Otherwise insert the docstring as a new top-level statement.

- [ ] **Step 6: Verify the file parses**

```bash
python -c "import ast; ast.parse(open('/scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/robots/r1_lite_rule_policy.py').read())"
```

Expected: silent exit 0.

- [ ] **Step 7: Commit**

```bash
git -C /scratch/SM/AAAI_challenge/gearboxAssembly add source/Galaxea_Lab_External/Galaxea_Lab_External/robots/r1_lite_rule_policy.py
git -C /scratch/SM/AAAI_challenge/gearboxAssembly commit -m "Add r1_lite_rule_policy.py (fork of galaxea_rule_policy).

Mechanical rename: GalaxeaRulePolicy -> R1LiteRulePolicy and gripper
joint literals _gripper_axis1 -> _gripper_finger_joint1 so the env
loads against R1_Lite. Geometry constants are still R1-tuned and
documented as tune-me in the module docstring."
```

---

### Task 7: Fork `r1_lite_recovery_rule_policy.py`

**Files:**
- Create: `source/Galaxea_Lab_External/Galaxea_Lab_External/robots/r1_lite_recovery_rule_policy.py`

Mechanical fork of `recovery_rule_policy.py`:
1. Class rename `RecoveryRulePolicy` → `R1LiteRecoveryRulePolicy`.
2. Joint-name literal rename `_gripper_axis1` → `_gripper_finger_joint1`.

- [ ] **Step 1: Copy the source file**

```bash
cp /scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/robots/recovery_rule_policy.py /scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/robots/r1_lite_recovery_rule_policy.py
```

- [ ] **Step 2: Replace the class name**

Use the Edit tool with `replace_all=true` on `r1_lite_recovery_rule_policy.py`:
- `old_string`: `RecoveryRulePolicy`
- `new_string`: `R1LiteRecoveryRulePolicy`

(This also renames any internal references; replace_all=true handles them all.)

- [ ] **Step 3: Replace the gripper joint suffix**

Use the Edit tool with `replace_all=true` on `r1_lite_recovery_rule_policy.py`:
- `old_string`: `_gripper_axis1`
- `new_string`: `_gripper_finger_joint1`

- [ ] **Step 4: Grep for any remaining R1-only frame names**

```bash
grep -nE "zed_link|left_realsense_link|right_realsense_link" /scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/robots/r1_lite_recovery_rule_policy.py
```

For each match, apply:
- `zed_link` → `camera_head_left_link`
- `left_realsense_link` → `left_D405_link`
- `right_realsense_link` → `right_D405_link`

- [ ] **Step 5: Add a tune-me docstring at the top**

Insert (or replace the existing module docstring) with:

```python
"""Forked from recovery_rule_policy.py for the R1_Lite robot.

Joint-name literals were mechanically renamed so the env loads against
R1_Lite's articulation. The geometry constants and recovery plans
below are still tuned for R1 and will not produce correct motions on
R1_Lite until re-tuned. Use --no_action when running rule_based_agent
against R1_Lite-recovery tasks to inspect the scene without firing
this policy.
"""
```

- [ ] **Step 6: Verify the file parses**

```bash
python -c "import ast; ast.parse(open('/scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/robots/r1_lite_recovery_rule_policy.py').read())"
```

Expected: silent exit 0.

- [ ] **Step 7: Commit**

```bash
git -C /scratch/SM/AAAI_challenge/gearboxAssembly add source/Galaxea_Lab_External/Galaxea_Lab_External/robots/r1_lite_recovery_rule_policy.py
git -C /scratch/SM/AAAI_challenge/gearboxAssembly commit -m "Add r1_lite_recovery_rule_policy.py (fork of recovery_rule_policy).

Mechanical rename: RecoveryRulePolicy -> R1LiteRecoveryRulePolicy and
gripper joint literals so the env loads against R1_Lite. Geometry
constants and recovery plans are still R1-tuned and documented as
tune-me in the module docstring."
```

---

### Task 8: Create the `RobotBundle` dataclass and the two bundles

**Files:**
- Create: `source/Galaxea_Lab_External/Galaxea_Lab_External/robots/robot_bundles.py`

- [ ] **Step 1: Write the file**

Create `/scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/robots/robot_bundles.py` with this exact content:

```python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-selection bundles.

A ``RobotBundle`` groups everything that varies per-robot (articulation
config, camera prim paths, joint dof-name strings, rule-policy classes)
into one frozen dataclass so each task env_cfg can read a single source
of truth. Switching between R1 and R1_Lite is one symbol edit:

    ACTIVE_ROBOT_BUNDLE = GALAXEA_R1_LITE_BUNDLE  # was GALAXEA_R1_BUNDLE

The two non-recovery env_cfgs read ``ACTIVE_ROBOT_BUNDLE.rule_policy_class``
for their rule-policy reference; the gearbox-recovery env_cfg reads
``ACTIVE_ROBOT_BUNDLE.recovery_rule_policy_class``.
"""

from dataclasses import dataclass

from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import CameraCfg

from .galaxea_robots import (
    GALAXEA_R1_CHALLENGE_CFG,
    GALAXEA_R1_LITE_CFG,
    GALAXEA_HEAD_CAMERA_CFG,
    GALAXEA_HAND_CAMERA_CFG,
    GALAXEA_R1_LITE_HEAD_CAMERA_CFG,
    GALAXEA_R1_LITE_HAND_CAMERA_CFG,
)
from .galaxea_rule_policy import GalaxeaRulePolicy
from .recovery_rule_policy import RecoveryRulePolicy
from .r1_lite_rule_policy import R1LiteRulePolicy
from .r1_lite_recovery_rule_policy import R1LiteRecoveryRulePolicy


@dataclass(frozen=True)
class RobotBundle:
    """Per-robot configuration aggregate.

    ``articulation_cfg``'s ``prim_path`` is overridden per env_cfg via
    ``.replace(prim_path=...)``, which returns a copy and does not
    mutate this frozen instance. The three camera cfgs are pre-set to
    the robot's frame paths.
    """

    name: str
    articulation_cfg: ArticulationCfg
    head_camera_cfg: CameraCfg
    left_hand_camera_cfg: CameraCfg
    right_hand_camera_cfg: CameraCfg
    left_arm_joint_pattern: str
    right_arm_joint_pattern: str
    left_gripper_dof_name: str
    right_gripper_dof_name: str
    torso_joint_pattern: str
    initial_torso_pos: tuple[float, float, float]
    rule_policy_class: type
    recovery_rule_policy_class: type


GALAXEA_R1_BUNDLE = RobotBundle(
    name="r1",
    articulation_cfg=GALAXEA_R1_CHALLENGE_CFG,
    head_camera_cfg=GALAXEA_HEAD_CAMERA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/zed_link/head_cam/head_cam",
    ),
    left_hand_camera_cfg=GALAXEA_HAND_CAMERA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/left_realsense_link/left_hand_cam/left_hand_cam",
    ),
    right_hand_camera_cfg=GALAXEA_HAND_CAMERA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/right_realsense_link/right_hand_cam/right_hand_cam",
    ),
    left_arm_joint_pattern="left_arm_joint.*",
    right_arm_joint_pattern="right_arm_joint.*",
    left_gripper_dof_name="left_gripper_axis1",
    right_gripper_dof_name="right_gripper_axis1",
    torso_joint_pattern="torso_joint[1-3]",
    initial_torso_pos=(0.5, -0.8, 0.5),
    rule_policy_class=GalaxeaRulePolicy,
    recovery_rule_policy_class=RecoveryRulePolicy,
)

GALAXEA_R1_LITE_BUNDLE = RobotBundle(
    name="r1_lite",
    articulation_cfg=GALAXEA_R1_LITE_CFG,
    head_camera_cfg=GALAXEA_R1_LITE_HEAD_CAMERA_CFG,
    left_hand_camera_cfg=GALAXEA_R1_LITE_HAND_CAMERA_CFG,
    right_hand_camera_cfg=GALAXEA_R1_LITE_HAND_CAMERA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/right_D405_link/right_hand_cam",
    ),
    left_arm_joint_pattern="left_arm_joint.*",
    right_arm_joint_pattern="right_arm_joint.*",
    left_gripper_dof_name="left_gripper_finger_joint1",
    right_gripper_dof_name="right_gripper_finger_joint1",
    torso_joint_pattern="torso_joint[1-3]",
    initial_torso_pos=(0.5, -0.8, 0.5),
    rule_policy_class=R1LiteRulePolicy,
    recovery_rule_policy_class=R1LiteRecoveryRulePolicy,
)


# The single switch. Edit this line to flip the active robot for all tasks.
ACTIVE_ROBOT_BUNDLE: RobotBundle = GALAXEA_R1_BUNDLE
```

- [ ] **Step 2: Verify the file parses**

```bash
python -c "import ast; ast.parse(open('/scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/robots/robot_bundles.py').read())"
```

Expected: silent exit 0.

- [ ] **Step 3: Commit**

```bash
git -C /scratch/SM/AAAI_challenge/gearboxAssembly add source/Galaxea_Lab_External/Galaxea_Lab_External/robots/robot_bundles.py
git -C /scratch/SM/AAAI_challenge/gearboxAssembly commit -m "Add RobotBundle dataclass and R1 / R1_Lite bundles.

The bundle aggregates ArticulationCfg + 3 CameraCfgs (with the right
frame paths) + joint dof-name strings + rule-policy class refs.
ACTIVE_ROBOT_BUNDLE defaults to GALAXEA_R1_BUNDLE so the existing
behavior is preserved bit-for-bit; switching to R1_Lite is a one-line
edit. frozen=True guards against accidental mutation."
```

---

### Task 9: Update `robots/__init__.py` to export new modules

**Files:**
- Modify: `source/Galaxea_Lab_External/Galaxea_Lab_External/robots/__init__.py`

The init currently re-exports from `galaxea_robots`, `galaxea_assets`, `gears_assets`, `galaxea_rule_policy`, `recovery_rule_policy`. We add three new lines for the bundles + two forks.

- [ ] **Step 1: Read the current file**

```bash
cat /scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/robots/__init__.py
```

Expected: 14 lines, last 5 are `from .galaxea_robots import *` etc.

- [ ] **Step 2: Append the new imports**

Use the Edit tool to append three lines AFTER `from .recovery_rule_policy import *`:

```python
from .r1_lite_rule_policy import *
from .r1_lite_recovery_rule_policy import *
from .robot_bundles import *
```

The order matters: `robot_bundles` last, because it imports from the others.

- [ ] **Step 3: Verify with an import smoke test (works without Isaac Sim)**

```bash
python -c "import ast; ast.parse(open('/scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/robots/__init__.py').read())"
```

Expected: silent exit 0. Note: actually importing `Galaxea_Lab_External.robots` requires `isaaclab` available, so we defer that to the env-active validation in Task 14.

- [ ] **Step 4: Commit**

```bash
git -C /scratch/SM/AAAI_challenge/gearboxAssembly add source/Galaxea_Lab_External/Galaxea_Lab_External/robots/__init__.py
git -C /scratch/SM/AAAI_challenge/gearboxAssembly commit -m "Re-export R1_Lite policies and robot bundles.

Add the R1_Lite rule-policy forks and the robot-bundles module to the
robots package's public surface so env_cfgs can import
ACTIVE_ROBOT_BUNDLE / RobotBundle directly."
```

---

### Task 10: Refactor `galaxea_lab_external` env_cfg + env

**Files:**
- Modify: `source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/galaxea_lab_external/galaxea_lab_external_env_cfg.py`
- Modify: `source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/galaxea_lab_external/galaxea_lab_external_env.py`

- [ ] **Step 1: Update the env_cfg imports**

In `galaxea_lab_external_env_cfg.py`, replace the import block (currently lines 27-36) using the Edit tool:

- `old_string`:
```python
from Galaxea_Lab_External.robots import (
    GALAXEA_R1_CHALLENGE_CFG,
    GALAXEA_HEAD_CAMERA_CFG,
    GALAXEA_HAND_CAMERA_CFG,
    TABLE_CFG,
    RING_GEAR_CFG,
    SUN_PLANETARY_GEAR_CFG,
    PLANETARY_CARRIER_CFG,
    PLANETARY_REDUCER_CFG,
)
```

- `new_string`:
```python
from Galaxea_Lab_External.robots import (
    ACTIVE_ROBOT_BUNDLE,
    RobotBundle,
    TABLE_CFG,
    RING_GEAR_CFG,
    SUN_PLANETARY_GEAR_CFG,
    PLANETARY_CARRIER_CFG,
    PLANETARY_REDUCER_CFG,
)
```

- [ ] **Step 2: Update the `robot_cfg` line in the env_cfg**

In the same file, replace:

- `old_string`:
```python
    # robot(s)
    robot_cfg: ArticulationCfg = GALAXEA_R1_CHALLENGE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
```

- `new_string`:
```python
    # robot(s)
    robot_bundle: RobotBundle = ACTIVE_ROBOT_BUNDLE
    robot_cfg: ArticulationCfg = ACTIVE_ROBOT_BUNDLE.articulation_cfg.replace(prim_path="/World/envs/env_.*/Robot")
    rule_policy_class: type = ACTIVE_ROBOT_BUNDLE.rule_policy_class
```

- [ ] **Step 3: Update the camera_cfg lines**

Replace:

- `old_string`:
```python
    # Camera
    head_camera_cfg: CameraCfg = GALAXEA_HEAD_CAMERA_CFG.replace(prim_path="/World/envs/env_.*/Robot/zed_link/head_cam/head_cam")
    left_hand_camera_cfg: CameraCfg = GALAXEA_HAND_CAMERA_CFG.replace(prim_path="/World/envs/env_.*/Robot/left_realsense_link/left_hand_cam/left_hand_cam")
    right_hand_camera_cfg: CameraCfg = GALAXEA_HAND_CAMERA_CFG.replace(prim_path="/World/envs/env_.*/Robot/right_realsense_link/right_hand_cam/right_hand_cam")
```

- `new_string`:
```python
    # Camera
    head_camera_cfg: CameraCfg = ACTIVE_ROBOT_BUNDLE.head_camera_cfg
    left_hand_camera_cfg: CameraCfg = ACTIVE_ROBOT_BUNDLE.left_hand_camera_cfg
    right_hand_camera_cfg: CameraCfg = ACTIVE_ROBOT_BUNDLE.right_hand_camera_cfg
```

- [ ] **Step 4: Update the joint-name fields**

Replace:

- `old_string`:
```python
    # custom parameters/scales
    # - controllable joint
    left_arm_joint_dof_name = "left_arm_joint.*"
    right_arm_joint_dof_name = "right_arm_joint.*"
    left_gripper_dof_name = "left_gripper_axis1"
    right_gripper_dof_name = "right_gripper_axis1"

    torso_joint_dof_name = "torso_joint[1-3]" # Since in current task, torso_joint4 will always be fixed at 0.0
```

- `new_string`:
```python
    # custom parameters/scales
    # - controllable joint (sourced from the active robot bundle)
    left_arm_joint_dof_name: str = ACTIVE_ROBOT_BUNDLE.left_arm_joint_pattern
    right_arm_joint_dof_name: str = ACTIVE_ROBOT_BUNDLE.right_arm_joint_pattern
    left_gripper_dof_name: str = ACTIVE_ROBOT_BUNDLE.left_gripper_dof_name
    right_gripper_dof_name: str = ACTIVE_ROBOT_BUNDLE.right_gripper_dof_name

    torso_joint_dof_name: str = ACTIVE_ROBOT_BUNDLE.torso_joint_pattern  # excludes torso_joint4 (R1-only)
```

- [ ] **Step 5: Update the env class to source the rule policy from cfg**

Open `galaxea_lab_external_env.py` and replace:

- `old_string`:
```python
from Galaxea_Lab_External.robots import GalaxeaRulePolicy
```

- `new_string`:
```python
# Rule-policy class is read from cfg (see ACTIVE_ROBOT_BUNDLE) so this env
# works for both R1 and R1_Lite without import-level coupling.
```

Then find the line that currently reads:
```python
        self.rule_policy = GalaxeaRulePolicy(sim_utils.SimulationContext.instance(), self.scene, self.obj_dict)
```

(Around line 575 per the most recent grep.) Replace it with:
```python
        self.rule_policy = self.cfg.rule_policy_class(sim_utils.SimulationContext.instance(), self.scene, self.obj_dict)
```

- [ ] **Step 6: Verify both files parse**

```bash
python -c "import ast; ast.parse(open('/scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/galaxea_lab_external/galaxea_lab_external_env_cfg.py').read())"
python -c "import ast; ast.parse(open('/scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/galaxea_lab_external/galaxea_lab_external_env.py').read())"
```

Expected: both silent exit 0.

- [ ] **Step 7: Commit**

```bash
git -C /scratch/SM/AAAI_challenge/gearboxAssembly add \
  source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/galaxea_lab_external/galaxea_lab_external_env_cfg.py \
  source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/galaxea_lab_external/galaxea_lab_external_env.py
git -C /scratch/SM/AAAI_challenge/gearboxAssembly commit -m "Wire galaxea_lab_external task to the active robot bundle.

env_cfg now reads ArticulationCfg, camera cfgs, joint dof-names, and
the rule-policy class from ACTIVE_ROBOT_BUNDLE. The env class drops
its hardcoded GalaxeaRulePolicy import and instantiates
self.cfg.rule_policy_class(...) instead. Default bundle is still R1,
so behavior is unchanged."
```

---

### Task 11: Refactor `galaxea_lab_agent` env_cfg + env

**Files:**
- Modify: `source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/galaxea_lab_agent/galaxea_lab_agent_env_cfg.py`
- Modify: `source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/galaxea_lab_agent/galaxea_lab_agent_env.py`

Same shape as Task 10. The agent env uses `GalaxeaRulePolicy` (line 511 of env.py per current grep).

- [ ] **Step 1: Read the current import block in env_cfg**

```bash
sed -n '25,40p' /scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/galaxea_lab_agent/galaxea_lab_agent_env_cfg.py
```

Note the exact set of imports — the agent env_cfg may pull a slightly different set of asset cfgs than the lab_external one. Preserve any non-robot imports (TABLE_CFG, gear cfgs, etc.).

- [ ] **Step 2: Update the env_cfg imports**

Use the Edit tool. Replace `GALAXEA_R1_CHALLENGE_CFG, GALAXEA_HEAD_CAMERA_CFG, GALAXEA_HAND_CAMERA_CFG` with `ACTIVE_ROBOT_BUNDLE, RobotBundle` in the import block. Keep all other imports.

- [ ] **Step 3: Update `robot_cfg` field**

Replace:

- `old_string`:
```python
    robot_cfg: ArticulationCfg = GALAXEA_R1_CHALLENGE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
```

- `new_string`:
```python
    robot_bundle: RobotBundle = ACTIVE_ROBOT_BUNDLE
    robot_cfg: ArticulationCfg = ACTIVE_ROBOT_BUNDLE.articulation_cfg.replace(prim_path="/World/envs/env_.*/Robot")
    rule_policy_class: type = ACTIVE_ROBOT_BUNDLE.rule_policy_class
```

- [ ] **Step 4: Update the camera_cfg lines**

Replace:

- `old_string`:
```python
    head_camera_cfg: CameraCfg = GALAXEA_HEAD_CAMERA_CFG.replace(prim_path="/World/envs/env_.*/Robot/zed_link/head_cam/head_cam")
    left_hand_camera_cfg: CameraCfg = GALAXEA_HAND_CAMERA_CFG.replace(prim_path="/World/envs/env_.*/Robot/left_realsense_link/left_hand_cam/left_hand_cam")
    right_hand_camera_cfg: CameraCfg = GALAXEA_HAND_CAMERA_CFG.replace(prim_path="/World/envs/env_.*/Robot/right_realsense_link/right_hand_cam/right_hand_cam")
```

- `new_string`:
```python
    head_camera_cfg: CameraCfg = ACTIVE_ROBOT_BUNDLE.head_camera_cfg
    left_hand_camera_cfg: CameraCfg = ACTIVE_ROBOT_BUNDLE.left_hand_camera_cfg
    right_hand_camera_cfg: CameraCfg = ACTIVE_ROBOT_BUNDLE.right_hand_camera_cfg
```

- [ ] **Step 5: Update the joint-name fields**

Replace:

- `old_string`:
```python
    left_arm_joint_dof_name = "left_arm_joint.*"
    right_arm_joint_dof_name = "right_arm_joint.*"
    left_gripper_dof_name = "left_gripper_axis1"
    right_gripper_dof_name = "right_gripper_axis1"

    torso_joint_dof_name = "torso_joint[1-3]" # Since in current task, torso_joint4 will always be fixed at 0.0
```

- `new_string`:
```python
    left_arm_joint_dof_name: str = ACTIVE_ROBOT_BUNDLE.left_arm_joint_pattern
    right_arm_joint_dof_name: str = ACTIVE_ROBOT_BUNDLE.right_arm_joint_pattern
    left_gripper_dof_name: str = ACTIVE_ROBOT_BUNDLE.left_gripper_dof_name
    right_gripper_dof_name: str = ACTIVE_ROBOT_BUNDLE.right_gripper_dof_name

    torso_joint_dof_name: str = ACTIVE_ROBOT_BUNDLE.torso_joint_pattern  # excludes torso_joint4 (R1-only)
```

- [ ] **Step 6: Update the env class to source rule_policy from cfg**

In `galaxea_lab_agent_env.py`, replace:

- `old_string`:
```python
from Galaxea_Lab_External.robots import GalaxeaRulePolicy
```

- `new_string`:
```python
# Rule-policy class is read from cfg (see ACTIVE_ROBOT_BUNDLE).
```

Then replace the instantiation (around line 511):

- `old_string`:
```python
        self.rule_policy = GalaxeaRulePolicy(sim_utils.SimulationContext.instance(), self.scene, self.obj_dict)
```

- `new_string`:
```python
        self.rule_policy = self.cfg.rule_policy_class(sim_utils.SimulationContext.instance(), self.scene, self.obj_dict)
```

- [ ] **Step 7: Verify both files parse**

```bash
python -c "import ast; ast.parse(open('/scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/galaxea_lab_agent/galaxea_lab_agent_env_cfg.py').read())"
python -c "import ast; ast.parse(open('/scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/galaxea_lab_agent/galaxea_lab_agent_env.py').read())"
```

Expected: both silent exit 0.

- [ ] **Step 8: Commit**

```bash
git -C /scratch/SM/AAAI_challenge/gearboxAssembly add \
  source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/galaxea_lab_agent/galaxea_lab_agent_env_cfg.py \
  source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/galaxea_lab_agent/galaxea_lab_agent_env.py
git -C /scratch/SM/AAAI_challenge/gearboxAssembly commit -m "Wire galaxea_lab_agent task to the active robot bundle.

Same refactor as galaxea_lab_external: env_cfg pulls ArticulationCfg,
camera cfgs, joint dof-names, and rule_policy_class from
ACTIVE_ROBOT_BUNDLE; the env class instantiates
self.cfg.rule_policy_class(...) instead of importing GalaxeaRulePolicy."
```

---

### Task 12: Refactor `gearbox_recovery` env_cfg + env

**Files:**
- Modify: `source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/gearbox_recovery/gearbox_recovery_env_cfg.py`
- Modify: `source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/gearbox_recovery/gearbox_recovery_env.py`

Almost identical to Task 11, with one critical difference: this task uses `RecoveryRulePolicy` (not `GalaxeaRulePolicy`) and instantiates it with FOUR args (the extra one is `cfg.initial_assembly_state`). The env_cfg therefore reads `recovery_rule_policy_class` from the bundle.

- [ ] **Step 1: Update the env_cfg imports**

Replace `GALAXEA_R1_CHALLENGE_CFG, GALAXEA_HEAD_CAMERA_CFG, GALAXEA_HAND_CAMERA_CFG` with `ACTIVE_ROBOT_BUNDLE, RobotBundle` in the import block.

- [ ] **Step 2: Update `robot_cfg` field — note the recovery class**

Replace:

- `old_string`:
```python
    robot_cfg: ArticulationCfg = GALAXEA_R1_CHALLENGE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
```

- `new_string`:
```python
    robot_bundle: RobotBundle = ACTIVE_ROBOT_BUNDLE
    robot_cfg: ArticulationCfg = ACTIVE_ROBOT_BUNDLE.articulation_cfg.replace(prim_path="/World/envs/env_.*/Robot")
    rule_policy_class: type = ACTIVE_ROBOT_BUNDLE.recovery_rule_policy_class
```

(Note: `rule_policy_class` here is bound to the bundle's `recovery_rule_policy_class`, not `rule_policy_class` — the env class refers to it as `rule_policy_class` but it points at the recovery policy.)

- [ ] **Step 3: Update the camera_cfg lines**

Same edit as Task 11 step 4.

- [ ] **Step 4: Update the joint-name fields**

Same edit as Task 11 step 5.

- [ ] **Step 5: Update the env class to source rule_policy from cfg (with the 4th arg)**

In `gearbox_recovery_env.py`, replace:

- `old_string`:
```python
from Galaxea_Lab_External.robots import RecoveryRulePolicy
```

- `new_string`:
```python
# Rule-policy class is read from cfg (see ACTIVE_ROBOT_BUNDLE.recovery_rule_policy_class).
```

Then replace the instantiation (around line 753):

- `old_string`:
```python
        self.rule_policy = RecoveryRulePolicy(sim_utils.SimulationContext.instance(), self.scene, self.obj_dict, self.cfg.initial_assembly_state)
```

- `new_string`:
```python
        self.rule_policy = self.cfg.rule_policy_class(sim_utils.SimulationContext.instance(), self.scene, self.obj_dict, self.cfg.initial_assembly_state)
```

- [ ] **Step 6: Verify both files parse**

```bash
python -c "import ast; ast.parse(open('/scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/gearbox_recovery/gearbox_recovery_env_cfg.py').read())"
python -c "import ast; ast.parse(open('/scratch/SM/AAAI_challenge/gearboxAssembly/source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/gearbox_recovery/gearbox_recovery_env.py').read())"
```

Expected: both silent exit 0.

- [ ] **Step 7: Commit**

```bash
git -C /scratch/SM/AAAI_challenge/gearboxAssembly add \
  source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/gearbox_recovery/gearbox_recovery_env_cfg.py \
  source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/gearbox_recovery/gearbox_recovery_env.py
git -C /scratch/SM/AAAI_challenge/gearboxAssembly commit -m "Wire gearbox_recovery task to the active robot bundle.

env_cfg's rule_policy_class field is bound to the bundle's
recovery_rule_policy_class (not rule_policy_class), since this task
uses the recovery variant which takes an extra initial_assembly_state
constructor arg. The env class still passes that 4th arg through."
```

---

### Task 13: Document the robot switch in README.md

**Files:**
- Modify: `README.md`

Add a short subsection explaining the bundle switch and the conversion script.

- [ ] **Step 1: Find the right insertion point**

```bash
grep -n "^### \|^## " /scratch/SM/AAAI_challenge/gearboxAssembly/README.md
```

Locate the last sub-section under "Run the rule-based agent" (the one ending with `--no_action` flag note for Inclined task). The new section goes after that block and before the next `## ` top-level heading (probably `## Troubleshooting` or similar).

- [ ] **Step 2: Insert the new subsection**

Use the Edit tool to insert the following block. Choose `old_string` as the line that currently appears immediately AFTER where the new section should be inserted (e.g., the next `### Set up IDE (Optional)` or `## Troubleshooting` heading), and prepend the new content + a blank line + the original line as `new_string`. Or insert before a stable anchor like `### Set up IDE (Optional)`.

New content to insert:

```markdown
### Switching between R1 and R1_Lite

The repo ships with two robot configs:
- `GALAXEA_R1_BUNDLE` (default) — the original Galaxea R1 (`r1_DVT_*.usd`).
- `GALAXEA_R1_LITE_BUNDLE` — the new R1_Lite (mobile-base variant; `r1_lite.usd`, base welded for tabletop tasks).

To switch, edit one line in `source/Galaxea_Lab_External/Galaxea_Lab_External/robots/robot_bundles.py`:

```python
ACTIVE_ROBOT_BUNDLE: RobotBundle = GALAXEA_R1_LITE_BUNDLE  # was GALAXEA_R1_BUNDLE
```

All three task envs (`Template-Galaxea-Lab-External-Direct-v0`, `Template-Galaxea-Lab-Agent-Direct-v0`, and the `Gearbox-*` recovery tasks) read from `ACTIVE_ROBOT_BUNDLE`, so no other edits are needed.

**Caveat — rule-based agent on R1_Lite.** `r1_lite_rule_policy.py` and `r1_lite_recovery_rule_policy.py` are forks of the R1 policies with mechanical joint-name renames so the env loads, but their pose/offset constants are still tuned for R1 dimensions. Running `rule_based_agent.py` against R1_Lite without `--no_action` will produce wrong motions. Use `--no_action` to inspect the scene visually until the constants are re-tuned.

### Re-running the R1_Lite URDF→USD conversion

If the vendor URDF under `source/Galaxea_Lab_External/assets/Robots/R1_Lite/urdf/` changes, regenerate the USD:

```bash
conda deactivate 2>/dev/null || true
source scripts/env.sh
python scripts/convert_r1_lite_urdf.py
```

The script rewrites `package://mobiman/...` mesh refs to relative paths in a temp URDF copy (the committed URDF is never modified) and runs `isaaclab.sim.converters.UrdfConverter` with `fix_base=True`. Output: `assets/Robots/R1_Lite/r1_lite.usd`.

```

- [ ] **Step 3: Commit**

```bash
git -C /scratch/SM/AAAI_challenge/gearboxAssembly add README.md
git -C /scratch/SM/AAAI_challenge/gearboxAssembly commit -m "README: document robot bundle switch and R1_Lite conversion script.

One-liner switch via ACTIVE_ROBOT_BUNDLE plus the documented caveat
that R1_Lite rule policies are R1-tuned and need --no_action for
clean inspection. Also documents the convert_r1_lite_urdf.py path."
```

---

### Task 14: Validation

This task is a manual smoke test, run with Isaac Sim active. No commit; if any step fails, fix the underlying code in the relevant earlier task and re-run.

- [ ] **Step 1: Activate the env**

```bash
cd /scratch/SM/AAAI_challenge/gearboxAssembly
conda deactivate 2>/dev/null || true
source scripts/env.sh
```

- [ ] **Step 2: Confirm imports work**

```bash
python -c "from Galaxea_Lab_External.robots import ACTIVE_ROBOT_BUNDLE, GALAXEA_R1_BUNDLE, GALAXEA_R1_LITE_BUNDLE, RobotBundle, R1LiteRulePolicy, R1LiteRecoveryRulePolicy; print(ACTIVE_ROBOT_BUNDLE.name)"
```

Expected: prints `r1` and exits 0.

- [ ] **Step 3: Confirm task registration still works**

```bash
python scripts/list_envs.py
```

Expected: lists `Template-Galaxea-Lab-External-Direct-v0`, `Template-Galaxea-Lab-Agent-Direct-v0`, and the three `Gearbox-*` recovery tasks (Lackfourth, Misplacedfourth, Inclinedfourth) without errors. Isaac Sim startup messages are noise.

- [ ] **Step 4: Default-robot regression — R1 still runs**

```bash
python scripts/rule_based_agent.py --task=Template-Galaxea-Lab-External-Direct-v0 --enable_cameras
```

Expected: scene loads, R1 articulates, rule policy executes the gearbox-assembly motions. This must behave identically to pre-refactor. Kill with Ctrl+C after a few seconds of confirmed correct behavior.

- [ ] **Step 5: R1_Lite swap smoke test**

Edit `source/Galaxea_Lab_External/Galaxea_Lab_External/robots/robot_bundles.py` and change the last line to:

```python
ACTIVE_ROBOT_BUNDLE: RobotBundle = GALAXEA_R1_LITE_BUNDLE
```

(Do NOT commit this change — it's a transient swap for the smoke test.)

Run:

```bash
python scripts/rule_based_agent.py --task=Template-Galaxea-Lab-External-Direct-v0 --enable_cameras --no_action
```

Expected:
- Scene loads without errors.
- R1_Lite USD spawns (visually confirm: stereo head cameras up top, mobile-base wheels visible).
- The three cameras (head, left hand, right hand) attach without "prim path not found" warnings.
- The `--no_action` flag prevents the (R1-tuned) rule policy from firing, so the robot just stands.

If the env crashes on a missing joint or frame, capture the error and fix the relevant bundle field or fork file in the appropriate earlier task. The most likely failure modes:
- Camera prim path mismatch → fix `GALAXEA_R1_LITE_HEAD_CAMERA_CFG` / `GALAXEA_R1_LITE_HAND_CAMERA_CFG` prim_path defaults in `galaxea_robots.py`.
- Missing joint name → fix the joint_pos dict in `GALAXEA_R1_LITE_CFG` or the `_gripper_finger_joint1` literal in the bundle.
- Rule-policy import error → fix the fork file in Task 6 or 7.

Kill with Ctrl+C once the scene is stable.

- [ ] **Step 6: Revert the bundle to R1**

Edit `robot_bundles.py` back to `ACTIVE_ROBOT_BUNDLE: RobotBundle = GALAXEA_R1_BUNDLE`. Verify with `git diff` that the file is byte-identical to the committed version.

- [ ] **Step 7: Confirm the working tree is clean**

```bash
git -C /scratch/SM/AAAI_challenge/gearboxAssembly status --short
```

Expected: empty output.

---

## Self-review checklist (run after writing the plan)

The author of this plan ran the spec-coverage / placeholder / type-consistency check before handoff:

- **Spec section 1 (Goal):** Covered by the overall plan; default behavior preserved by Task 14 step 4.
- **Spec section 2 (Source material):** Tasks 2 and 4 stage them.
- **Spec section 3 (Decisions):** Task 1 (LFS), Task 2 (raw assets), Task 3+4 (script + USD), Tasks 5–9 (bundle), Tasks 10–12 (env switch), Task 13 (README).
- **Spec section 4 (Architecture):** Task 8 implements the dataclass and bundles; Task 9 wires the public surface.
- **Spec section 5 (Conversion script):** Task 3 writes it; Task 4 runs it.
- **Spec section 6 (`GALAXEA_R1_LITE_CFG`):** Task 5.
- **Spec section 7 (Forked policies):** Tasks 6 and 7.
- **Spec section 8 (Env-cfg / env-class refactor):** Tasks 10, 11, 12.
- **Spec section 9 (`.gitattributes`):** Task 1.
- **Spec section 10 (README):** Task 13.
- **Spec section 11 (File touch summary):** Every file in the spec's list appears in some task's "Files" header.
- **Spec section 12 (Validation):** Task 14.
- **Spec section 13 (Known gaps):** Tasks 6, 7, and 13 carry the tune-me docstrings/notes.

Type/name consistency: `ACTIVE_ROBOT_BUNDLE`, `RobotBundle`, `rule_policy_class`, `recovery_rule_policy_class`, `GalaxeaRulePolicy`, `RecoveryRulePolicy`, `R1LiteRulePolicy`, `R1LiteRecoveryRulePolicy`, `GALAXEA_R1_LITE_CFG`, `GALAXEA_R1_LITE_HEAD_CAMERA_CFG`, `GALAXEA_R1_LITE_HAND_CAMERA_CFG`, `_gripper_finger_joint1` — every name is used identically across the tasks where it appears.

No placeholders ("TODO", "TBD", etc.) in the task bodies. The deferred grep in Tasks 6 and 7 ("if any matches, apply analogous renames") is a deliberate runtime safety net, not a plan gap — the rename rules are spelled out in advance.
