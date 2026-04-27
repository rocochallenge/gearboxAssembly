# R1_Lite Robot Integration — Design

**Date:** 2026-04-27
**Status:** Approved (user sign-off section by section)
**Branch:** `dev`

## 1. Goal

Convert the new Galaxea R1_Lite robot URDF (under `source/Galaxea_Lab_External/assets/Robots/R1_Lite/`) into a USD usable by Isaac Sim 5.1 / Isaac Lab 2.3.0, and integrate it into the existing gearbox-assembly task envs so the user can switch between the existing R1 and R1_Lite by editing a single symbol. R1 must remain the default and behave identically to today.

## 2. Source material

- **URDF:** `source/Galaxea_Lab_External/assets/Robots/R1_Lite/urdf/mmp_revB_invconfig_upright_a1x.urdf` — 34 links, 33 joints (18 revolute, 4 prismatic, 3 continuous, 8 fixed). Auto-exported from SolidWorks. Mesh refs use `package://mobiman/urdf/R1_Lite/meshes/*.STL`.
- **Meshes:** 33 STL files under `assets/Robots/R1_Lite/meshes/` (~18 MB total).
- **Existing R1 USDs:** 5 LFS-tracked files in `assets/Robots/Galaxea/r1_*.usd`. Project conventions: USDs in LFS via `*.usd filter=lfs` in repo-root `.gitattributes`.

## 3. Decisions (recorded during brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| End state | Coexist as switchable robots | User asked for "easily change the robot in the simulator". |
| Mobile base | `fix_base=True`; wheels remain articulated | Tabletop tasks; mirrors R1's effectively-stationary behavior. |
| Asset storage | Convert once via project script + commit USD via LFS | Consistent with how existing R1 USDs ship; deterministic. |
| USD location | `assets/Robots/R1_Lite/r1_lite.usd` | Self-contained next to URDF + meshes. |
| Robot switch | Single `ACTIVE_ROBOT_BUNDLE` pointer | One symbol to flip; no CLI/env-var plumbing across many scripts. |
| Joint naming | Keep R1_Lite names; bundle carries dof-name strings | Faithful to upstream Galaxea release; no URDF rewrites. |
| Rule policy | Fork to `r1_lite_rule_policy.py` with mechanical name substitutions | Preserves R1 path; gives R1_Lite a clean home for future tuning. |
| Mesh paths | Rewrite `package://mobiman/...` → `../meshes/...` at conversion time only | Self-contained; original URDF stays unchanged. |
| Actuator gains | Reuse R1 CHALLENGE values; wheels/steers zero-locked | Sane starting point that mirrors R1's tuning. |
| Fork content | Substitute joint names so it loads | Avoid crash on missing `_gripper_axis1`; geometry tune-me later. |
| STL LFS | Add `*.stl` / `*.STL` to `.gitattributes` | Consistent with existing USD/OBJ LFS handling. |

## 4. Architecture

### 4.1 New code modules

```
source/Galaxea_Lab_External/Galaxea_Lab_External/robots/
├── robot_bundles.py                  # NEW — RobotBundle dataclass, R1_BUNDLE, R1_LITE_BUNDLE, ACTIVE_ROBOT_BUNDLE
├── r1_lite_rule_policy.py            # NEW — fork of galaxea_rule_policy.py
├── r1_lite_recovery_rule_policy.py   # NEW — fork of recovery_rule_policy.py
├── galaxea_robots.py                 # EDITED — add GALAXEA_R1_LITE_CFG + R1_Lite camera cfgs
├── __init__.py                       # EDITED — export new modules
├── galaxea_rule_policy.py            # UNCHANGED (R1 path)
├── recovery_rule_policy.py           # UNCHANGED (R1 path)
├── galaxea_assets.py                 # UNCHANGED
└── gears_assets.py                   # UNCHANGED
```

### 4.2 The `RobotBundle` dataclass

```python
@dataclass(frozen=True)
class RobotBundle:
    name: str
    articulation_cfg: ArticulationCfg
    head_camera_cfg: CameraCfg
    left_hand_camera_cfg: CameraCfg
    right_hand_camera_cfg: CameraCfg
    left_arm_joint_pattern: str        # regex, e.g. "left_arm_joint.*"
    right_arm_joint_pattern: str
    left_gripper_dof_name: str         # exact joint name
    right_gripper_dof_name: str
    torso_joint_pattern: str           # regex, e.g. "torso_joint[1-3]"
    initial_torso_pos: tuple[float, float, float]
    rule_policy_class: type
    recovery_rule_policy_class: type
```

`frozen=True` prevents accidental mutation of the shared bundle. `articulation_cfg.replace(prim_path=...)` returns a copy and is safe.

### 4.3 The two bundles + active pointer

`GALAXEA_R1_BUNDLE` wraps the existing R1 (joints `*_gripper_axis1`, frames `zed_link/head_cam`, `left_realsense_link/left_hand_cam`, `right_realsense_link/right_hand_cam`, classes `GalaxeaRulePolicy` / `RecoveryRulePolicy`).

`GALAXEA_R1_LITE_BUNDLE` wraps R1_Lite (joints `*_gripper_finger_joint1`, frames `camera_head_left_link/head_cam`, `left_D405_link/left_hand_cam`, `right_D405_link/right_hand_cam`, classes `R1LiteRulePolicy` / `R1LiteRecoveryRulePolicy`).

Note: the recovery class is named `RecoveryRulePolicy` in `recovery_rule_policy.py` (no `Galaxea` prefix). The `gearbox_recovery_env_cfg.py` reads its `rule_policy_class` field from `ACTIVE_ROBOT_BUNDLE.recovery_rule_policy_class` (not `.rule_policy_class`); the other two env_cfgs read `.rule_policy_class`. The env classes call `self.cfg.rule_policy_class(...)` with the right constructor arguments — gearbox_recovery passes an extra `initial_assembly_state` arg.

`ACTIVE_ROBOT_BUNDLE: RobotBundle = GALAXEA_R1_BUNDLE` — the single pointer that env_cfgs read. Default is R1 to preserve existing behavior.

## 5. URDF→USD conversion script

`scripts/convert_r1_lite_urdf.py`:

1. Read URDF text from `source/Galaxea_Lab_External/assets/Robots/R1_Lite/urdf/mmp_revB_invconfig_upright_a1x.urdf`.
2. Replace `package://mobiman/urdf/R1_Lite/meshes/` → `../meshes/` in memory.
3. Write the patched URDF to a temp file inside `assets/Robots/R1_Lite/urdf/` (so `../meshes/` resolves).
4. Launch Isaac Sim via `AppLauncher` (mirrors `IsaacLab/scripts/tools/convert_urdf.py` lifecycle).
5. Run `UrdfConverter` with:
   - `asset_path` = temp URDF
   - `usd_dir` = `assets/Robots/R1_Lite/`, `usd_file_name` = `r1_lite.usd`
   - `fix_base=True`, `merge_fixed_joints=False`
   - `force_usd_conversion=True`, `make_instanceable=True`
   - `self_collision=False`, `collider_type="convex_hull"`
   - `joint_drive=JointDriveCfg(target_type="position", drive_type="force", gains=PDGainsCfg(stiffness=1050.0, damping=100.0))`
6. Delete the temp URDF.
7. Print the USD path; exit.

The committed URDF is never modified. Re-running is idempotent.

## 6. `GALAXEA_R1_LITE_CFG` (added to `galaxea_robots.py`)

`ArticulationCfg` mirroring `GALAXEA_R1_CHALLENGE_CFG`'s structure with R1_Lite-specific fields:

- `usd_path` → `r1_lite.usd`
- `init_state.joint_pos` → keys for arms (`*_arm_joint1..6`), grippers (`*_gripper_finger_joint1`), torso (`torso_joint1..3`), all zero or task-default
- Five actuator groups:
  - `r1_lite_arms` — `.*_arm_joint[1-5]`, stiffness 1050, damping 100
  - `r1_lite_eefs` — `.*_arm_joint6`, stiffness 1050, damping 100
  - `r1_lite_grippers` — `.*_gripper_finger_joint[12]`, stiffness 25000, damping 1000
  - `r1_lite_torso` — `torso_joint[1-3]`, stiffness 1050, damping 100
  - `r1_lite_wheels` — `(steer|wheel)_motor_joint[1-3]`, stiffness 0, effort 0 (locks them with `fix_base=True`)

Plus two new `CameraCfg` constants `GALAXEA_R1_LITE_HEAD_CAMERA_CFG` and `GALAXEA_R1_LITE_HAND_CAMERA_CFG` carrying the same optical settings as the R1 originals but with R1_Lite default prim paths.

## 7. Forked rule policies

`r1_lite_rule_policy.py`:
- `cp galaxea_rule_policy.py r1_lite_rule_policy.py`
- Rename class `GalaxeaRulePolicy` → `R1LiteRulePolicy`
- Replace literal `_gripper_axis1` → `_gripper_finger_joint1`
- Replace any R1-specific link-prim lookups (`zed_link`, `left_realsense_link`, `right_realsense_link`) with R1_Lite equivalents
- Add module-level docstring noting geometry constants are still R1-tuned and must be re-tuned for R1_Lite

`r1_lite_recovery_rule_policy.py` — same treatment for `recovery_rule_policy.py`.

A fresh grep across both source files at implementation time identifies any other R1-specific string literals to rename.

## 8. Env-cfg / env-class refactor

Three task envs touched the same way:
- `tasks/direct/galaxea_lab_external/`
- `tasks/direct/galaxea_lab_agent/`
- `tasks/direct/gearbox_recovery/`

In each `*_env_cfg.py`:
- Replace imports of `GALAXEA_R1_CHALLENGE_CFG`, `GALAXEA_HEAD_CAMERA_CFG`, `GALAXEA_HAND_CAMERA_CFG` with `ACTIVE_ROBOT_BUNDLE`, `RobotBundle`.
- `robot_bundle: RobotBundle = ACTIVE_ROBOT_BUNDLE`
- `robot_cfg: ArticulationCfg = ACTIVE_ROBOT_BUNDLE.articulation_cfg.replace(prim_path="/World/envs/env_.*/Robot")`
- `head_camera_cfg / left_hand_camera_cfg / right_hand_camera_cfg` ← bundle's camera cfgs
- `left_arm_joint_dof_name / right_arm_joint_dof_name / left_gripper_dof_name / right_gripper_dof_name / torso_joint_dof_name` ← bundle's pattern strings
- `rule_policy_class: type = ACTIVE_ROBOT_BUNDLE.rule_policy_class` (and `recovery_rule_policy_class` where applicable)

In each `*_env.py`:
- Drop the top-level `from Galaxea_Lab_External.robots import GalaxeaRulePolicy` literal import
- Replace `GalaxeaRulePolicy(...)` instantiation with `self.cfg.rule_policy_class(...)`
- (For `gearbox_recovery_env.py`, verify whether it uses `GalaxeaRulePolicy` or `GalaxeaRecoveryRulePolicy` and route accordingly via `cfg.rule_policy_class` or `cfg.recovery_rule_policy_class`.)

Switching robots is now: edit one line in `robot_bundles.py`.

## 9. `.gitattributes` change

Append two lines to repo-root `.gitattributes`:

```
*.stl filter=lfs diff=lfs merge=lfs -text
*.STL filter=lfs diff=lfs merge=lfs -text
```

Order of operations (critical):
1. Edit `.gitattributes`, commit.
2. `git add` the R1_Lite STL/URDF/config tree → STLs go to LFS.
3. Generate the USD and `git add` it → matches existing `*.usd filter=lfs` rule.

## 10. README addition

A subsection in `README.md` ("Switching the active robot" or similar) documenting:
- How to swap robots (edit `ACTIVE_ROBOT_BUNDLE` in `robot_bundles.py`).
- How to re-run the conversion script if the URDF source changes.
- That R1_Lite is supported as an articulation/cameras swap; rule policies are forked but geometry is still R1-tuned (tune-me).

## 11. File touch summary

**Created:**
- `scripts/convert_r1_lite_urdf.py`
- `source/Galaxea_Lab_External/Galaxea_Lab_External/robots/robot_bundles.py`
- `source/Galaxea_Lab_External/Galaxea_Lab_External/robots/r1_lite_rule_policy.py`
- `source/Galaxea_Lab_External/Galaxea_Lab_External/robots/r1_lite_recovery_rule_policy.py`
- `source/Galaxea_Lab_External/assets/Robots/R1_Lite/r1_lite.usd` (generated, LFS-tracked)

**Modified:**
- `.gitattributes`
- `source/Galaxea_Lab_External/Galaxea_Lab_External/robots/__init__.py`
- `source/Galaxea_Lab_External/Galaxea_Lab_External/robots/galaxea_robots.py`
- `source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/galaxea_lab_external/galaxea_lab_external_env_cfg.py`
- `source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/galaxea_lab_external/galaxea_lab_external_env.py`
- `source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/galaxea_lab_agent/galaxea_lab_agent_env_cfg.py`
- `source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/galaxea_lab_agent/galaxea_lab_agent_env.py`
- `source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/gearbox_recovery/gearbox_recovery_env_cfg.py`
- `source/Galaxea_Lab_External/Galaxea_Lab_External/tasks/direct/gearbox_recovery/gearbox_recovery_env.py`
- `README.md`

**Untouched:**
- `assets/Robots/R1_Lite/urdf/*.urdf` (script reads, never writes)
- `assets/Robots/R1_Lite/meshes/*.STL`
- `galaxea_rule_policy.py`, `recovery_rule_policy.py` (R1 stays exactly as-is)
- All gear/asset cfgs

## 12. Validation

Manual checks at end of implementation:

1. **Conversion smoke test** — `python scripts/convert_r1_lite_urdf.py` exits cleanly, produces `r1_lite.usd` (>1 MB, LFS-tracked verified by `git check-attr lfs`).
2. **Default-robot regression** — with `ACTIVE_ROBOT_BUNDLE = GALAXEA_R1_BUNDLE`, `python scripts/rule_based_agent.py --task=Template-Galaxea-Lab-External-Direct-v0 --enable_cameras` runs identically to pre-refactor.
3. **Robot-swap smoke test** — flip `ACTIVE_ROBOT_BUNDLE` to `GALAXEA_R1_LITE_BUNDLE`, re-run the same task with `--no_action`. Env loads without errors, R1_Lite USD spawns, cameras attach to R1_Lite frames.

## 13. Known gaps (out of scope)

- `r1_lite_rule_policy.py` and `r1_lite_recovery_rule_policy.py` ship with R1-tuned geometry constants (poses, offsets, approach vectors). Running rule-based agent against R1_Lite without `--no_action` will produce wrong motions until those constants are re-tuned. Documented in fork docstrings and README.
- The R1_Lite mobile base is welded via `fix_base=True`; navigation extensions (e.g., driving the wheels) are out of scope.
- ACT/VLA agent compatibility with R1_Lite is not addressed — the VLA path is independent of the bundle and may need its own swap layer in a follow-up.
