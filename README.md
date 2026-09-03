# RoCo Challenge: Robotic Collaborative Assembling - HMI Workshop @ AAAI 2026
![RoCo Challenge Poster](docs/images/poster.png)
Code Repository for the [RoCo Challenge@AAAI 2026](https://rocochallenge.github.io/RoCo2026/doc.html)

The Gearbox Assembly Assistance Challenge evaluates bimanual robotic systems in collaborative gearbox assembly within manufacturing environments. It targets scenarios where robots work seamlessly with human operators.

#### ⭐🎄 News 24 Dec 2025:
Merry Christmas! We added 3 new environment definitions for task 2 (resume from partial state) and task 3 (error detection and recovery), serving as environment settings for the final examination, and also to facilitate your self-evaluation of your own models. Please check out the quickstart commands [here](#run-the-rule-based-agent).

## Overview

In this project, we setup a Isaac Lab environment for the Galaxea R1 gearbox assembly task. 

## Installation
### ⚙️ Model File Management Instructions

---

**⚠️ IMPORTANT: This Repository Contains Large Files (LFS)!**

The **gearbox part models (`.usd` files)** within this repository are managed using **Git Large File Storage (LFS)**. If you clone the repository without using LFS, these files will not be properly checked out; you will only receive text pointers instead of the actual model data.

**To correctly retrieve the model files after cloning or pulling the repository content, you must follow these steps:**

1.  ### **Install and Initialize Git LFS**

    Ensure Git LFS is installed on your system. You can set it up by running the following command:

    ```bash
    git lfs install
    ```

    *This command only needs to be run **once** on your machine.*

2.  ### **Fetch the Model Files**

    If you are cloning the repository for the first time, or if you cloned it before running `git lfs install`, run the following commands to check out all LFS-managed files:

    ```bash
    git lfs pull
    ```

    *(Alternatively, you can use `git lfs fetch` followed by `git lfs checkout`)*

    **Please ensure these steps are completed before attempting to compile or run any code that depends on the `.usd` model files.**

**Due to LFS bandwidth limits, the LFS files may fail to download. As an alternative, download the LFS files by clicking [this link](https://drive.google.com/file/d/1L7u89xxiHGkd72CzvZln3P5uPPqMu7b7/view?usp=drive_link), extract it, and place its contents into `gearboxAssembly/source/Galaxea_Lab_External/assets`.**

---
### Installation Steps

This project uses [uv](https://docs.astral.sh/uv/) to manage a project-local `.venv` that pulls in Isaac Lab 2.3.0 (as a git submodule) and its dependencies. Isaac Sim 5.1.0 itself is provided either by a locally installed binary (recommended — no multi-GB pip download) **or** by a pip package. Both options are documented below.

#### Prerequisites
- x86-64 Linux, a GPU/driver compatible with Isaac Sim 5.1 ([requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html#system-requirements))
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Git + Git LFS (see the previous section)
- Python 3.11 (uv will fetch it automatically if missing)

#### Clone with submodules

```bash
git clone --recursive <this-repo-url> gearboxAssembly
cd gearboxAssembly
# or, if you already cloned without --recursive:
git submodule update --init --recursive
git lfs pull
```

The `IsaacLab/` directory is a submodule pinned to tag `v2.3.0`.

---

#### Option A — Binary Isaac Sim 5.1.0 (recommended)

Best if you already have (or are willing to install) the Isaac Sim 5.1.0 binary tarball. It avoids pip-downloading ~10 GB of isaacsim wheels into the venv.

1. **Download Isaac Sim 5.1.0** from NVIDIA's [Omniverse Launcher](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html) (or unpack a tarball). Note its install directory; the examples below assume `/home/hliu/isaac-sim-5.1`.

2. **Link Isaac Lab to the binary install** (IsaacLab expects a `_isaac_sim` entry pointing at the Kit runtime):

    ```bash
    ln -s /home/hliu/isaac-sim-5.1 IsaacLab/_isaac_sim
    ```

3. **Create the uv venv.** This installs Python 3.11, torch 2.7.0 + CUDA 12.8, IsaacLab 2.3.0 (editable from the submodule), and `Galaxea_Lab_External` (editable from `source/`). It does **not** install `isaacsim` — that comes from the binary:

    ```bash
    uv sync
    ```

4. **Activate the environment.** Use the provided helper, which activates the venv, exports `ISAAC_PATH` / `EXP_PATH` / `CARB_APP_PATH`, wires up `PYTHONPATH` (and on Linux `LD_LIBRARY_PATH` + a torch-`libgomp.so.1` preload), and auto-accepts the Omniverse EULA:

    Linux:

    ```bash
    # If you use conda, deactivate first — conda's isaaclab/isaacsim envs
    # export sticky ISAAC_PATH / PYTHONPATH that will shadow the binary you
    # just pointed at. scripts/env.sh will refuse to run otherwise.
    conda deactivate 2>/dev/null || true

    source scripts/env.sh
    ```

    Override the Isaac Sim location with `ISAAC_SIM_PATH=/path/to/isaac-sim-5.1 source scripts/env.sh`.

    Windows (PowerShell — must be **dot-sourced**):

    ```powershell
    . .\scripts\env.ps1
    ```

    Override the Isaac Sim location with `$env:ISAAC_SIM_PATH = "C:\path\to\isaac-sim-standalone-5.1.0"; . .\scripts\env.ps1` (default: `C:\isaac-sim-standalone-5.1.0`). On Windows the script just puts Isaac Sim's `site\` directory on `PYTHONPATH` — its `sitecustomize.py` does the rest of the wiring (kit / exts / extscache + DLL search paths via `os.add_dll_directory`).

5. **Verify.** You should see the environment banner and be able to launch a rule-based agent:

    ```bash
    python -c "import torch, isaaclab; print(torch.__version__, isaaclab.__version__)"
    # -> 2.7.0+cu128 0.47.2
    python scripts/list_envs.py
    python scripts/rule_based_agent.py --task=Template-Galaxea-Lab-External-Direct-v0 --enable_cameras
    ```

---

#### Option B — pip-installed Isaac Sim 5.1.0

Use this if you prefer a fully self-contained venv at the cost of disk (~18 GB total) and a longer initial sync.

1. **Edit `pyproject.toml`.** Add `isaacsim` to `dependencies` and the corresponding source/index:

    ```toml
    [project]
    dependencies = [
        # ... existing entries ...
        "isaacsim[all,extscache]==5.1.0",
    ]

    [tool.uv.sources]
    # ... existing entries ...
    isaacsim = { index = "nvidia" }

    [[tool.uv.index]]
    name = "nvidia"
    url = "https://pypi.nvidia.com"
    explicit = true
    ```

2. **Sync and activate.** No `scripts/env.sh` needed — the `isaacsim` wheel installs `.pth` files and `$VENV/bin/isaacsim`, so plain venv activation is enough:

    ```bash
    uv sync
    conda deactivate 2>/dev/null || true
    source .venv/bin/activate
    export OMNI_KIT_ACCEPT_EULA=YES
    ```

3. **Verify** as in Option A.

---

#### Notes

- **Editable install of the local package.** `source/Galaxea_Lab_External` is already wired as an editable dep in `pyproject.toml` — `uv sync` installs it. You do **not** need to run `pip install -e source/Galaxea_Lab_External` manually.
- **Conda-env leakage.** If you normally live in a conda env that activates Isaac Sim (e.g. `isaaclab`), run `conda deactivate` before `source scripts/env.sh`. The conda activation script exports `ISAAC_PATH`, `CARB_APP_PATH`, `PYTHONPATH`, `LD_LIBRARY_PATH` pointing at whichever Isaac Sim that env was built against; those leak into child processes and silently shadow whatever this project points at. `scripts/env.sh` refuses to run when `$CONDA_DEFAULT_ENV` is set to guard against this.
- **inotify watch warnings.** On first launch, Kit may log many `errno=28 No space left on device` messages — that's the inotify watch limit being hit, not disk pressure. They're non-fatal. To silence: `sudo sysctl fs.inotify.max_user_watches=524288`.
- **EULA.** `scripts/env.sh` exports `OMNI_KIT_ACCEPT_EULA=YES` for you; on Option B you must do it yourself on first run (or answer `Yes` at the prompt).

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```


    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Run the rule-based agent

- **Running the R1 gearbox assembly task with rule-based agent**
    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/rule_based_agent.py --task=Template-Galaxea-Lab-External-Direct-v0 --enable_cameras
    ```

- **Running the R1 gearbox assembly task with ACT agent**
    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/VLA_agent.py --task=Template-Galaxea-Lab-Agent-Direct-v0 --enable_cameras --checkpoint='Your-VLA-Checkpoint-File-Path'
    ```

- **Running the R1 gearbox assembly recovery tasks with rule-based agent (task 2 and 3 in the challenge setting)**
    
    We defined 3 different environments for task 2 and 3, where:

    The fourth gear (sun gear) is not yet installed, for task 2.
    ```bash
    python scripts/rule_based_agent.py --task=Gearbox-Partial-Lackfourth --enable_cameras
    ```
    The fourth gear is placed on top of one installed gear by mistake, for task 3.
    ```bash
    python scripts/rule_based_agent.py --task=Gearbox-Recovery-Misplacedfourth --enable_cameras
    ```
    The fourth gear is inclined during installation, for task 3. Rule-based agent do not perform well in this setting, and thus not provided yet. Use --no_action flag to disable actions when checking out the environment.
    ```bash
    python scripts/rule_based_agent.py --task=Gearbox-Recovery-Inclinedfourth --enable_cameras --no_action
    ```

### Switching between R1, R1_Lite and R1Pro

The repo ships with three robot configs:
- `GALAXEA_R1_BUNDLE` — the original Galaxea R1 (`r1_DVT_*.usd`).
- `GALAXEA_R1_LITE_BUNDLE` — R1_Lite (mobile-base variant; `r1_lite.usd`, base welded for tabletop tasks).
- `GALAXEA_R1_PRO_BUNDLE` (default) — R1Pro 2026 with the G1Z gripper (`r1_pro.usd`, converted from the vendor URDF at https://github.com/userguide-galaxea/URDF; 7-DOF arms, 4-DOF torso, base welded).

To switch, edit one line in `source/Galaxea_Lab_External/Galaxea_Lab_External/robots/robot_bundles.py`:

```python
ACTIVE_ROBOT_BUNDLE: RobotBundle = GALAXEA_R1_PRO_BUNDLE  # or GALAXEA_R1_LITE_BUNDLE / GALAXEA_R1_BUNDLE
```

All three task envs (`Template-Galaxea-Lab-External-Direct-v0`, `Template-Galaxea-Lab-Agent-Direct-v0`, and the `Gearbox-*` recovery tasks) read from `ACTIVE_ROBOT_BUNDLE`, so no other edits are needed. The action/observation vector size also follows the bundle (`2 * num_arm_joints + 2`: 14 for R1/R1_Lite, 16 for R1Pro). **Edit-then-restart is the supported workflow** — `ACTIVE_ROBOT_BUNDLE` is read at class-definition time by the env_cfg defaults, so reassigning it at runtime after the env_cfgs have been imported has no effect on already-defined classes.

**Caveat — rule-based agent on R1_Lite.** `r1_lite_rule_policy.py` and `r1_lite_recovery_rule_policy.py` are forks of the R1 policies with mechanical joint-name renames so the env loads, but their pose/offset constants are mainly tuned for R1 dimensions. Running `rule_based_agent.py` against R1_Lite without `--no_action` may produce unreachable motions. Use `--no_action` to inspect the scene visually.

**Caveat — R1_Lite head cameras.** The vendor URDF defines `camera_head_left_link` (collision only, no visual) and `camera_head_right_link` (empty link, no visual / collision) and does not reference the `camera_head_*_link.STL` meshes via `<visual>` tags. The `Camera` sensor in the env still attaches to those frames correctly, but the rendered scene will not show a visible camera body for the head. STL files for both head cameras are committed under `assets/Robots/R1_Lite/meshes/` and can be wired in via a URDF edit + re-conversion if a visible head body is needed.

**R1Pro notes.**
- The rule policies for R1Pro (`R1ProRulePolicy`, `R1ProRecoveryRulePolicy` in `robots/r1_pro_rule_policy.py`) are thin subclasses of the R1_Lite policies: they only override the robot-frame class attributes (`EE_LINK_SUFFIX = "_arm_link7"`, gripper axis `-Z`, a mirrored ∓90° "gripper down" wrist yaw per arm, IK-based tooth-meshing wiggle, fingertip extension, half-size DLS steps and a 1.25× phase timetable). Everything task-related is shared.
- R1Pro's arm reach (shoulder to `arm_link7`) spans only 0.30–0.57 m and the G1Z gripper adds ~0.27 m below `arm_link7`, so `GALAXEA_R1_PRO_CFG`'s torso lean, arm ready pose and the policies' ∓135° wrist yaw were chosen by offline URDF-IK against the env's randomized layout (carrier fixed at (0.45, 0), gears per side). If you move the table or change the randomization, re-check reach first — the method and the remaining unreachable corner are in `docs/r1_pro_integration.md` §6 and §10.
- The vendor URDF ships the four `*_gripper_finger_joint*` prismatic joints with zero limits/effort and no `<mimic>`; the committed `assets/Robots/R1_Pro/urdf/r1pro_2026.urdf` gives them the same treatment as R1_Lite's URDF (joint 1 in `[0, 0.065]`, joint 2 in `[-0.065, 0]` with a `<mimic multiplier=-1>` on joint 1), and only joint 1 is actuated. The Isaac Sim 5.1 importer emits that mimic as a soft 25 Hz PhysX spring, so `scripts/convert_r1_pro_urdf.py` rewrites it to a rigid constraint after conversion. `0.0` is closed, `0.055` is open. The finger collision meshes are replaced by tight pad boxes at conversion time; the convex hull of the vendor finger mesh overlaps the centreline and squeezes gears out.
- R1Pro's camera links are ROS optical frames, so the camera cfgs use `convention="ros"` with an identity offset.
- Status: the rule-based agent picks, carries and seats gears on the carrier pins with R1Pro (scores of 1–2 in headless test episodes), but it is not yet as reliable as on R1_Lite — see the known gaps in `docs/r1_pro_integration.md` (far-centre gears out of reach, occasional slip on long cross-body swings, right-arm reach to pins that rotate to the far side of the carrier).

### Re-running the R1_Lite / R1Pro URDF→USD conversion

If the vendor URDF under `source/Galaxea_Lab_External/assets/Robots/R1_Lite/urdf/` changes, regenerate the USD:

```bash
conda deactivate 2>/dev/null || true
source scripts/env.sh
python scripts/convert_r1_lite_urdf.py
```

The script rewrites `package://mobiman/...` mesh refs to relative paths in a temp URDF copy (the committed URDF is never modified) and runs `isaaclab.sim.converters.UrdfConverter` with `fix_base=True` and `make_instanceable=True`. Output is a 7-file USD bundle at `assets/Robots/R1_Lite/`:
- `r1_lite.usd` — thin wrapper, what env_cfgs reference.
- `configuration/r1_lite_base.usd` — bulk geometry (~19 MB).
- `configuration/r1_lite_physics.usd`, `configuration/r1_lite_robot.usd`, `configuration/r1_lite_sensor.usd` — composition layers.
- `config.yaml`, `.asset_hash` — metadata.

All 5 `*.usd` files route through Git LFS automatically.

The R1Pro conversion is the same shape (`scripts/convert_r1_pro_urdf.py`, vendor package under `assets/Robots/R1_Pro/`, output bundle `assets/Robots/R1_Pro/r1_pro.usd` + `configuration/r1_pro_*.usd`), with two extra URDF patches applied to the temp copy: `package://r1pro_urdf/meshes/` → `../meshes/`, and the gripper finger joint limits/mimic described above.

```bash
python scripts/convert_r1_pro_urdf.py
```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/Galaxea_Lab_External/Galaxea_Lab_External/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.


## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/Galaxea_Lab_External"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
