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

    Linux (bash or zsh):

    ```bash
    # If you use conda, deactivate first — conda's isaaclab/isaacsim envs
    # export sticky ISAAC_PATH / PYTHONPATH that will shadow the binary you
    # just pointed at. scripts/env.sh will refuse to run otherwise.
    conda deactivate 2>/dev/null || true

    source scripts/env.sh
    ```

    By default `scripts/env.sh` follows the `IsaacLab/_isaac_sim` symlink from step 2 (falling back to `$HOME/isaac-sim-5.1`). Override with `ISAAC_SIM_PATH=/path/to/isaac-sim-5.1 source scripts/env.sh`.

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

Select the robot before starting Python, without editing source:

```bash
export ROCO_ROBOT_BUNDLE=r1_lite  # or r1_pro / r1
```

All three task envs (`Template-Galaxea-Lab-External-Direct-v0`, `Template-Galaxea-Lab-Agent-Direct-v0`, and the `Gearbox-*` recovery tasks) read from `ACTIVE_ROBOT_BUNDLE`, so no other edits are needed. The action/observation vector size also follows the bundle (`2 * num_arm_joints + 2`: 14 for R1/R1_Lite, 16 for R1Pro). Restart Python after changing the selection: `ACTIVE_ROBOT_BUNDLE` is read at class-definition time by the env_cfg defaults, so reassigning it after import has no effect on already-defined classes. If the environment variable is unset, the default remains R1Pro.

**R1 Lite assembly policy.** The bundle selects `R1LiteFeedbackPolicy`, which adapts the feedback pickup, pin insertion, and central-gear meshing stages to Lite's six-axis arms and +X-facing grippers. It verifies each pickup and seating, retries bounded failures, and checks that mounted gears remain seated after retreat. The original timed `R1LiteRulePolicy` remains available for comparisons with `--policy legacy` in the evaluator.

Lite uses its own finger geometry, a 50 mm jaw opening, 35 mm pickup clearance, and 45 mm planetary transfer height. Its torso is unfolded to reach the far edge of the randomized workspace; the lower transfer path keeps a carried gear clear of the chest near the rear pin. IK retains the wrist-yaw constraint during transfer.

For the fourth gear, Lite retains its stock 100 N jaw-effort ceiling during meshing; the gentler 8 N R1Pro setting did not retain Lite's shallow rim grasp. Axial position preload remains bounded at 1.5 mm.

The policy uses live simulator object poses, supports one environment, and allows 160 simulated seconds per episode. The later ring and separate recovery sequences still need their own validation. See [R1 Lite assembly validation](docs/r1lite_validation.json) for measured results and limitations.

GPU checks on 2026-09-18 seated and retained the first three gears in **6/6 fresh runs**, covering seeds 43 and 44 with both solver profiles. Their largest final XY error was 0.594 mm against the existing 2 mm tolerance. These are two tested layouts, not a general success-rate guarantee. All 54 regression checks passed (12 Lite, 42 R1Pro/shared).

The **fourth gear remains experimental**: only **1/3 fresh runs with the final Lite settings** passed its post-retreat and parking checks:

| Seed | Physics | Fourth-gear result |
| --- | --- | --- |
| 44 | Fast, with cameras and recording | Passed at 82.75 simulated seconds |
| 43 | Fast | Moved away during release after partial meshing |
| 44 | Normal | Left the table workspace after release |

All three mounted gears remained seated in these failed fourth-gear attempts. The successful camera run retained 1,655 RGB/depth frames per camera at 20 Hz and finite six-joint arm actions. It is a four-gear partial demonstration, correctly marked `success=false`; the full five-point task was not attempted. Use `--keep_failed 3` with the main runner to retain attempts ending with at least three points in the data directory's `fail/` folder.

**Faster runs with camera recordings.** Opt in to a smaller R1Pro or R1 Lite solver budget
and run without the simulator window, keeping camera rendering enabled:

```bash
source scripts/env.sh
ROCO_ROBOT_BUNDLE=r1_pro python scripts/rule_based_agent.py \
  --task Template-Galaxea-Lab-External-Direct-v0 \
  --num_envs 1 --headless --enable_cameras --fast_physics \
  > /tmp/r1pro-recording.log 2>&1
```

For R1 Lite, use the same flag with its robot selection:

```bash
ROCO_ROBOT_BUNDLE=r1_lite python scripts/rule_based_agent.py \
  --task Template-Galaxea-Lab-External-Direct-v0 \
  --num_envs 1 --headless --enable_cameras --fast_physics \
  > /tmp/r1lite-recording.log 2>&1
```

`scripts/r1lite_rulegen.sh` also accepts and forwards `--fast_physics` when
using that script's separately configured data-generation environment.

This retains all three 320×240 RGB/depth cameras, 20 Hz recording, the HDF5
format, and the existing episode retention rules. Redirection avoids terminal
scrolling; inspect progress with `tail -f /tmp/r1pro-recording.log`. Use a different
log filename for each run if you want to retain previous logs. Keep Fabric enabled
(the default). `evaluate_r1pro.py` disables recordings and is intended for policy
evaluation rather than demonstration collection.

`--fast_physics` changes only the robot's solver budget from 128 position / 128
velocity iterations to 32 / 8. Physics remains at 100 Hz and control at 20 Hz;
collision shapes, contact parameters, controller thresholds and camera settings
are unchanged. Omit the flag to retain the original solver budget. New HDF5
files store the timestep and robot solver iterations in their attributes so
datasets collected with different physics settings can be distinguished.

For **R1 Lite**, the same short seed-44 benchmark took **5.90 s instead of
17.39 s** for 1.5 simulated seconds (about **2.9× faster**). Both runs saved
35 frames from each RGB/depth camera at 20 Hz, including five warmup frames,
with finite six-joint arm actions and the expected solver metadata. This checks
runtime and recording compatibility for the earlier timed policy. Assembly
checks for the new feedback policy are recorded separately in
[R1 Lite assembly validation](docs/r1lite_validation.json). See the
[R1 Lite performance report](docs/r1lite_performance.json).

On the RTX A6000, a short **R1Pro** seed-44 benchmark with all three cameras and recording
enabled took **5.06 s instead of 14.60 s** for 1.5 simulated seconds (about
**2.9× faster**, excluding scene startup and final HDF5 serialization). Both
configurations saved 35 RGB/depth frames per camera, including five warmup frames,
at 20 Hz. These are throughput measurements, not whole-episode speed or success
rates. A repeat through the integrated fast-profile helper took 4.95 s and
verified the saved timestep/solver metadata. All 42 regression checks passed.
Removing the window alone and reducing CPU thread counts did not produce
a meaningful improvement in this benchmark; physics was the main cost.

A lower solver budget changes numerical contact resolution, so the fast profile
requires its own seating validation. A fresh R1Pro seed-44 run with cameras and
recording enabled verified all four gears after retreat and parking at 99.25
simulated seconds (333.2 wall-clock seconds, excluding startup and final file
writing). The largest planetary XY error was 0.380 mm; the centre-gear XY error
was 3.673 mm against the unchanged 5 mm tolerance. This single layout does not
establish a general success rate or validate the later ring stage. Measurements
and validation details are in [the performance report](docs/r1pro_performance.json).
Increasing the timestep would additionally
change contact and controller sampling and is not part of this profile. See the
[NVIDIA performance guide](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/sim_performance_optimization_handbook.html)
for the distinction between viewport and sensor rendering.

To compare the feedback policy with the previous timed policy on the same seed:

```bash
source scripts/env.sh
python scripts/evaluate_r1pro.py --headless --seed 42 --policy legacy --output /tmp/r1pro-legacy-42.json
python scripts/evaluate_r1pro.py --headless --seed 42 --output /tmp/r1pro-feedback-42.json
python scripts/test_r1pro_planetary.py --headless
```

For Lite, select its bundle explicitly (the default remains R1Pro):

```bash
python scripts/evaluate_r1pro.py --robot r1_lite --headless --seed 43 --output /tmp/r1lite-normal-43.json
python scripts/evaluate_r1pro.py --robot r1_lite --headless --seed 43 --fast_physics --first-four --output /tmp/r1lite-fast-43.json
python scripts/test_r1lite_feedback.py --headless
```

Use a fresh evaluator process for each seed. JSON reports include per-gear seating,
XY errors, completion time and any failure reason; `.jsonl` traces and `.log`
files are written alongside them. The evaluator omits camera rendering and HDF5
recording while retaining production physics and scoring. Add `--first-four`
to stop after the central gear is verified, or `--full-assembly` to check
retention through the later stages. Add `--fast_physics` to evaluate the faster
solver budget explicitly; the JSON reports identify the selected settings.
For example:

```bash
ROCO_ROBOT_BUNDLE=r1_pro python scripts/evaluate_r1pro.py --headless --seed 43 --first-four --output /tmp/r1pro-four-43.json
```

`--policy planetary --full-assembly` compares the old timed fourth-gear sequence
while keeping the improved first-three controller. Use the JSON
`first_three_success`, `first_four_success`, and `assembly_success` fields for pass/fail;
Isaac Kit shutdown can override a process exit code.

Central-stage validation in Isaac Sim 5.1 on GPU (2026-09-17):

| Seed / method | First four after retreat and parking | Verified at (simulated seconds) | Central XY error (mm) | Largest planetary XY error (mm) |
| --- | --- | --- | --- | --- |
| 44, fresh episode | 4/4 | 102.70 | 2.498 | 0.517 |
| 43, continuation from three seated gears | 4/4 | 98.50 | 1.968 | 0.475 |

The seed-43 continuation starts from a naturally assembled checkpoint at 73.95
seconds and includes the complete fourth-gear pickup; no object poses were
edited. It is not an independent fresh episode. All 40 controller and geometry
regression checks passed. The first-three placement behavior remains unchanged;
the additional meshing control applies to the fourth gear. These targeted runs
do not establish a guaranteed success rate or validate completion of the later
ring/reducer sequence. GPU results, CPU diagnostic limitations, baseline results,
poses and source hashes are saved in the
[central-gear validation record](docs/r1pro_sun_validation.json).

Earlier first-three controller validation in Isaac Sim 5.1 (2026-09-17, before the central-stage changes):

| Seed | First three after retreat | Verified by (simulated seconds) | Largest XY error (mm) |
| --- | --- | --- | --- |
| 42, development | 3/3 | 88.90 | 0.412 |
| 43, development | 3/3 | 73.95 | 0.439 |
| 44, held out | 3/3 | ≤77.00 | 0.306 |

Seed 42 used one automatic regrasp; its original timed-policy baseline had only
two gears seated at 27 seconds. Seed 44 retained all three through the full run,
but finished with an overall score of 3/5: the remaining timed assembly stages
still needed improvement in that version. All 20 controller regression checks passed at that time. These three
layouts are not enough to establish a guaranteed success rate across the random
layout distribution. Poses, outcomes, conditions and source hashes are saved in
[the validation record](docs/r1pro_validation.json).

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
