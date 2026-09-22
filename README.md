# RoCo Challenge: Gearbox Assembly

An Isaac Lab simulation for bimanual gearbox assembly with Galaxea R1, R1 Lite,
and R1Pro robots. It includes scripted assembly, partial-assembly and recovery
scenarios, camera recording, and an ACT policy runner for the original R1.

See the [RoCo Challenge @ AAAI 2026](https://rocochallenge.github.io/RoCo2026/doc.html)
for challenge details.

![RoCo Challenge Poster](docs/images/poster.png)

## Installation

You will need:

- [Isaac Sim 5.1.0](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html), installed using NVIDIA's workstation instructions, and a machine meeting its [system requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html#system-requirements).
- [uv](https://docs.astral.sh/uv/getting-started/installation/), Git, and Git LFS.

The project uses Python 3.11 and Isaac Lab 2.3.0. The setup below installs the
Python dependencies into the project's `.venv`; uv can download Python if needed.

### Download the repository and assets

```bash
git lfs install
git clone --recursive https://github.com/rocochallenge/gearboxAssembly.git
cd gearboxAssembly
git submodule update --init --recursive
git lfs pull
```

For an existing clone, run the last two commands from the repository root.
Git LFS is required to download the actual robot and gearbox models.
If the download fails because of LFS bandwidth limits, use the
[asset archive](https://drive.google.com/file/d/1L7u89xxiHGkd72CzvZln3P5uPPqMu7b7/view?usp=drive_link)
and extract its contents into `source/Galaxea_Lab_External/assets`.

### Set up the environment

Run setup and all examples from the repository root. If a Conda environment is
active, run `conda deactivate` first.

On **Linux (Bash or Zsh)**, replace the Isaac Sim path below if you installed it
elsewhere. Create the link once, then install the dependencies and activate:

```bash
ln -s "$HOME/isaac-sim-5.1" IsaacLab/_isaac_sim
uv sync
source scripts/env.sh
```

In each new terminal, run `source scripts/env.sh` again before launching a task.

On **Windows (PowerShell)**, set the path to your Isaac Sim installation:

```powershell
uv sync
$env:ISAAC_SIM_PATH = "C:\isaac-sim-standalone-5.1.0"
. .\scripts\env.ps1
```

Repeat the last two lines in each new PowerShell session. Both activation helpers
set Isaac Sim's EULA acceptance by default.

## Run the assembly task

Choose a robot before starting the Python process:

| Robot | `ROCO_ROBOT_BUNDLE` value |
| --- | --- |
| R1Pro (default) | `r1_pro` |
| R1 Lite | `r1_lite` |
| Original R1 | `r1` |

The following examples use Bash or Zsh. Change the selection to run another robot:

```bash
export ROCO_ROBOT_BUNDLE=r1_pro
python scripts/rule_based_agent.py \
  --task Template-Galaxea-Lab-External-Direct-v0 \
  --num_envs 1 --enable_cameras
```

In PowerShell, select the robot with `$env:ROCO_ROBOT_BUNDLE = "r1_pro"` and put
the Python command on one line. If no robot is selected, R1Pro is used.
Start a new Python process after changing robots.

The simulator opens a window and runs repeated episodes. Close the window or
press `Ctrl+C` to stop. Use `--num_envs 1` with the supplied assembly policies.
R1Pro and R1 Lite use feedback to place the three planetary gears, central gear,
and outer ring. Both verify that all five remain seated after releasing the
gripper and parking the arm.

### Faster runs with camera recording

For R1Pro or R1 Lite, add `--headless --fast_physics`:

```bash
python scripts/rule_based_agent.py \
  --task Template-Galaxea-Lab-External-Direct-v0 \
  --num_envs 1 --headless --enable_cameras --fast_physics
```

`--headless` hides the simulator window. Keep `--enable_cameras` to retain camera
images and demonstrations. `--fast_physics` reduces physics computation and can
change assembly outcomes; omit it to use the standard physics settings.
This flag is available for R1Pro and R1 Lite.

## Recordings

For the assembly task above, demonstrations are saved as **HDF5 files** containing
RGB and depth images from three cameras, joint states, actions, scores, and
timestamps at 20 Hz. MP4 videos are not created automatically.
Recordings also include torso joint states and targets, so R1 Lite's torso
movement during ring placement is captured alongside its six-joint arm actions.

The default output folder is `../data`, relative to the working directory. When
running from the repository root, this is a `data` folder beside `gearboxAssembly`.
To choose another folder, set this before starting the runner:

```bash
export ROCO_DATA_DIR="$PWD/data"
```

In PowerShell, use `$env:ROCO_DATA_DIR = "$PWD\data"`.

By default, only complete, successful assemblies (score 5/5) are saved. To retain
partial or unsuccessful attempts, append one of these options to the assembly
command:

| Option | Episodes saved in addition to successes |
| --- | --- |
| `--keep_failed` | All unsuccessful episodes |
| `--keep_failed 3` | Unsuccessful episodes with a final score of at least 3 |

Successful files are named `data_<timestamp>.hdf5`. Retained unsuccessful files
are written to `fail/fail_score<score>_data_<timestamp>.hdf5` within the output
folder.

Files are written when an episode finishes; the terminal prints the saved path.
Stopping in the middle of an episode does not save that unfinished episode.

### Collect a fixed number of demonstrations (Linux)

The batch collector runs fresh seeds until it has the requested number of
successful, verified recordings, or reaches the attempt limit:

```bash
source scripts/env.sh
python scripts/collect_assembly_data.py \
  --robot r1_pro --successes 10 --seed-start 1000 --max-attempts 50 \
  --output "$PWD/data/r1_pro_batch" --runtime-label workstation-isaacsim5.1 \
  --fast_physics
```

Use `--robot r1_lite` and a separate output folder for R1 Lite. For multiple
workers, give each one its own folder and a non-overlapping seed range. Cameras
are enabled automatically. Each accepted seed folder contains `episode.hdf5`,
episode metadata, a score trace, and a verification report. Failed attempts keep
their logs but do not count toward the quota.

Read `status.json` in the output folder for progress. Repeating the same command
resumes the collection and skips completed attempts. To stop after the current
episode, create an empty `STOP` file in that folder; remove it before resuming.
The collector leaves partial files unaccepted and stops after two consecutive
runtime or recording errors. Data under the project's `data/` folder is excluded
from Git.

For a collection organized as `data/batch/<workstation>/<robot>/seedNNNN/`,
build a combined index after copying the worker folders to one machine:

```bash
python scripts/index_assembly_data.py data/batch --verify-hashes
```

This writes `dataset.json` and `episodes.jsonl` with relative paths to accepted
recordings, preserving robot, seed, workstation, and simulator version.

## Other tasks

### Partial assembly and recovery

These scenarios start from an already partially assembled gearbox:

| Task name | Starting condition |
| --- | --- |
| `Gearbox-Partial-Lackfourth` | The fourth (central) gear has not been installed |
| `Gearbox-Recovery-Misplacedfourth` | The fourth gear is resting on an installed gear |
| `Gearbox-Recovery-Inclinedfourth` | The fourth gear is tilted during insertion |

For example, run the partial-assembly scenario with the original R1:

```bash
ROCO_ROBOT_BUNDLE=r1 python scripts/rule_based_agent.py \
  --task Gearbox-Partial-Lackfourth --num_envs 1 --enable_cameras
```

Replace the task name with `Gearbox-Recovery-Misplacedfourth` for the misplaced
gear scenario. The inclined-gear scenario has no working scripted recovery;
inspect it with robot actions disabled:

```bash
ROCO_ROBOT_BUNDLE=r1 python scripts/rule_based_agent.py \
  --task Gearbox-Recovery-Inclinedfourth \
  --num_envs 1 --enable_cameras --no_action
```

Recovery recordings use `../data/data_recovery_<timestamp>.hdf5`. The assembly
runner's `ROCO_DATA_DIR` and `--keep_failed` settings do not apply to these tasks.

### Run an ACT checkpoint

The supplied ACT runner uses the original R1. Provide a checkpoint matching its
ACT model configuration and place the accompanying `dataset_stats.pkl` in the
same directory as the checkpoint:

```bash
ROCO_ROBOT_BUNDLE=r1 python scripts/VLA_agent.py \
  --task Template-Galaxea-Lab-Agent-Direct-v0 \
  --num_envs 1 --enable_cameras \
  --checkpoint /path/to/policy_best.ckpt
```

## Troubleshooting

- **`ModuleNotFoundError: No module named 'h5py'` or another project dependency:**
  run `uv sync`, then activate with `source scripts/env.sh` (or `. .\scripts\env.ps1`
  on Windows) in the same terminal used to launch the task.
- **Isaac Sim cannot be found:** check the installation path. On Linux, set
  `export ISAAC_SIM_PATH="/path/to/isaac-sim-5.1"` before sourcing `scripts/env.sh`;
  on Windows, set `$env:ISAAC_SIM_PATH` before dot-sourcing `scripts/env.ps1`.
- **Missing robot or gearbox assets:** run `git lfs pull` from the repository root.
  Use the asset archive above if the download is blocked by the LFS quota.
- **No assembly recordings appear:** keep `--enable_cameras`, wait for an episode
  to finish, and use `--keep_failed` if you want to save unsuccessful attempts.
  Check `ROCO_DATA_DIR` or the default `../data` folder.
