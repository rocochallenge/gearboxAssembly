# RoCo Challenge: Robotic Collaborative Assembling - HMI Workshop @ AAAI 2026
![RoCo Challenge Poster](docs/images/poster.png)
Code Repository for the [RoCo Challenge@AAAI 2026](https://rocochallenge.github.io/RoCo2026/doc.html)

The Gearbox Assembly Assistance Challenge evaluates bimanual robotic systems in collaborative gearbox assembly within manufacturing environments. It targets scenarios where robots work seamlessly with human operators.

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

---
### Installation Steps
- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/Galaxea_Lab_External

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

- **Running the R1 gearbox assembly task with rule-based agent**
    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
     python scripts/rule_based_agent.py --task=Template-Galaxea-Lab-External-Direct-v0 --enable_cameras
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

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

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