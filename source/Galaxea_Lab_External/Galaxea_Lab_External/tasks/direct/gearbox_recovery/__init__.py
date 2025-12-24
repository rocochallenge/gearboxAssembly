# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

# from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Gearbox-Partial-Lackfourth",
    entry_point=f"{__name__}.gearbox_recovery_env:GalaxeaLabExternalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gearbox_recovery_env_cfg:GalaxeaLabExternalEnvCfg",
        "initial_assembly_state": "lack_fourth_gear",
    },
)
gym.register(
    id="Gearbox-Recovery-Misplacedfourth",
    entry_point=f"{__name__}.gearbox_recovery_env:GalaxeaLabExternalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gearbox_recovery_env_cfg:GalaxeaLabExternalEnvCfg",
        "initial_assembly_state": "misplaced_fourth_gear",
    },
)
gym.register(
    id="Gearbox-Recovery-Inclinedfourth",
    entry_point=f"{__name__}.gearbox_recovery_env:GalaxeaLabExternalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gearbox_recovery_env_cfg:GalaxeaLabExternalEnvCfg",
        "initial_assembly_state": "inclined_fourth_gear",
    },
)