# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

import os
import toml

# Conveniences to other module directories via relative paths
# ISAACLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "./assets"))
GALAXEA_LAB_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

# ISAACLAB_ASSETS_DATA_DIR = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "data")
GALAXEA_LAB_ASSETS_DIR = os.path.join(GALAXEA_LAB_EXT_DIR, "assets")
"""Path to the extension data directory."""

# ISAACLAB_ASSETS_METADATA = toml.load(os.path.join(ISAACLAB_ASSETS_EXT_DIR, "config", "extension.toml"))
GALAXEA_LAB_METADATA = toml.load(os.path.join(GALAXEA_LAB_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *

from .robots import *

from .VLA import *