import isaaclab.sim as sim_utils
from Galaxea_Lab_External import GALAXEA_LAB_ASSETS_DIR
from isaaclab.assets import (
    RigidObjectCfg,
    AssetBaseCfg,
)
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg


##
# Configuration
##
x_offset = 0.2

# TABLE_CFG: AssetBaseCfg = AssetBaseCfg(
#     prim_path="/World/envs/env_.*/Table",
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{GALAXEA_LAB_ASSETS_DIR}/Props/table/OakTableLarge.usd",
#         scale=(0.005, 0.008, 0.009),
#         collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0005),
#     ),
#     init_state=AssetBaseCfg.InitialStateCfg(
#         # pos=(0.55, 0.0, 0.0),
#         pos=(0.55 + x_offset, 0.0, 0.0),
#         rot=(-0.70711, 0.0, 0.0, 0.70711),
#     ),
# )

TABLE_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Table",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{GALAXEA_LAB_ASSETS_DIR}/Props/table/OakTableLarge.usd",
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=1.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=0.0,
            max_angular_velocity=0.0,
            enable_gyroscopic_forces=False,
            solver_position_iteration_count=1,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1.0,
        ),
        scale=(0.005, 0.008, 0.009),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.001),
    ),
    # articulation_root_properties=ArticulationRootPropertiesCfg(
    #     articulation_enabled=False
    # ),
    init_state=RigidObjectCfg.InitialStateCfg(
        # pos=(0.55, 0.0, 0.0),
        pos=(0.55 + x_offset, 0.0, 0.0),
        rot=(-0.70711, 0.0, 0.0, 0.70711),
    ),
)



# table: AssetBaseCfg = OAK_TABLE_CFG.copy()# box: AssetBaseCfg = AssetBaseCfg(
#     prim_path="{ENV_REGEX_NS}/Box",
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Props/box/box.usd",
#     ),
#     init_state=AssetBaseCfg.InitialStateCfg(
#         pos=(0.55, 0.0, 1.0),
#         rot=(1.0, 0.0, 0.0, 0.0),
#         # pos=(0.55, 0.0, 0.75), rot=(0.0, 0.0, 0.0, 1.0)  # for xht_box
#     ),
# )# t_block: RigidObjectCfg = T_BLOCK_CFG.copy()# ring_gear_CFG = RigidObjectCfg(
#     prim_path="{ENV_REGEX_NS}/ring_gear",
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Gearbox/ring_gear_3x_scale.usd",
#         rigid_props=RigidBodyPropertiesCfg(
#             solver_position_iteration_count=16,
#             solver_velocity_iteration_count=1,
#             max_angular_velocity=1000.0,
#             max_linear_velocity=1000.0,
#             max_depenetration_velocity=5.0,
#             disable_gravity=False,
#         ),
#         scale=(0.001, 0.001, 0.001),
#     ),
#     init_state=RigidObjectCfg.InitialStateCfg(
#         pos=(0.55, 0.0, 1.0),
#         rot=(1.0, 0.0, 0.0, 0.0),
#     ),
# )

RING_GEAR_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/ring_gear",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{GALAXEA_LAB_ASSETS_DIR}/Gearbox/ring_gear_3x_scale.usd",
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.5,
            linear_damping=0.01,
            angular_damping=0.01,
            max_linear_velocity=1.0,
            max_angular_velocity=1.0,
            enable_gyroscopic_forces=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=16,
            max_contact_impulse=0.5,
        ),
        scale=(0.001, 0.001, 0.001),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0, rest_offset=0.0005),
        # physics_material=sim_utils.RigidBodyMaterialCfg(),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.55, 0.0, 1.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

SUN_PLANETARY_GEAR_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/sun_planetary_gear",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{GALAXEA_LAB_ASSETS_DIR}/Gearbox/sun_planetary_gear_3x_scale.usd",
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.5,
            linear_damping=0.01,
            angular_damping=0.01,
            max_linear_velocity=1.0,
            max_angular_velocity=1.0,
            enable_gyroscopic_forces=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=16,
            max_contact_impulse=0.5,
        ),
        scale=(0.001, 0.001, 0.001),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0, rest_offset=0.0005),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.5, 0.0, 1.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

PLANETARY_CARRIER_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/planetary_carrier",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{GALAXEA_LAB_ASSETS_DIR}/Gearbox/planetary_carrier_3x_scale.usd",
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.5,
            linear_damping=0.01,
            angular_damping=0.01,
            max_linear_velocity=1.0,
            max_angular_velocity=1.0,
            enable_gyroscopic_forces=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=16,
            max_contact_impulse=0.5,
        ),
        scale=(0.001, 0.001, 0.001),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0, rest_offset=0.0005),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.6, 0.4, 1.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

# planetary_carrier_CFG = AssetBaseCfg(
#     prim_path="{ENV_REGEX_NS}/planetary_carrier",
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Gearbox/planetary_carrier_3x_scale.usd",
#         scale=(0.001, 0.001, 0.001),
#     ),
#     init_state=AssetBaseCfg.InitialStateCfg(
#         pos=(0.5, 0.0, 1.0),
#         rot=(1.0, 0.0, 0.0, 0.0),
#     ),
# )

PLANETARY_REDUCER_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/planetary_reducer",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{GALAXEA_LAB_ASSETS_DIR}/Gearbox/planetary_reducer_3x_scale.usd",
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
            linear_damping=0.01,
            angular_damping=0.01,
            max_linear_velocity=1.0,
            max_angular_velocity=1.0,
            enable_gyroscopic_forces=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=16,
            max_contact_impulse=1.0,
        ),
        scale=(0.001, 0.001, 0.001),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0, rest_offset=0.0),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.5, 0.0, 1.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)