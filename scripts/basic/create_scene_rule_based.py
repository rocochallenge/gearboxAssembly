import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="create scene using the interactive scene interface"
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="number of environments to spawn"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils

from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR
from Galaxea_Lab_External import GALAXEA_LAB_ASSETS_DIR
print("GALAXEA_LAB_ASSETS_DIR: ", GALAXEA_LAB_ASSETS_DIR)
from Galaxea_Lab_External.robots import (
    GALAXEA_R1_CHALLENGE_CFG,
    TABLE_CFG,
    RING_GEAR_CFG,
    SUN_PLANETARY_GEAR_CFG,
    PLANETARY_CARRIER_CFG,
    PLANETARY_REDUCER_CFG,
)

import omni.appwindow
import carb.input
from carb.input import KeyboardEventType
from isaaclab.sensors import ContactSensorCfg, CameraCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils


from pxr import Usd, Sdf, UsdPhysics, UsdGeom
from isaaclab.sim.spawners.materials import physics_materials, physics_materials_cfg
from isaaclab.sim.spawners.materials import spawn_rigid_body_material

# from isaacsim.core.experimental.materials import set_friction_coefficients

# print("GALAXEA_LAB_ASSETS_DIR: ", GALAXEA_LAB_ASSETS_DIR)

left_gripper_position = 0.0
right_gripper_position = 0.0

target_position_left = torch.zeros(3)
target_position_right = torch.zeros(3)

TCP_offset_z = 1.1475 - 1.05661
TCP_offset_x = 0.3864 - 0.3785
table_height = 0.9
grasping_height = -0.003

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_rigidbody(path):
    stage = Usd.Stage.Open(Usd.Stage.GetCurrent().GetRootLayer().realPath) 
    prim = stage.GetPrimAtPath(Sdf.Path(path))
    if not prim.IsValid():
        print(f"[WARN] Prim not found: {path}")
        return
    # Apply RigidBody + Collision to the prim (common case)
    if not UsdPhysics.RigidBodyAPI(prim):
        UsdPhysics.RigidBodyAPI.Apply(prim)
        print(f"[OK] Applied RigidBodyAPI -> {path}")
    # If you also need collision
    if not UsdPhysics.CollisionAPI(prim):
        UsdPhysics.CollisionAPI.Apply(prim)
        print(f"[OK] Applied CollisionAPI -> {path}")

@configclass
class R1SceneCfg(InteractiveSceneCfg):
    """Configuration for a galaxea-R1 scene."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(color=(1.0, 1.0, 1.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75)),
    )

    table: AssetBaseCfg = TABLE_CFG.copy()

    robot: ArticulationCfg = GALAXEA_R1_CHALLENGE_CFG.copy()

    # target_frame_left = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/TargetFrameLeft",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.01, 0.01, 0.01),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(0.3864, 0.5237, 1.1475),
    #         # rot=(9.6247e-05, 9.7698e-01, -2.1335e-01, 3.9177e-04),
    #         rot=(0.0, -1.0, 0.0, 0.0),
    #     ),
    # )

    # target_frame_right = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/TargetFrameRight",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.01, 0.01, 0.01),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(0.3864, -0.5237, 1.1475),
    #         # rot=(9.6247e-05, 9.7698e-01, -2.1335e-01, 3.9177e-04),
    #         rot=(0.0, -1.0, 0.0, 0.0),
    #     ),
    # )

    # head_camera: CameraCfg = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/head_cam",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=12,
    #         focus_distance=100.0,
    #         horizontal_aperture=20.955,
    #         clipping_range=(0.01, 100),
    #     ),
    #     offset=CameraCfg.OffsetCfg(  # offset from the parent frame
    #         pos=(0.0, 0.0, 0.0),
    #         rot=(1.0, 0.0, 0.0, 0.0),
    #         convention="ros",
    #     ),
    # )

    # contact_forces_l1 = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/left_gripper_link1",
    #     update_period=0.0,
    #     history_length=6,
    #     debug_vis=False,
    # )

    # contact_forces_l2 = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/left_gripper_link2",
    #     update_period=0.0,
    #     history_length=6,
    #     debug_vis=False,
    # )

    # contact_forces_r1 = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/right_gripper_link1",
    #     update_period=0.0,
    #     history_length=6,
    #     debug_vis=False,
    # )

    # contact_forces_r2 = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/right_gripper_link2",
    #     update_period=0.0,
    #     history_length=6,
    #     debug_vis=False,
    # )
    
    ring_gear: RigidObjectCfg = RING_GEAR_CFG.replace(prim_path="{ENV_REGEX_NS}/ring_gear",
                                                                       init_state=RigidObjectCfg.InitialStateCfg(
                                                                           pos=(0.45, 0.0, 1.0),
                                                                           rot=(1.0, 0.0, 0.0, 0.0),
                                                                       ))


    sun_planetary_gear_1: RigidObjectCfg = SUN_PLANETARY_GEAR_CFG.replace(prim_path="{ENV_REGEX_NS}/sun_planetary_gear_1",
                                                                       init_state=RigidObjectCfg.InitialStateCfg(
                                                                           pos=(0.4, -0.2, 1.0),
                                                                           rot=(1.0, 0.0, 0.0, 0.0),
                                                                       ))
    
    sun_planetary_gear_2: RigidObjectCfg = SUN_PLANETARY_GEAR_CFG.replace(prim_path="{ENV_REGEX_NS}/sun_planetary_gear_2",
                                                                       init_state=RigidObjectCfg.InitialStateCfg(
                                                                           pos=(0.5, -0.25, 1.0),
                                                                           rot=(1.0, 0.0, 0.0, 0.0),
                                                                       ))
    sun_planetary_gear_3: RigidObjectCfg = SUN_PLANETARY_GEAR_CFG.replace(prim_path="{ENV_REGEX_NS}/sun_planetary_gear_3",
                                                                       init_state=RigidObjectCfg.InitialStateCfg(
                                                                           pos=(0.45, -0.15, 1.0),
                                                                           rot=(1.0, 0.0, 0.0, 0.0),
                                                                       ))
    sun_planetary_gear_4: RigidObjectCfg = SUN_PLANETARY_GEAR_CFG.replace(prim_path="{ENV_REGEX_NS}/sun_planetary_gear_4",
                                                                       init_state=RigidObjectCfg.InitialStateCfg(
                                                                           pos=(0.55, -0.3, 1.0),
                                                                           rot=(1.0, 0.0, 0.0, 0.0),
                                                                       ))
    planetary_carrier: RigidObjectCfg = PLANETARY_CARRIER_CFG.replace(prim_path="{ENV_REGEX_NS}/planetary_carrier",
                                                                       init_state=RigidObjectCfg.InitialStateCfg(
                                                                           pos=(0.5, 0.25, 1.0),
                                                                           rot=(1.0, 0.0, 0.0, 0.0),
                                                                       ))
    planetary_reducer: RigidObjectCfg = PLANETARY_REDUCER_CFG.replace(prim_path="{ENV_REGEX_NS}/planetary_reducer",
                                                                       init_state=RigidObjectCfg.InitialStateCfg(
                                                                           pos=(0.3, 0.1, 1.0),
                                                                           rot=(0.7071068 , 0.0, 0.0, 0.7071068),
                                                                       ))

def randomize_object_positions(sim: sim_utils.SimulationContext, scene: InteractiveScene, object_names: list,
                              safety_margin: float = 0.02, max_attempts: int = 1000):
    """Randomize positions of objects on table without overlapping

    This function places objects on the table surface ensuring they don't overlap by checking
    their bounding radii. Each object type has a defined approximate radius that represents
    its circular footprint on the table.

    Args:
        sim: Simulation context
        scene: Interactive scene containing the objects
        object_names: List of object names to randomize
        safety_margin: Additional safety distance to add between objects (in meters)
        max_attempts: Maximum attempts to find a non-overlapping position per object per environment

    Note:
        Adjust the OBJECT_RADII dictionary below if your objects have different sizes.
        To measure the radius of an object, observe it in the simulation and estimate
        the distance from its center to its furthest edge in the XY plane.
    """
    # Define approximate radii for each object type (in meters)
    # These values represent the circular bounding area of each object on the table surface
    # Adjust these based on your actual object sizes
    OBJECT_RADII = {
        'ring_gear': 0.1,              # Largest gear
        'sun_planetary_gear_1': 0.035,  # Small planetary gears
        'sun_planetary_gear_2': 0.035,
        'sun_planetary_gear_3': 0.035,
        'sun_planetary_gear_4': 0.035,
        'planetary_carrier': 0.07,     # Medium-large carrier
        'planetary_reducer': 0.04,     # Medium reducer
    }

    initial_root_state = {obj_name: torch.zeros((scene.num_envs, 7), device=sim.device) for obj_name in object_names}

    num_envs = scene.num_envs

    # Store positions and object names of already placed objects for each environment
    # Each entry is a tuple: (position_tensor, object_name)
    placed_objects = [[] for _ in range(num_envs)]

    for obj_name in object_names:
        obj_cfg = SceneEntityCfg(obj_name, body_names=['node_'])
        print(f"obj_cfg: {obj_cfg}")
        obj_cfg.resolve(scene)

        obj = scene[obj_name]
        root_state = obj.data.default_root_state.clone()

        # Get radius for current object
        current_radius = OBJECT_RADII.get(obj_name, 0.05)  # Default to 0.05m if not specified

        # Generate non-overlapping positions for each environment
        for env_idx in range(num_envs):
            position_found = False

            for attempt in range(max_attempts):
                # Generate random position
                x = torch.rand(1, device=sim.device).item() * 0.2 + 0.3  # range [0.3, 0.6]
                y = torch.rand(1, device=sim.device).item() * 0.6 - 0.3  # range [-0.3, 0.3]
                z = 0.92

                if obj_name == "ring_gear":
                    x = 0.24
                    y = 0.0
                elif obj_name == "planetary_carrier":
                    x = 0.42
                    y = 0.0
                elif obj_name == "sun_planetary_gear_1":
                    y = torch.rand(1, device=sim.device).item() * 0.3
                elif obj_name == "sun_planetary_gear_2":
                    y = torch.rand(1, device=sim.device).item() * 0.3
                elif obj_name == "sun_planetary_gear_3":
                    y = -torch.rand(1, device=sim.device).item() * 0.3
                elif obj_name == "sun_planetary_gear_4":
                    y = -torch.rand(1, device=sim.device).item() * 0.3


                pos = torch.tensor([x, y, z], device=sim.device)

                # Check for overlaps with already placed objects in this environment
                is_valid = True
                for placed_pos, placed_obj_name in placed_objects[env_idx]:
                    # Get radius of the already placed object
                    placed_radius = OBJECT_RADII.get(placed_obj_name, 0.05)

                    # Calculate minimum required distance (sum of radii + safety margin)
                    min_distance = current_radius + placed_radius + safety_margin

                    # Check only x, y distance (ignore z for table surface)
                    distance = torch.norm(pos[:2] - placed_pos[:2]).item()
                    if distance < min_distance:
                        is_valid = False
                        break

                if is_valid:
                    # Position is valid, use it
                    root_state[env_idx, :3] = pos
                    placed_objects[env_idx].append((pos, obj_name))
                    position_found = True
                    break

            if not position_found:
                # Max attempts reached, use the last generated position anyway with a warning
                print(f"[WARN] Could not find non-overlapping position for {obj_name} in env {env_idx} after {max_attempts} attempts.")
                print(f"       This may indicate the table area is too crowded. Consider reducing the number of objects")
                print(f"       or increasing the table area (x: [0.2, 0.5], y: [-0.3, 0.3]).")
                root_state[env_idx, :3] = pos
                placed_objects[env_idx].append((pos, obj_name))

        # Write the state to simulation
        obj.write_root_state_to_sim(root_state)
        initial_root_state[obj_name] = root_state.clone()

    return initial_root_state

def get_config(sim: sim_utils.SimulationContext, scene: InteractiveScene,
               arm_name: str):
    # arm_name: left or right

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg, num_envs=scene.num_envs, device=sim.device
    )

    # Specify robot-specific parameters
    arm_entity_cfg = SceneEntityCfg(
        "robot", joint_names=[f"{arm_name}_arm_joint.*"], body_names=[f"{arm_name}_arm_link6"]
    )
    gripper_entity_cfg = SceneEntityCfg(
        "robot", joint_names=[f"{arm_name}_gripper_axis.*"]
    )

    # Resolving the scene entities
    arm_entity_cfg.resolve(scene)
    gripper_entity_cfg.resolve(scene)
    
    # gripper_entity_cfg = SceneEntityCfg("robot", joint_names=[f"{arm_name}_gripper_.*"], body_names=[f"{arm_name}_gripper_link1"])
    # gripper_entity_cfg.resolve(scene)
    
    return diff_ik_controller, arm_entity_cfg, gripper_entity_cfg

def move_robot_to_position(sim: sim_utils.SimulationContext, scene: InteractiveScene, 
                           arm_entity_cfg: SceneEntityCfg,
                           gripper_entity_cfg: SceneEntityCfg,
                           diff_ik_controller: DifferentialIKController,
                           target_position: torch.Tensor, target_orientation: torch.Tensor,
                           target_marker: VisualizationMarkers):
    robot = scene["robot"]

    arm_joint_ids = arm_entity_cfg.joint_ids
    arm_body_ids = arm_entity_cfg.body_ids
    num_arm_joints = len(arm_joint_ids)

    gripper_joint_ids = gripper_entity_cfg.joint_ids
    gripper_body_ids = gripper_entity_cfg.body_ids
    num_gripper_joints = len(gripper_joint_ids)

    if robot.is_fixed_base:
        ee_jacobi_idx = arm_body_ids[0] - 1
    else:
        ee_jacobi_idx = arm_body_ids[0]

    # Get the target position and orientation of the arm
    # print(f"target_position: {target_position}, target_orientation: {target_orientation}")
    ik_commands = torch.cat([target_position, target_orientation], dim=-1)
    diff_ik_controller.set_command(ik_commands)

    # IK solver
    # obtain quantities from simulation
    jacobian = robot.root_physx_view.get_jacobians()[
        :, ee_jacobi_idx, :, arm_entity_cfg.joint_ids
    ]
    ee_pose_w = robot.data.body_state_w[
        :, arm_body_ids[0], 0:7
    ]
    root_pose_w = robot.data.root_state_w[:, 0:7]
    joint_pos = robot.data.joint_pos[:, arm_entity_cfg.joint_ids]
    # compute frame in root frame
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],
        ee_pose_w[:, 0:3],
        ee_pose_w[:, 3:7],
    )
    # compute the joint commands
    joint_pos_des = diff_ik_controller.compute(
        ee_pos_b, ee_quat_b, jacobian, joint_pos
    )

    print(f"ee_pos_b: {ee_pos_b}, ee_quat_b: {ee_quat_b}")
    print(f"joint_pos_des: {joint_pos_des}")

    # Apply
    # print(f"joint_pos_des: {joint_pos_des}")
    robot.set_joint_position_target(
        joint_pos_des, joint_ids=arm_entity_cfg.joint_ids
    )
    # gripper_joint_pos_des = torch.full(
    #     (num_gripper_joints,), gripper_position, device=robot.device
    # )
    # robot.set_joint_position_target(
    #     gripper_joint_pos_des, joint_ids=gripper_joint_ids
    # )

    # print(f"target_position: {target_position}, target_orientation: {target_orientation}")
    


def prepare_mounting_plan(sim: sim_utils.SimulationContext, scene: InteractiveScene,
                          left_arm_entity_cfg: SceneEntityCfg,
                          right_arm_entity_cfg: SceneEntityCfg,
                          initial_root_state: dict,
                          gear_names: list = None):
    """
    Plan which gear mounts to which arm and which pin by finding the nearest arm and pin for each gear.
    Each pin can only be used once.

    Args:
        sim: Simulation context
        scene: Interactive scene containing the robot and objects
        left_arm_entity_cfg: Configuration for left arm
        right_arm_entity_cfg: Configuration for right arm
        initial_root_state: Dictionary containing initial states of all objects
        gear_names: List of gear names to plan for (e.g., ['sun_planetary_gear_1', 'sun_planetary_gear_2'])
                   If None, defaults to all 4 sun planetary gears

    Returns:
        gear_to_arm_pin_map: Dictionary mapping gear_name -> {'arm': 'left'/'right', 'pin': pin_index,
                                                               'pin_world_pos': tensor, 'pin_world_quat': tensor}
    """

    # Default to all 4 gears if not specified
    if gear_names is None:
        gear_names = ['sun_planetary_gear_1', 'sun_planetary_gear_2',
                     'sun_planetary_gear_3', 'sun_planetary_gear_4']

    # Get the planetary carrier positions and orientations
    root_state = initial_root_state["planetary_carrier"]
    planetary_carrier_pos = root_state[:, :3].clone()
    planetary_carrier_quat = root_state[:, 3:7].clone()
    num_envs = planetary_carrier_pos.shape[0]

    # Define pin positions in local coordinates (relative to planetary carrier)
    pin_local_positions = [
        torch.tensor([0.0, -0.054, 0.0], device=sim.device),      # pin_0
        torch.tensor([0.0465, 0.0268, 0.0], device=sim.device),   # pin_1
        torch.tensor([-0.0465, 0.0268, 0.0], device=sim.device),  # pin_2
    ]

    # Calculate world positions of all pins
    pin_world_positions = []
    pin_world_quats = []
    for pin_local_pos in pin_local_positions:
        pin_local_pos_batch = pin_local_pos.unsqueeze(0).expand(num_envs, -1)
        pin_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device).unsqueeze(0).expand(num_envs, -1)

        pin_world_pos = torch_utils.tf_combine(
            planetary_carrier_quat, planetary_carrier_pos, pin_quat, pin_local_pos_batch)[1]

        pin_world_positions.append(pin_world_pos)
        pin_world_quats.append(pin_quat)

    # Stack all pin positions: shape (num_envs, num_pins, 3)
    pin_world_positions = torch.stack(pin_world_positions, dim=1)
    pin_world_quats = torch.stack(pin_world_quats, dim=1)

    # Get end-effector positions for both arms
    left_ee_pos = scene["robot"].data.body_state_w[:, left_arm_entity_cfg.body_ids[0], 0:3]
    right_ee_pos = scene["robot"].data.body_state_w[:, right_arm_entity_cfg.body_ids[0], 0:3]

    # Track which pins are occupied
    occupied_pins = set()

    # Result mapping
    gear_to_arm_pin_map = {}

    # For each gear, find nearest arm and nearest available pin
    for gear_name in gear_names:
        if gear_name not in initial_root_state:
            print(f"[WARN] Gear {gear_name} not found in initial_root_state, skipping")
            continue

        # Get gear position
        gear_pos = initial_root_state[gear_name][:, :3].clone()  # shape: (num_envs, 3)

        # For the first environment (env_idx=0), calculate distances to both arms
        env_idx = 0
        gear_pos_env = gear_pos[env_idx]

        # Calculate distance to both arms
        left_dist = torch.norm(gear_pos_env - left_ee_pos[env_idx])
        right_dist = torch.norm(gear_pos_env - right_ee_pos[env_idx])

        # Choose the nearest arm
        if left_dist < right_dist:
            chosen_arm = 'left'
            chosen_arm_pos = left_ee_pos[env_idx]
        else:
            chosen_arm = 'right'
            chosen_arm_pos = right_ee_pos[env_idx]

        # Find the nearest available pin
        min_pin_dist = float('inf')
        nearest_pin_idx = None

        for pin_idx in range(len(pin_local_positions)):
            if pin_idx in occupied_pins:
                continue  # Skip occupied pins

            pin_pos = pin_world_positions[env_idx, pin_idx]
            pin_dist = torch.norm(gear_pos_env - pin_pos)

            if pin_dist < min_pin_dist:
                min_pin_dist = pin_dist
                nearest_pin_idx = pin_idx

        if nearest_pin_idx is None:
            print(f"[WARN] No available pins for gear {gear_name}, all pins occupied!")
            continue

        # Mark this pin as occupied
        occupied_pins.add(nearest_pin_idx)

        # Store the mapping
        gear_to_arm_pin_map[gear_name] = {
            'arm': chosen_arm,
            'pin': nearest_pin_idx,
            'pin_local_pos': pin_local_positions[nearest_pin_idx],
            'pin_world_pos': pin_world_positions[:, nearest_pin_idx],  # All environments
            'pin_world_quat': pin_world_quats[:, nearest_pin_idx],
        }

        print(f"[INFO] {gear_name} -> {chosen_arm} arm, pin_{nearest_pin_idx}")
        print(f"       Gear pos: {gear_pos_env}, Pin pos: {pin_world_positions[env_idx, nearest_pin_idx]}")
        print(f"       Distance: {min_pin_dist:.4f}m")

    return gear_to_arm_pin_map


# Step 1: Pick up the sun_planetary_gear_1
# grasping_state = 1
# joint_pos = None
def pick_up_sun_planetary_gear(sim: sim_utils.SimulationContext, scene: InteractiveScene, 
                                 count: int, 
                                 gear_id: int, # gear_id: 1, 2, 3, 4, 5(carrier)
                                 gear_to_pin_map: dict,
                                 count_step: torch.Tensor,
                                 arm_entity_cfg: SceneEntityCfg,
                                 gripper_entity_cfg: SceneEntityCfg,
                                 diff_ik_controller: DifferentialIKController,
                                 initial_root_state: dict,
                                 target_marker: VisualizationMarkers):
    
    sim_dt = sim.get_physics_dt()

    if gear_id == 5:
        # root_state = initial_root_state["planetary_carrier"]
        planetary_carrier_pos = scene["planetary_carrier"].data.root_state_w[:, :3].clone()
        planetary_carrier_quat = scene["planetary_carrier"].data.root_state_w[:, 3:7].clone()

        print(f"planetary_carrier_pos: {planetary_carrier_pos}")
        print(f"planetary_carrier_quat: {planetary_carrier_quat}")

        # planetary_carrier_pos = root_state[:, :3].clone()
        # planetary_carrier_quat = root_state[:, 3:7].clone()
        local_pos = torch.tensor([0.0, 0.054, 0.0], device=sim.device).unsqueeze(0)

        target_orientation, target_position = torch_utils.tf_combine(
            planetary_carrier_quat, planetary_carrier_pos, 
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device), local_pos
        )
        root_state = torch.cat([target_position, target_orientation], dim=-1)
        
    else:
        root_state = initial_root_state[f"sun_planetary_gear_{gear_id}"]
    # print(f"obj: {obj}")
    # target_position, target_orientation = target_frame.get_local_pose()
    # target_position, target_orientation = target_frame.get_world_poses()
    target_position = root_state[:, :3].clone()
    target_position[:, 2] = table_height + grasping_height
    target_position = target_position + torch.tensor([TCP_offset_x, 0.0, TCP_offset_z], device=sim.device)
    # target_orientation = obj.data.default_root_state[:, 3:7].clone()
    # print(f"target_position: {target_position}, target_orientation: {target_orientation}")
    # Step 1.1: Move the arm to the target position above the gear and keep the orientation
    target_position_h = target_position + torch.tensor([0.0, 0.0, 0.1], device=sim.device)
    
    # target_orientation = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=sim.device)
    target_orientation = root_state[:, 3:7].clone()
    # Rotate the target orientation 180 degrees around the y-axis
    target_orientation, target_position = torch_utils.tf_combine(
        target_orientation, target_position, 
        torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=sim.device), torch.tensor([[0.0, 0.0, 0.0]], device=sim.device)
    )

    # print(f"target_position: {target_position}, target_orientation: {target_orientation}")
    # print(f"target_position_h: {target_position_h}, target_orientation: {target_orientation}")



    # ik_commands = torch.cat([target_position, target_orientation], dim=-1,)

    # print("scene['robot'].data.joint_pos: ", scene["robot"].data.joint_pos)
    if count >= count_step[0] and count < count_step[1]:
        move_robot_to_position(sim, scene, arm_entity_cfg, gripper_entity_cfg, diff_ik_controller, 
                                target_position_h, target_orientation, target_marker)
        target_marker.visualize(target_position_h, target_orientation)
    
    # Step 1.2: Open the gripper
    gripper_joint_ids = gripper_entity_cfg.joint_ids
    gripper_body_ids = gripper_entity_cfg.body_ids
    num_gripper_joints = len(gripper_joint_ids)

    # Step 1.3: Move the arm to the target position and keep the orientation
    # target_position_2 = target_position
    # target_orientation = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=sim.device)
    if count >= count_step[1] and count < count_step[2]:
        move_robot_to_position(sim, scene, arm_entity_cfg, gripper_entity_cfg, diff_ik_controller, 
                                target_position, target_orientation, target_marker)
        target_marker.visualize(target_position, target_orientation)

    # Step 1.4: Close the gripper
    
    if count >= count_step[2] and count < count_step[3]:
        gripper_joint_pos_des = torch.full(
                (num_gripper_joints,), 0.0, device=sim.device
            )
        scene["robot"].set_joint_position_target(
                gripper_joint_pos_des, joint_ids=gripper_joint_ids
            )


    # Step 1.5: Move the arm to the target position above the gear and keep the orientation
    # target_position = target_position + torch.tensor([0.0, 0.0, 0.1 + TCP_offset], device=sim.device)
    # target_orientation = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=sim.device)
    if count >= count_step[3] and count < count_step[4]:
        move_robot_to_position(sim, scene, arm_entity_cfg, gripper_entity_cfg, diff_ik_controller, 
                                target_position_h, target_orientation, target_marker)
        gripper_joint_pos_des = torch.full(
                (num_gripper_joints,), 0.0, device=sim.device
            )
        scene["robot"].set_joint_position_target(
                gripper_joint_pos_des, joint_ids=gripper_joint_ids
            )
        target_marker.visualize(target_position_h, target_orientation)


def mount_gear_to_planetary_carrier(sim: sim_utils.SimulationContext, scene: InteractiveScene, 
                                 count: int,
                                 gear_id: int,
                                 count_step: torch.Tensor,
                                 gear_to_pin_map: dict,
                                 arm_entity_cfg: SceneEntityCfg,
                                 gripper_entity_cfg: SceneEntityCfg,
                                 diff_ik_controller: DifferentialIKController,
                                 initial_root_state: dict,
                                 target_marker: VisualizationMarkers):


    if gear_id == 5: # Place the carrier on the ring gear
        root_state = initial_root_state["ring_gear"]
        ring_gear_pos = root_state[:, :3].clone()
        ring_gear_quat = root_state[:, 3:7].clone()
        local_pos = torch.tensor([0.0, 0.054, 0.0], device=sim.device).unsqueeze(0)
        # local_pos = torch.tensor([0.054, 0.0, 0.0], device=sim.device).unsqueeze(0)

        target_orientation, target_position = torch_utils.tf_combine(
            ring_gear_quat, ring_gear_pos, 
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device), local_pos
        )
        # root_state = target_position
        # target_orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device)
        
    else: # Mount the gear on the planetary carrier
        # root_state = initial_root_state[f"sun_planetary_gear_{gear_id}"]

        planetary_carrier_pos = scene["planetary_carrier"].data.root_state_w[:, :3].clone()
        planetary_carrier_quat = scene["planetary_carrier"].data.root_state_w[:, 3:7].clone()
        original_planetary_carrier_pos = initial_root_state["planetary_carrier"][:, :3].clone()
        original_planetary_carrier_quat = initial_root_state["planetary_carrier"][:, 3:7].clone()

        # Local pose of the pin
        pin_local_pos = gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]['pin_local_pos'].clone()
        # Transfer the local pose of the pin to the world frame after the planetary carrier is moved
        # target_orientation = planetary_carrier_quat.clone()
        target_orientation, pin_world_pos = torch_utils.tf_combine(
            planetary_carrier_quat, planetary_carrier_pos, 
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device), pin_local_pos.unsqueeze(0)
        )
        _, original_pin_world_pos = torch_utils.tf_combine(
            original_planetary_carrier_quat, original_planetary_carrier_pos, 
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device), pin_local_pos.unsqueeze(0)
        )
        target_position = pin_world_pos.clone()
        # target_orientation = planetary_carrier_quat.clone()
    
    target_marker.visualize(target_position, target_orientation)


    target_position[:, 2] = table_height + grasping_height

    target_position += torch.tensor([TCP_offset_x, 0.0, TCP_offset_z], device=sim.device)
    
    target_position_h = target_position + torch.tensor([0.0, 0.0, 0.1], device=sim.device)
    target_orientation = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=sim.device)

    target_position_h_down = target_position + torch.tensor([0.0, 0.0, 0.02], device=sim.device)

    if count >= count_step[0] and count < count_step[1]:
        move_robot_to_position(sim, scene, arm_entity_cfg, gripper_entity_cfg, diff_ik_controller, 
                                target_position_h, target_orientation, target_marker)
        # target_marker.visualize(target_position_h, target_orientation)

    if count >= count_step[1] and count < count_step[2]:

        move_robot_to_position(sim, scene, arm_entity_cfg, gripper_entity_cfg, diff_ik_controller, 
                                target_position_h_down, target_orientation, target_marker)
        # target_marker.visualize(target_position_h_down, target_orientation)

    if count >= count_step[2] and count < count_step[3]:
        gripper_joint_ids = gripper_entity_cfg.joint_ids
        gripper_body_ids = gripper_entity_cfg.body_ids
        num_gripper_joints = len(gripper_joint_ids)

        gripper_joint_pos_des = torch.full(
                (num_gripper_joints,), 0.04, device=sim.device
            )

        if gear_id == 5:
            gripper_joint_pos_des = torch.full(
                (num_gripper_joints,), 0.017, device=sim.device
            )

        scene["robot"].set_joint_position_target(
                gripper_joint_pos_des, joint_ids=gripper_joint_ids
            )
        
    if count >= count_step[3] and count < count_step[4]:
        move_robot_to_position(sim, scene, arm_entity_cfg, gripper_entity_cfg, diff_ik_controller, 
                                target_position_h, target_orientation, target_marker)
        # target_marker.visualize(target_position_h, target_orientation)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    global left_gripper_position, right_gripper_position
    global target_position_left, target_position_right

    gripper_mat_cfg = physics_materials_cfg.RigidBodyMaterialCfg(
        static_friction=1.5,
        dynamic_friction=1.5,
        restitution=0.0,
        # (optional) combination modes if you need them:
        friction_combine_mode="average"
    )
    spawn_rigid_body_material("/World/Materials/gripper_material", gripper_mat_cfg)
    # mat_cfg.func("{ENV_REGEX_NS}/Robot/left_gripper_*", mat_cfg)
    # mat_cfg.func("{ENV_REGEX_NS}/Robot/right_gripper_*", mat_cfg)
    # sim_utils.bind_physics_material("/World/envs/env_0/Robot/left_gripper_link1/collisions", "/World/Materials/gripper_material")

    num_envs = scene.num_envs
    for env_idx in range(num_envs):
        sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/Robot/left_gripper_link1/collisions", "/World/Materials/gripper_material")
        sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/Robot/left_gripper_link2/collisions", "/World/Materials/gripper_material")  
        sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/Robot/right_gripper_link1/collisions", "/World/Materials/gripper_material")
        sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/Robot/right_gripper_link2/collisions", "/World/Materials/gripper_material")



    gear_mat_cfg = physics_materials_cfg.RigidBodyMaterialCfg(
        static_friction=0.1,
        dynamic_friction=0.1,
        restitution=0.0,
        friction_combine_mode="average"
    )
    spawn_rigid_body_material("/World/Materials/gear_material", gear_mat_cfg)
    for env_idx in range(num_envs):
        sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/ring_gear/node_/mesh_", "/World/Materials/gear_material")
        sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/sun_planetary_gear_1/node_/mesh_", "/World/Materials/gear_material")
        sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/sun_planetary_gear_2/node_/mesh_", "/World/Materials/gear_material")
        sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/sun_planetary_gear_3/node_/mesh_", "/World/Materials/gear_material")
        sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/sun_planetary_gear_4/node_/mesh_", "/World/Materials/gear_material")
        sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/planetary_carrier/node_/mesh_", "/World/Materials/gear_material")
        sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/planetary_reducer/node_/mesh_", "/World/Materials/gear_material")
    
    table_mat_cfg = physics_materials_cfg.RigidBodyMaterialCfg(
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.0,
        friction_combine_mode="average"
    )
    spawn_rigid_body_material("/World/Materials/table_material", table_mat_cfg)
    for env_idx in range(num_envs):
        sim_utils.bind_physics_material(f"/World/envs/env_{env_idx}/Table/table/body_whiteLarge", "/World/Materials/table_material")
    


    sim_dt = sim.get_physics_dt()
    print(f"sim_dt: {sim_dt}")
    count = 0

    # Time for intital stabilization
    time_step_0 = 1.0
    count_step_0 = int(time_step_0 / sim_dt)
    print(f"count_step_0: {count_step_0}")

    # Times for each step
    # 1. Move the arm to the target position above the gear and keep the orientation
    # 2. Move the arm to the target position and keep the orientation
    # 3. Close the gripper
    # 4. Move the arm to the target position above the gear and keep the orientation
    # time_step_1 = torch.tensor([0.0, 5.0, 1.0, 2.0, 1.0, 2.0], device=sim.device)
    time_step_1 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
    time_step_1 = torch.cumsum(time_step_1, dim=0) + time_step_0
    count_step_1 = time_step_1 / sim_dt
    count_step_1 = count_step_1.int()
    print(f"count_step_1: {count_step_1}")

    # Mount the gear to the planetary_carrier
    time_step_2 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
    time_step_2 = torch.cumsum(time_step_2, dim=0) + time_step_1[-1]
    count_step_2 = time_step_2 / sim_dt
    count_step_2 = count_step_2.int()
    print(f"count_step_2: {count_step_2}")

    # Pick up the 2nd gear
    time_step_3 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
    time_step_3 = torch.cumsum(time_step_3, dim=0) + time_step_2[-1]
    count_step_3 = time_step_3 / sim_dt
    count_step_3 = count_step_3.int()
    print(f"count_step_3: {count_step_3}")

    # Mount the 2nd gear to the planetary_carrier
    time_step_4 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
    time_step_4 = torch.cumsum(time_step_4, dim=0) + time_step_3[-1]
    count_step_4 = time_step_4 / sim_dt
    count_step_4 = count_step_4.int()
    print(f"count_step_4: {count_step_4}")

    # Pick up the 3rd gear
    time_step_5 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
    time_step_5 = torch.cumsum(time_step_5, dim=0) + time_step_4[-1]
    count_step_5 = time_step_5 / sim_dt
    count_step_5 = count_step_5.int()
    print(f"count_step_5: {count_step_5}")

    # Mount the 3rd gear to the planetary_carrier
    time_step_6 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
    time_step_6 = torch.cumsum(time_step_6, dim=0) + time_step_5[-1]
    count_step_6 = time_step_6 / sim_dt
    count_step_6 = count_step_6.int()
    print(f"count_step_6: {count_step_6}")

    # Reset right arm
    time_step_7 = torch.tensor([0.0, 0.5], device=sim.device)
    time_step_7 = torch.cumsum(time_step_7, dim=0) + time_step_6[-1]
    count_step_7 = time_step_7 / sim_dt
    count_step_7 = count_step_7.int()
    print(f"count_step_7: {count_step_7}")

    # Pick up the carrier
    time_step_8 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
    time_step_8 = torch.cumsum(time_step_8, dim=0) + time_step_7[-1]
    count_step_8 = time_step_8 / sim_dt
    count_step_8 = count_step_8.int()
    print(f"count_step_8: {count_step_8}")

    # Mount the carrier on the ring gear
    time_step_9 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
    time_step_9 = torch.cumsum(time_step_9, dim=0) + time_step_8[-1]
    count_step_9 = time_step_9 / sim_dt
    count_step_9 = count_step_9.int()
    print(f"count_step_9: {count_step_9}")


    left_init_pos = [0.3864, 0.5237, 1.1475]
    left_init_rot = [0.0, -1.0, 0.0, 0.0]
    right_init_pos = [0.3864, -0.5237, 1.1475]
    right_init_rot = [0.0, -1.0, 0.0, 0.0]

    initial_root_state = randomize_object_positions(sim, scene, ['ring_gear', 'planetary_carrier',
                                        'sun_planetary_gear_1', 'sun_planetary_gear_2',
                                        'sun_planetary_gear_3', 'sun_planetary_gear_4',
                                        'planetary_reducer'])
    
    # initial_root_state["planetary_carrier"] = torch.tensor([[0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]], device=sim.device)

    """Runs the simulation loop."""
    robot = scene["robot"]

    diff_ik_controller, left_arm_entity_cfg, left_gripper_entity_cfg = get_config(sim, scene, "left")
    diff_ik_controller, right_arm_entity_cfg, right_gripper_entity_cfg = get_config(sim, scene, "right")

    # get left/right joint ids
    left_arm_joint_ids = left_arm_entity_cfg.joint_ids
    right_arm_joint_ids = right_arm_entity_cfg.joint_ids
    num_arm_joints = len(left_arm_joint_ids)

    print("robot.is_fixed_base: ", robot.is_fixed_base)
    if robot.is_fixed_base:
        left_ee_jacobi_idx = left_arm_entity_cfg.body_ids[0] - 1
        right_ee_jacobi_idx = right_arm_entity_cfg.body_ids[0] - 1
    else:
        left_ee_jacobi_idx = left_arm_entity_cfg.body_ids[0]
        right_ee_jacobi_idx = right_arm_entity_cfg.body_ids[0]

    left_gripper_joint_ids = left_gripper_entity_cfg.joint_ids
    right_gripper_joint_ids = right_gripper_entity_cfg.joint_ids
    num_gripper_joints = len(left_gripper_joint_ids)
    if num_gripper_joints != len(right_gripper_joint_ids):
        raise ValueError(
            "The number of left and right gripper joints should be the same."
        )

    print("-------------------------------------------------")
    print("left body_ids: ", left_arm_entity_cfg.body_ids)
    print("left joint_ids: ", left_arm_joint_ids)
    print("left_ee_jacobi_idx: ", left_ee_jacobi_idx)
    print("right body_ids: ", right_arm_entity_cfg.body_ids)
    print("right joint_ids: ", right_arm_joint_ids)
    print("right_ee_jacobi_idx: ", right_ee_jacobi_idx)
    print("num_arm_joints: ", num_arm_joints)
    print("*************************************************")
    print("left gripper joint_ids: ", left_gripper_joint_ids)
    print("right gripper joint_ids: ", right_gripper_joint_ids)
    print("num gripper joints: ", num_gripper_joints)
    print("-------------------------------------------------")


    target_position_left = torch.tensor(left_init_pos, device=sim.device)
    target_orientation_left = torch.tensor(left_init_rot, device=sim.device)
    target_position_right = torch.tensor(right_init_pos, device=sim.device)
    target_orientation_right = torch.tensor(right_init_rot, device=sim.device)


    # Prepare markers
    # frame_marker_cfg = FRAME_MARKER_CFG.copy()
    # frame_marker_cfg.markers["frame"].scale = (0.02, 0.02, 0.02)
    # pin_1_pos_marker = VisualizationMarkers(
    #         frame_marker_cfg.replace(prim_path="/Visuals/pin_1_pos")
    # )

    # pin_2_pos_marker = VisualizationMarkers(
    #         frame_marker_cfg.replace(prim_path="/Visuals/pin_2_pos")
    # )

    # pin_3_pos_marker = VisualizationMarkers(
    #         frame_marker_cfg.replace(prim_path="/Visuals/pin_3_pos")
    # )

    target_marker_cfg = FRAME_MARKER_CFG.copy()
    target_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    target_marker = VisualizationMarkers(
            target_marker_cfg.replace(prim_path="/Visuals/target_marker")
    )


    gear_to_pin_map = prepare_mounting_plan(sim, scene, left_arm_entity_cfg, 
                                            right_arm_entity_cfg, initial_root_state)
    
    
    while simulation_app.is_running():

        # print(f"count: {count}")
        print(f"Time: {count*sim_dt}")


        if count < count_step_0:
            # left arm
            move_robot_to_position(sim, scene, left_arm_entity_cfg, left_gripper_entity_cfg, diff_ik_controller, 
                                    target_position_left, target_orientation_left, target_marker)
            # right arm
            move_robot_to_position(sim, scene, right_arm_entity_cfg, right_gripper_entity_cfg, diff_ik_controller, 
                                    target_position_right, target_orientation_right, target_marker)
            
            gripper_joint_pos_des = torch.full(
                (num_gripper_joints,), 0.04, device=sim.device
            )

            scene["robot"].set_joint_position_target(
                gripper_joint_pos_des, joint_ids=right_gripper_joint_ids
            )
            scene["robot"].set_joint_position_target(
                gripper_joint_pos_des, joint_ids=left_gripper_joint_ids
            )


            # print("scene['robot'].data.joint_pos[:, left_gripper_joint_ids]: ", scene["robot"].data.joint_pos[:, left_gripper_joint_ids])
            # print("scene['robot'].data.joint_pos[:, right_gripper_joint_ids]: ", scene["robot"].data.joint_pos[:, right_gripper_joint_ids])

            # print("left_gripper_joint_ids+right_gripper_joint_ids: ", left_gripper_joint_ids+right_gripper_joint_ids)
        
        # Only for testing purpose
        # if count >= count_step_1[0] and count < count_step_1[-1]:
        #     gear_id = 5
        #     pick_up_sun_planetary_gear(sim, scene, 
        #                                 count, gear_id,
        #                                 gear_to_pin_map,
        #                                 count_step_1,
        #                                 left_arm_entity_cfg, 
        #                                 left_gripper_entity_cfg, 
        #                                 diff_ik_controller, 
        #                                 initial_root_state,
        #                                 target_marker)
        # if count >= count_step_2[0] and count < count_step_2[-1]:
        #     gear_id = 5
        #     mount_gear_to_planetary_carrier(sim, scene, 
        #                                 count, gear_id,
        #                                 count_step_2,
        #                                 gear_to_pin_map,
        #                                 left_arm_entity_cfg, 
        #                                 left_gripper_entity_cfg, 
        #                                 diff_ik_controller, initial_root_state, target_marker)


        # Pick up the 1st gear
        if count >= count_step_1[0] and count < count_step_1[-1]:
        
            gear_id = 1
            current_arm_str = gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]['arm']
            if current_arm_str == 'left':
                current_arm = left_arm_entity_cfg
                current_gripper = left_gripper_entity_cfg
            else:
                current_arm = right_arm_entity_cfg
                current_gripper = right_gripper_entity_cfg

            pick_up_sun_planetary_gear(sim, scene, 
                                         count, gear_id,
                                         gear_to_pin_map,
                                         count_step_1,
                                         current_arm, 
                                         current_gripper, 
                                         diff_ik_controller, 
                                         initial_root_state,
                                         target_marker)

        # Mount the 1st gear to the planetary_carrier
        if count >= count_step_2[0] and count < count_step_2[-1]:
            gear_id = 1
            mount_gear_to_planetary_carrier(sim, scene, 
                                         count, gear_id,
                                         count_step_2,
                                         gear_to_pin_map,
                                         current_arm, current_gripper, 
                                         diff_ik_controller, initial_root_state,
                                         target_marker)
            
        # Only for testing purpose
        if count >= count_step_3[0] and count < count_step_3[-1]:
            current_arm = left_arm_entity_cfg
            current_gripper = left_gripper_entity_cfg
            gear_id = 5
            pick_up_sun_planetary_gear(sim, scene, 
                                         count, gear_id,
                                         gear_to_pin_map,
                                         count_step_3,
                                         current_arm, current_gripper, 
                                         diff_ik_controller, initial_root_state,
                                         target_marker)

        # Mount the carrier on the ring gear
        if count >= count_step_4[0] and count < count_step_4[-1]:
            current_arm = left_arm_entity_cfg
            current_gripper = left_gripper_entity_cfg
            gear_id = 5
            mount_gear_to_planetary_carrier(sim, scene, 
                                         count, gear_id,
                                         count_step_4,
                                         gear_to_pin_map,
                                         current_arm, current_gripper, 
                                         diff_ik_controller, initial_root_state, target_marker)
        
        # Pick up the 2nd gear
        # if count >= count_step_3[0] and count < count_step_3[-1]:
        #     gear_id = 2
        #     current_arm_str = gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]['arm']
        #     if current_arm_str == 'left':
        #         current_arm = left_arm_entity_cfg
        #         current_gripper = left_gripper_entity_cfg
        #     else:
        #         current_arm = right_arm_entity_cfg
        #         current_gripper = right_gripper_entity_cfg

        #     pick_up_sun_planetary_gear(sim, scene, 
        #                                  count, gear_id,
        #                                  gear_to_pin_map,
        #                                  count_step_3,
        #                                  current_arm, current_gripper, 
        #                                  diff_ik_controller, 
        #                                  initial_root_state,
        #                                  target_marker)

        # # Mount the 2nd gear to the planetary_carrier
        # if count >= count_step_4[0] and count < count_step_4[-1]:
        #     gear_id = 2
        #     mount_gear_to_planetary_carrier(sim, scene, 
        #                                  count, gear_id,
        #                                  count_step_4,
        #                                  gear_to_pin_map,
        #                                  current_arm, current_gripper, 
        #                                  diff_ik_controller, initial_root_state,
        #                                  target_marker)
            
        # # Pick up the 3rd gear
        # if count >= count_step_5[0] and count < count_step_5[-1]:
        #     gear_id = 3
        #     current_arm_str = gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]['arm']
        #     if current_arm_str == 'left':
        #         current_arm = left_arm_entity_cfg
        #         current_gripper = left_gripper_entity_cfg
        #     else:
        #         current_arm = right_arm_entity_cfg
        #         current_gripper = right_gripper_entity_cfg
        #     pick_up_sun_planetary_gear(sim, scene, 
        #                                  count, gear_id,
        #                                  gear_to_pin_map,
        #                                  count_step_5,
        #                                  current_arm, current_gripper, 
        #                                  diff_ik_controller, 
        #                                  initial_root_state,
        #                                  target_marker)
        # # Mount the 3rd gear to the planetary_carrier
        # if count >= count_step_6[0] and count < count_step_6[-1]:
        #     gear_id = 3
        #     mount_gear_to_planetary_carrier(sim, scene, 
        #                                  count, gear_id,
        #                                  count_step_6,
        #                                  gear_to_pin_map,
        #                                  current_arm, current_gripper, 
        #                                  diff_ik_controller, initial_root_state,
        #                                  target_marker)

        # # Reset right arm
        # if count >= count_step_7[0] and count < count_step_7[-1]:
        #     move_robot_to_position(sim, scene, right_arm_entity_cfg, right_gripper_entity_cfg, diff_ik_controller, 
        #                             target_position_right, target_orientation_right, target_marker)
            
        # # Pick up the carrier
        # if count >= count_step_8[0] and count < count_step_8[-1]:
        #     current_arm = left_arm_entity_cfg
        #     current_gripper = left_gripper_entity_cfg
        #     gear_id = 5
        #     pick_up_sun_planetary_gear(sim, scene, 
        #                                  count, gear_id,
        #                                  gear_to_pin_map,
        #                                  count_step_8,
        #                                  current_arm, current_gripper, 
        #                                  diff_ik_controller, initial_root_state,
        #                                  target_marker)

        # # Mount the carrier on the ring gear
        # if count >= count_step_9[0] and count < count_step_9[-1]:
        #     current_arm = left_arm_entity_cfg
        #     current_gripper = left_gripper_entity_cfg
        #     gear_id = 5
        #     mount_gear_to_planetary_carrier(sim, scene, 
        #                                  count, gear_id,
        #                                  count_step_9,
        #                                  gear_to_pin_map,
        #                                  current_arm, current_gripper, 
        #                                  diff_ik_controller, initial_root_state, target_marker)

        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        # left_ee_pose_w = robot.data.body_state_w[
        #     :, left_arm_entity_cfg.body_ids[0], 0:7
        # ]
        # right_ee_pose_w = robot.data.body_state_w[
        #     :, right_arm_entity_cfg.body_ids[0], 0:7
        # ]

        


def main():
    app_window = omni.appwindow.get_default_app_window()
    # keyboard = app_window.get_keyboard()
    # input_interface = carb.input.acquire_input_interface()

    # subscribe (stores subscription id)
    # sub_id = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    sim_cfg = sim_utils.SimulationCfg()
    sim_cfg.dt = 0.01
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    scene_cfg = R1SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
