import torch
import math
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

from isaaclab.sim import SimulationContext

class RecoveryRulePolicy:
    def __init__(self, sim: sim_utils.SimulationContext, scene: InteractiveScene,
    obj_dict: dict, initial_assembly_state: str = "default"):
        self.sim = sim
        self.scene = scene
        # self.arm_name = arm_name
        self.device = sim.device
        self.initial_assembly_state = initial_assembly_state

        self.obj_dict = obj_dict

        self.planetary_carrier = obj_dict["planetary_carrier"]
        self.ring_gear = obj_dict["ring_gear"]
        self.sun_planetary_gear_1 = obj_dict["sun_planetary_gear_1"]
        self.sun_planetary_gear_2 = obj_dict["sun_planetary_gear_2"]
        self.sun_planetary_gear_3 = obj_dict["sun_planetary_gear_3"]
        self.sun_planetary_gear_4 = obj_dict["sun_planetary_gear_4"]
        self.planetary_reducer = obj_dict["planetary_reducer"]

        # Define pin positions in local coordinates (relative to planetary carrier)
        self.pin_local_positions = [
            torch.tensor([0.0, -0.054, 0.0], device=self.device),      # pin_0
            torch.tensor([0.0465, 0.0268, 0.0], device=self.device),   # pin_1
            torch.tensor([-0.0465, 0.0268, 0.0], device=self.device),  # pin_2
        ]

        self.TCP_offset_z = 1.1475 - 1.05661
        self.TCP_offset_x = 0.3864 - 0.3785
        self.table_height = 0.9
        self.grasping_height = -0.003
        self.lifting_height = 0.2

        self.diff_ik_controller, self.left_arm_entity_cfg, self.left_gripper_entity_cfg = self.get_config("left")
        self.diff_ik_controller, self.right_arm_entity_cfg, self.right_gripper_entity_cfg = self.get_config("right")

        self.right_gripper_joint_ids = self.right_gripper_entity_cfg.joint_ids
        self.left_gripper_joint_ids = self.left_gripper_entity_cfg.joint_ids

        # self.target_position_left = torch.tensor([0.3864, 0.5237, 1.1475], device=self.device)
        # self.target_orientation_left = torch.tensor([0.0, -1.0, 0.0, 0.0], device=self.device)
        # self.target_position_right = torch.tensor([0.3864, -0.5237, 1.1475], device=self.device)
        # self.target_orientation_right = torch.tensor([0.0, -1.0, 0.0, 0.0], device=self.device)

        self.initial_pos_left = torch.tensor([-20.0 / 180.0 * math.pi, 100.6 / 180.0 * math.pi,
                                         -24.0 / 180.0 * math.pi, 17.8 / 180.0 * math.pi,
                                         38.7 / 180.0 * math.pi, 20.1 / 180.0 * math.pi], device=self.device)
        self.initial_pos_right = torch.tensor([-20.0 / 180.0 * math.pi, 100.6 / 180.0 * math.pi,
                                         -22.0 / 180.0 * math.pi, -40.0 / 180.0 * math.pi,
                                         -67.6 / 180.0 * math.pi, 18.1 / 180.0 * math.pi], device=self.device)

        self.num_gripper_joints = None

        self.gear_to_pin_map = None
        
        self.current_target_position = None
        self.current_target_orientation = None
        
        self.current_target_joint_pos = None
        self.step_initial_joint_pos = None


        self.sim_dt = sim.get_physics_dt()
        print(f"sim_dt: {self.sim_dt}")
        self.count = 0

        # Time for intital stabilization
        self.time_step_0 = 0.2
        self.count_step_0 = int(self.time_step_0 / self.sim_dt)
        print(f"count_step_0: {self.count_step_0}")

        # Times for each step
        # 1. Move the arm to the target position above the gear and keep the orientation
        # 2. Move the arm to the target position and keep the orientation
        # 3. Close the gripper
        # 4. Move the arm to the target position above the gear and keep the orientation
        # time_step_1 = torch.tensor([0.0, 5.0, 1.0, 2.0, 1.0, 2.0], device=sim.device)
        self.time_step_1 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_1 = torch.cumsum(self.time_step_1, dim=0) + self.time_step_0
        self.count_step_1 = self.time_step_1 / self.sim_dt
        self.count_step_1 = self.count_step_1.int()
        print(f"count_step_1: {self.count_step_1}")

        # Mount the gear to the planetary_carrier
        self.time_step_2 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_2 = torch.cumsum(self.time_step_2, dim=0) + self.time_step_1[-1]
        self.count_step_2 = self.time_step_2 / self.sim_dt
        self.count_step_2 = self.count_step_2.int()
        print(f"count_step_2: {self.count_step_2}")

        # Pick up the 2nd gear
        self.time_step_3 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_3 = torch.cumsum(self.time_step_3, dim=0) + self.time_step_2[-1]
        self.count_step_3 = self.time_step_3 / self.sim_dt
        self.count_step_3 = self.count_step_3.int()
        print(f"count_step_3: {self.count_step_3}")

        # Mount the 2nd gear to the planetary_carrier
        self.time_step_4 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_4 = torch.cumsum(self.time_step_4, dim=0) + self.time_step_3[-1]
        self.count_step_4 = self.time_step_4 / self.sim_dt
        self.count_step_4 = self.count_step_4.int()
        print(f"count_step_4: {self.count_step_4}")

        # Reset left arm
        self.time_step_5 = torch.tensor([0.0, 0.5], device=sim.device)
        self.time_step_5 = torch.cumsum(self.time_step_5, dim=0) + self.time_step_4[-1]
        self.count_step_5 = self.time_step_5 / self.sim_dt
        self.count_step_5 = self.count_step_5.int()
        print(f"count_step_5: {self.count_step_5}")

        # Pick up the 3rd gear
        self.time_step_6 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_6 = torch.cumsum(self.time_step_6, dim=0) + self.time_step_5[-1]
        self.count_step_6 = self.time_step_6 / self.sim_dt
        self.count_step_6 = self.count_step_6.int()
        print(f"count_step_6: {self.count_step_6}")

        # Mount the 3rd gear to the planetary_carrier
        self.time_step_7 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_7 = torch.cumsum(self.time_step_7, dim=0) + self.time_step_6[-1]
        self.count_step_7 = self.time_step_7 / self.sim_dt
        self.count_step_7 = self.count_step_7.int()
        print(f"count_step_7: {self.count_step_7}")

        # Pick up the 4th gear
        self.time_step_8 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_8 = torch.cumsum(self.time_step_8, dim=0) + self.time_step_7[-1]
        self.count_step_8 = self.time_step_8 / self.sim_dt
        self.count_step_8 = self.count_step_8.int()
        print(f"count_step_8: {self.count_step_8}")

        # Mount the 4th gear to the planetary_carrier. 
        # Another rotation is performed to aid the insertion
        self.time_step_9 = torch.tensor([0.0, 0.5, 0.5, 5.0, 0.5, 0.5], device=sim.device)
        self.time_step_9 = torch.cumsum(self.time_step_9, dim=0) + self.time_step_8[-1]
        self.count_step_9 = self.time_step_9 / self.sim_dt
        self.count_step_9 = self.count_step_9.int()
        print(f"count_step_9: {self.count_step_9}")

        # Reset right arm
        self.time_step_10 = torch.tensor([0.0, 0.5], device=sim.device)
        self.time_step_10 = torch.cumsum(self.time_step_10, dim=0) + self.time_step_9[-1]
        self.count_step_10 = self.time_step_10 / self.sim_dt
        self.count_step_10 = self.count_step_10.int()
        print(f"count_step_10: {self.count_step_10}")

        # Pick up the big ring gear
        self.time_step_11 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_11 = torch.cumsum(self.time_step_11, dim=0) + self.time_step_10[-1]
        self.count_step_11 = self.time_step_11 / self.sim_dt
        self.count_step_11 = self.count_step_11.int()
        print(f"count_step_11: {self.count_step_11}")

        # Mount the ring on the carrier
        self.time_step_12 = torch.tensor([0.0, 0.5, 0.5, 3.0, 0.5, 0.5], device=sim.device)
        self.time_step_12 = torch.cumsum(self.time_step_12, dim=0) + self.time_step_11[-1]
        self.count_step_12 = self.time_step_12 / self.sim_dt
        self.count_step_12 = self.count_step_12.int()
        print(f"count_step_12: {self.count_step_12}")
        

        # Pick up the reducer
        self.time_step_13 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_13 = torch.cumsum(self.time_step_13, dim=0) + self.time_step_12[-1]
        self.count_step_13 = self.time_step_13 / self.sim_dt
        self.count_step_13 = self.count_step_13.int()
        print(f"count_step_13: {self.count_step_13}")

        # Mount the reducer to the gear
        self.time_step_14 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_14 = torch.cumsum(self.time_step_14, dim=0) + self.time_step_13[-1]
        self.count_step_14 = self.time_step_14 / self.sim_dt
        self.count_step_14 = self.count_step_14.int()
        print(f"count_step_14: {self.count_step_14}")

        # Special time steps for misplaced_fourth_gear state
        # Step 1: Pick up the misplaced 4th gear from on top of other gears
        self.time_step_misplaced_1 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_misplaced_1 = torch.cumsum(self.time_step_misplaced_1, dim=0) + self.time_step_0
        self.count_step_misplaced_1 = self.time_step_misplaced_1 / self.sim_dt
        self.count_step_misplaced_1 = self.count_step_misplaced_1.int()
        print(f"count_step_misplaced_1: {self.count_step_misplaced_1}")

        # Step 2: Mount the 4th gear to the carrier
        self.time_step_misplaced_2 = torch.tensor([0.0, 0.5, 0.5, 5.0, 0.5, 0.5], device=sim.device)
        self.time_step_misplaced_2 = torch.cumsum(self.time_step_misplaced_2, dim=0) + self.time_step_misplaced_1[-1]
        self.count_step_misplaced_2 = self.time_step_misplaced_2 / self.sim_dt
        self.count_step_misplaced_2 = self.count_step_misplaced_2.int()
        print(f"count_step_misplaced_2: {self.count_step_misplaced_2}")

        # Step 3: Reset right arm
        self.time_step_misplaced_3 = torch.tensor([0.0, 0.5], device=sim.device)
        self.time_step_misplaced_3 = torch.cumsum(self.time_step_misplaced_3, dim=0) + self.time_step_misplaced_2[-1]
        self.count_step_misplaced_3 = self.time_step_misplaced_3 / self.sim_dt
        self.count_step_misplaced_3 = self.count_step_misplaced_3.int()
        print(f"count_step_misplaced_3: {self.count_step_misplaced_3}")

        # Step 4: Pick up the ring gear
        self.time_step_misplaced_4 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
        self.time_step_misplaced_4 = torch.cumsum(self.time_step_misplaced_4, dim=0) + self.time_step_misplaced_3[-1]
        self.count_step_misplaced_4 = self.time_step_misplaced_4 / self.sim_dt
        self.count_step_misplaced_4 = self.count_step_misplaced_4.int()
        print(f"count_step_misplaced_4: {self.count_step_misplaced_4}")

        # Step 5: Mount the ring gear on the carrier
        self.time_step_misplaced_5 = torch.tensor([0.0, 0.5, 0.5, 3.0, 0.5, 0.5], device=sim.device)
        self.time_step_misplaced_5 = torch.cumsum(self.time_step_misplaced_5, dim=0) + self.time_step_misplaced_4[-1]
        self.count_step_misplaced_5 = self.time_step_misplaced_5 / self.sim_dt
        self.count_step_misplaced_5 = self.count_step_misplaced_5.int()
        print(f"count_step_misplaced_5: {self.count_step_misplaced_5}")

        left_init_pos = [0.3864, 0.5237, 1.1475]
        left_init_rot = [0.0, -1.0, 0.0, 0.0]
        right_init_pos = [0.3864, -0.5237, 1.1475]
        right_init_rot = [0.0, -1.0, 0.0, 0.0]

        self.initial_root_state = None

        # Adjust time steps for lack_fourth_gear state (skip steps 1-7)
        if self.initial_assembly_state == "lack_fourth_gear":
            # Recalculate steps 8-14 to start immediately after step 0
            self.time_step_8 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
            self.time_step_8 = torch.cumsum(self.time_step_8, dim=0) + self.time_step_0
            self.count_step_8 = self.time_step_8 / self.sim_dt
            self.count_step_8 = self.count_step_8.int()
            
            self.time_step_9 = torch.tensor([0.0, 0.5, 0.5, 5.0, 0.5, 0.5], device=sim.device)
            self.time_step_9 = torch.cumsum(self.time_step_9, dim=0) + self.time_step_8[-1]
            self.count_step_9 = self.time_step_9 / self.sim_dt
            self.count_step_9 = self.count_step_9.int()
            
            self.time_step_10 = torch.tensor([0.0, 0.5], device=sim.device)
            self.time_step_10 = torch.cumsum(self.time_step_10, dim=0) + self.time_step_9[-1]
            self.count_step_10 = self.time_step_10 / self.sim_dt
            self.count_step_10 = self.count_step_10.int()
            
            self.time_step_11 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
            self.time_step_11 = torch.cumsum(self.time_step_11, dim=0) + self.time_step_10[-1]
            self.count_step_11 = self.time_step_11 / self.sim_dt
            self.count_step_11 = self.count_step_11.int()
            
            self.time_step_12 = torch.tensor([0.0, 0.5, 0.5, 3.0, 0.5, 0.5], device=sim.device)
            self.time_step_12 = torch.cumsum(self.time_step_12, dim=0) + self.time_step_11[-1]
            self.count_step_12 = self.time_step_12 / self.sim_dt
            self.count_step_12 = self.count_step_12.int()
            
            self.time_step_13 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
            self.time_step_13 = torch.cumsum(self.time_step_13, dim=0) + self.time_step_12[-1]
            self.count_step_13 = self.time_step_13 / self.sim_dt
            self.count_step_13 = self.count_step_13.int()
            
            self.time_step_14 = torch.tensor([0.0, 0.5, 0.5, 0.5, 0.5], device=sim.device)
            self.time_step_14 = torch.cumsum(self.time_step_14, dim=0) + self.time_step_13[-1]
            self.count_step_14 = self.time_step_14 / self.sim_dt
            self.count_step_14 = self.count_step_14.int()
            
            print(f"[lack_fourth_gear] Recalculated time steps:")
            print(f"  count_step_8: {self.count_step_8}")
            print(f"  count_step_9: {self.count_step_9}")
            print(f"  count_step_10: {self.count_step_10}")
            print(f"  count_step_11: {self.count_step_11}")
            print(f"  count_step_12: {self.count_step_12}")
            print(f"  count_step_13: {self.count_step_13}")
            print(f"  count_step_14: {self.count_step_14}")

        if self.initial_assembly_state == "misplaced_fourth_gear":
            self.total_time_steps = self.count_step_misplaced_5[-1]
        else:
            self.total_time_steps = self.count_step_14[-1]

    def set_initial_root_state(self, initial_root_state: dict):
        self.initial_root_state = initial_root_state.copy()


    def get_config(self, arm_name: str):
        # arm_name: left or right

        # Create controller
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        )
        diff_ik_controller = DifferentialIKController(
            diff_ik_cfg, num_envs=self.scene.num_envs, device=self.sim.device
        )

        # Specify robot-specific parameters
        arm_entity_cfg = SceneEntityCfg(
            "robot", joint_names=[f"{arm_name}_arm_joint.*"], body_names=[f"{arm_name}_arm_link6"]
        )
        gripper_entity_cfg = SceneEntityCfg(
            "robot", joint_names=[f"{arm_name}_gripper_axis1"]
        )

        # Resolving the scene entities
        arm_entity_cfg.resolve(self.scene)
        gripper_entity_cfg.resolve(self.scene)
        
        # gripper_entity_cfg = SceneEntityCfg("robot", joint_names=[f"{arm_name}_gripper_.*"], body_names=[f"{arm_name}_gripper_link1"])
        # gripper_entity_cfg.resolve(self.scene)
        
        return diff_ik_controller, arm_entity_cfg, gripper_entity_cfg
        


    def move_robot_to_position(self,
                            arm_entity_cfg: SceneEntityCfg,
                            gripper_entity_cfg: SceneEntityCfg,
                            diff_ik_controller: DifferentialIKController,
                            target_position: torch.Tensor, target_orientation: torch.Tensor,
                            target_marker: VisualizationMarkers):
        robot = self.scene["robot"]

        arm_joint_ids = arm_entity_cfg.joint_ids
        arm_body_ids = arm_entity_cfg.body_ids
        num_arm_joints = len(arm_joint_ids)

        gripper_joint_ids = gripper_entity_cfg.joint_ids
        gripper_body_ids = gripper_entity_cfg.body_ids
        self.num_gripper_joints = len(gripper_joint_ids)

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

        # print(f"ee_pos_b: {ee_pos_b}, ee_quat_b: {ee_quat_b}")
        # print(f"joint_pos_des: {joint_pos_des}")

        # Apply
        # print(f"joint_pos_des: {joint_pos_des}")
        # robot.set_joint_position_target(
        #     joint_pos_des, joint_ids=arm_entity_cfg.joint_ids
        # )

        return joint_pos_des, arm_entity_cfg.joint_ids
        


    def prepare_mounting_plan(self,
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
            gear_to_pin_map: Dictionary mapping gear_name -> {'arm': 'left'/'right', 'pin': pin_index,
                                                                'pin_world_pos': tensor, 'pin_world_quat': tensor}
        """

        # Default to all 4 gears if not specified
        if gear_names is None:
            gear_names = ['sun_planetary_gear_1', 'sun_planetary_gear_2',
                        'sun_planetary_gear_3', 'sun_planetary_gear_4',
                        'ring_gear', 'planetary_reducer']

        # Get the planetary carrier positions and orientations
        root_state = self.initial_root_state["planetary_carrier"]
        planetary_carrier_pos = root_state[:, :3].clone()
        planetary_carrier_quat = root_state[:, 3:7].clone()
        num_envs = planetary_carrier_pos.shape[0]


        # Calculate world positions of all pins
        pin_world_positions = []
        pin_world_quats = []
        for pin_local_pos in self.pin_local_positions:
            pin_local_pos_batch = pin_local_pos.unsqueeze(0).expand(num_envs, -1)
            pin_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(num_envs, -1)

            pin_world_pos = torch_utils.tf_combine(
                planetary_carrier_quat, planetary_carrier_pos, pin_quat, pin_local_pos_batch)[1]

            pin_world_positions.append(pin_world_pos)
            pin_world_quats.append(pin_quat)

        # Stack all pin positions: shape (num_envs, num_pins, 3)
        pin_world_positions = torch.stack(pin_world_positions, dim=1)
        pin_world_quats = torch.stack(pin_world_quats, dim=1)

        # Get end-effector positions for both arms
        left_ee_pos = self.scene["robot"].data.body_state_w[:, self.left_arm_entity_cfg.body_ids[0], 0:3]
        right_ee_pos = self.scene["robot"].data.body_state_w[:, self.right_arm_entity_cfg.body_ids[0], 0:3]

        # Track which pins are occupied
        occupied_pins = set()

        # Result mapping
        self.gear_to_pin_map = {}

        # For each gear, find nearest arm and nearest available pin
        for gear_name in gear_names:
            if gear_name not in self.initial_root_state:
                print(f"[WARN] Gear {gear_name} not found in initial_root_state, skipping")
                continue

            # Get gear position
            gear_pos = self.initial_root_state[gear_name][:, :3].clone()  # shape: (num_envs, 3)

            # For the first environment (env_idx=0), calculate distances to both arms
            env_idx = 0
            gear_pos_env = gear_pos[env_idx]

            # Calculate distance to both arms
            left_dist = torch.norm(gear_pos_env - left_ee_pos[env_idx])
            right_dist = torch.norm(gear_pos_env - right_ee_pos[env_idx])

            # Choose the nearest arm
            # if left_dist < right_dist:
            #     chosen_arm = 'left'
            #     chosen_arm_pos = left_ee_pos[env_idx]
            # else:
            #     chosen_arm = 'right'
            #     chosen_arm_pos = right_ee_pos[env_idx]
            if gear_pos_env[1] > 0.0:
                chosen_arm = 'left'
                chosen_arm_pos = left_ee_pos[env_idx]
            else:
                chosen_arm = 'right'
                chosen_arm_pos = right_ee_pos[env_idx]
                
            if gear_name == 'ring_gear' or gear_name == 'planetary_reducer':
                self.gear_to_pin_map[gear_name] = {
                    'arm': chosen_arm,
                    'pin': None,
                    'pin_local_pos': None,
                    'pin_world_pos': None, 
                    'pin_world_quat': None,
                }
                continue

            # Find the nearest available pin
            min_pin_dist = float('inf')
            nearest_pin_idx = None

            for pin_idx in range(len(self.pin_local_positions)):
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
            self.gear_to_pin_map[gear_name] = {
                'arm': chosen_arm,
                'pin': nearest_pin_idx,
                'pin_local_pos': self.pin_local_positions[nearest_pin_idx],
                'pin_world_pos': pin_world_positions[:, nearest_pin_idx],  # All environments
                'pin_world_quat': pin_world_quats[:, nearest_pin_idx],
            }

            print(f"[INFO] {gear_name} -> {chosen_arm} arm, pin_{nearest_pin_idx}")
            print(f"       Gear pos: {gear_pos_env}, Pin pos: {pin_world_positions[env_idx, nearest_pin_idx]}")
            print(f"       Distance: {min_pin_dist:.4f}m")


        print(f"self.gear_to_pin_map: {self.gear_to_pin_map}")

        return self.gear_to_pin_map


    # Step 1: Pick up the sun_planetary_gear_1
    # grasping_state = 1
    # joint_pos = None
    def pick_up_target_gear(self,
                                    gear_id: int,
                                    count_step: torch.Tensor,
                                    arm_entity_cfg: SceneEntityCfg,
                                    gripper_entity_cfg: SceneEntityCfg,
                                    diff_ik_controller: DifferentialIKController):
        
        sim_dt = self.sim.get_physics_dt()

        obj_height_offset = 0.0

        if gear_id == 4:
            obj_height_offset = 0.01

        if gear_id == 5:
            # planetary_carrier_pos = self.planetary_carrier.data.root_state_w[:, :3].clone()
            # planetary_carrier_quat = self.planetary_carrier.data.root_state_w[:, 3:7].clone()

            ring_gear_pos = self.initial_root_state["ring_gear"][:, :3].clone()
            ring_gear_quat = self.initial_root_state["ring_gear"][:, 3:7].clone()

            # local_pos = torch.tensor([0.0, 0.054, 0], device=self.sim.device).unsqueeze(0)
            local_pos = torch.tensor([0.0, 0.0, 0.0], device=self.sim.device).unsqueeze(0)

            target_orientation, target_position = torch_utils.tf_combine(
                ring_gear_quat, ring_gear_pos, 
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.sim.device), local_pos
            )
            root_state = torch.cat([target_position, target_orientation], dim=-1)
            obj_height_offset = 0.030

        elif gear_id == 6: # Reducer
            root_state = self.initial_root_state["planetary_reducer"]
            # target_position = root_state[:, :3].clone()
            obj_height_offset = 0.05

        else:
            root_state = self.initial_root_state[f"sun_planetary_gear_{gear_id}"]
        # print(f"obj: {obj}")
        # target_position, target_orientation = target_frame.get_local_pose()
        # target_position, target_orientation = target_frame.get_world_poses()
        target_position = root_state[:, :3].clone()
        
        # For misplaced_fourth_gear state, preserve actual Z position of stacked gear
        if self.initial_assembly_state == "misplaced_fourth_gear" and gear_id == 4:
            target_position[:, 2] = root_state[:, 2] + obj_height_offset - 0.01
        else:
            target_position[:, 2] = self.table_height + self.grasping_height + obj_height_offset
            
        target_position = target_position + torch.tensor([self.TCP_offset_x, 0.0, self.TCP_offset_z], device=self.sim.device)
        
        # target_orientation = obj.data.default_root_state[:, 3:7].clone()
        # print(f"target_position: {target_position}, target_orientation: {target_orientation}")
        # Step 1.1: Move the arm to the target position above the gear and keep the orientation
        target_position_h = target_position + torch.tensor([0.0, 0.0, self.lifting_height], device=self.sim.device)
        
        # target_orientation = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=sim.device)
        target_orientation = root_state[:, 3:7].clone()
        # Rotate the target orientation 180 degrees around the y-axis
        target_orientation, target_position = torch_utils.tf_combine(
            target_orientation, target_position, 
            torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=self.sim.device), torch.tensor([[0.0, 0.0, 0.0]], device=self.sim.device)
        )

        # print(f"target_position: {target_position}, target_orientation: {target_orientation}")
        # print(f"target_position_h: {target_position_h}, target_orientation: {target_orientation}")



        # ik_commands = torch.cat([target_position, target_orientation], dim=-1,)

        # print("scene['robot'].data.joint_pos: ", scene["robot"].data.joint_pos)
        if self.count >= count_step[0] and self.count < count_step[1]:
            # self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, diff_ik_controller, 
            #                         target_position_h, target_orientation, None)
            # target_marker.visualize(target_position_h, target_orientation)
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, diff_ik_controller, 
                                    target_position_h, target_orientation, None)
        
        # Step 1.2: Open the gripper
        gripper_joint_ids = gripper_entity_cfg.joint_ids
        gripper_body_ids = gripper_entity_cfg.body_ids
        num_gripper_joints = len(gripper_joint_ids)

        # Step 1.3: Move the arm to the target position and keep the orientation
        # target_position_2 = target_position
        # target_orientation = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=sim.device)
        if self.count >= count_step[1] and self.count < count_step[2]:
            # self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, diff_ik_controller, 
            #                         target_position, target_orientation, None)
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, diff_ik_controller, 
                                    target_position, target_orientation, None)
            # target_marker.visualize(target_position, target_orientation)

        # Step 1.4: Close the gripper
        
        if self.count >= count_step[2] and self.count < count_step[3]:
            # gripper_joint_pos_des = torch.full(
            #         (num_gripper_joints,), 0.0, device=self.sim.device
            #     )
            # self.scene["robot"].set_joint_position_target(
            #         gripper_joint_pos_des, joint_ids=gripper_joint_ids
            #     )
            action = torch.tensor([[0.0]], device=self.sim.device)
            joint_ids = gripper_joint_ids


        # Step 1.5: Move the arm to the target position above the gear and keep the orientation
        # target_position = target_position + torch.tensor([0.0, 0.0, 0.1 + TCP_offset], device=sim.device)
        # target_orientation = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=sim.device)
        if self.count >= count_step[3] and self.count < count_step[4]:
            # self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
            #                         target_position_h, target_orientation, None)
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
                                    target_position_h, target_orientation, None)
            # gripper_joint_pos_des = torch.full(
            #         (num_gripper_joints,), 0.0, device=self.sim.device
            #     )
            # self.scene["robot"].set_joint_position_target(
            #         gripper_joint_pos_des, joint_ids=gripper_joint_ids
            #     )
            # target_marker.visualize(target_position_h, target_orientation)

        return action, joint_ids


    def mount_gear_to_target(self,
                                    gear_id: int,
                                    count_step: torch.Tensor,
                                    arm_entity_cfg: SceneEntityCfg,
                                    gripper_entity_cfg: SceneEntityCfg):

        obj_height_offset = 0.0
        mount_height_offset = 0.023

        if gear_id == 4:
            root_state = self.planetary_carrier.data.root_state_w.clone()
            if self.count == count_step[0]:
                self.current_target_position = root_state[:, :3].clone()
            # planetary_carrier_quat = root_state[:, 3:7].clone()
            obj_height_offset = 0.01
            mount_height_offset = 0.03

        elif gear_id == 6: # Reducer
            root_state = self.sun_planetary_gear_4.data.root_state_w.clone()
            if self.count == count_step[0]:
                self.current_target_position = root_state[:, :3].clone()
                self.current_target_orientation = root_state[:, 3:7].clone()
            obj_height_offset = 0.023 + 0.02
            mount_height_offset = 0.025


        else: # Mount the gear on the planetary carrier
            # root_state = initial_root_state[f"sun_planetary_gear_{gear_id}"]

            planetary_carrier_pos = self.planetary_carrier.data.root_state_w[:, :3].clone()
            planetary_carrier_quat = self.planetary_carrier.data.root_state_w[:, 3:7].clone()
            # original_planetary_carrier_pos = self.initial_root_state["planetary_carrier"][:, :3].clone()
            # original_planetary_carrier_quat = self.initial_root_state["planetary_carrier"][:, 3:7].clone()

            # Local pose of the pin
            pin_local_pos = self.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]['pin_local_pos'].clone()
            # Transfer the local pose of the pin to the world frame after the planetary carrier is moved
            # target_orientation = planetary_carrier_quat.clone()
            target_orientation, pin_world_pos = torch_utils.tf_combine(
                planetary_carrier_quat, planetary_carrier_pos, 
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.sim.device), pin_local_pos.unsqueeze(0)
            )
            # _, original_pin_world_pos = torch_utils.tf_combine(
            #     original_planetary_carrier_quat, original_planetary_carrier_pos, 
            #     torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.sim.device), pin_local_pos.unsqueeze(0)
            # )
            if self.count == count_step[0]:
                self.current_target_position = pin_world_pos.clone()
            # target_orientation = planetary_carrier_quat.clone()
        
        # target_marker.visualize(target_position, target_orientation)

        # print(f"self.current_target_position: {self.current_target_position}")

        target_position = self.current_target_position.clone()


        target_position[:, 2] = self.table_height + self.grasping_height
        target_position[:, 2] += obj_height_offset

        target_position += torch.tensor([self.TCP_offset_x, 0.0, self.TCP_offset_z], device=self.sim.device)
        
        target_position_h = target_position + torch.tensor([0.0, 0.0, self.lifting_height], device=self.sim.device)

        target_orientation = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=self.sim.device)

        if gear_id == 6:
            # Rotate the target orientation 180 degrees around the y-axis
            target_orientation, target_position = torch_utils.tf_combine(
                self.current_target_orientation, target_position, 
                torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=self.sim.device), torch.tensor([[0.0, 0.0, 0.0]], device=self.sim.device)
            )

        target_position_h_down = target_position + torch.tensor([0.0, 0.0, mount_height_offset], device=self.sim.device)

        if self.count >= count_step[0] and self.count < count_step[1]:
            # self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
            #                         target_position_h, target_orientation, None)
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
                                    target_position_h, target_orientation, None)
            # target_marker.visualize(target_position_h, target_orientation)

        if self.count >= count_step[1] and self.count < count_step[2]:

            # self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
            #                         target_position_h_down, target_orientation, None)
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
                                    target_position_h_down, target_orientation, None)
            # target_marker.visualize(target_position_h_down, target_orientation)

        if self.count >= count_step[2] and self.count < count_step[3]:
            gripper_joint_ids = gripper_entity_cfg.joint_ids
            gripper_body_ids = gripper_entity_cfg.body_ids
            num_gripper_joints = len(gripper_joint_ids)

            gripper_joint_pos_des = torch.full(
                    (num_gripper_joints,), 0.04, device=self.device
                )

            # if gear_id == 5:
            #     gripper_joint_pos_des = torch.full(
            #         (num_gripper_joints,), 0.017, device=self.device
            #     )

            # self.scene["robot"].set_joint_position_target(
            #         gripper_joint_pos_des, joint_ids=gripper_joint_ids
            #     )
            action = gripper_joint_pos_des.unsqueeze(0)
            joint_ids = gripper_joint_ids
            
        if self.count >= count_step[3] and self.count < count_step[4]:
            # self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
            #                         target_position_h, target_orientation, None)
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
                                    target_position_h, target_orientation, None)
            # target_marker.visualize(target_position_h, target_orientation)

        return action, joint_ids
    

    def mount_gear_to_target_and_rotate(self,
                                    gear_id: int,
                                    count_step: torch.Tensor,
                                    arm_entity_cfg: SceneEntityCfg,
                                    gripper_entity_cfg: SceneEntityCfg):

        root_state = self.planetary_carrier.data.root_state_w.clone()
        if self.count == count_step[0]:
            self.current_target_position = root_state[:, :3].clone()
        # planetary_carrier_quat = root_state[:, 3:7].clone()
        obj_height_offset = 0.01
        mount_height_offset = 0.040

        if gear_id == 5: # Place the ring gear on the carrier

            planetary_carrier_pos = self.planetary_carrier.data.root_state_w[:, :3].clone()
            planetary_carrier_quat = self.planetary_carrier.data.root_state_w[:, 3:7].clone()

            local_pos = torch.tensor([0.0, 0.0, 0.0], device=self.sim.device).unsqueeze(0)

            if self.count == count_step[0]:
                self.current_target_orientation, self.current_target_position = torch_utils.tf_combine(
                    planetary_carrier_quat, planetary_carrier_pos, 
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.sim.device), local_pos
                )

            mount_height_offset = 0.025 + 0.028


        target_position = self.current_target_position.clone()

        target_position[:, 2] = self.table_height + self.grasping_height
        target_position[:, 2] += obj_height_offset

        target_position += torch.tensor([self.TCP_offset_x, 0.0, self.TCP_offset_z], device=self.sim.device)
        
        target_position_h = target_position + torch.tensor([0.0, 0.0, self.lifting_height], device=self.sim.device)
        target_orientation = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=self.sim.device)

        target_position_h_down = target_position + torch.tensor([0.0, 0.0, mount_height_offset], device=self.sim.device)

        if self.count >= count_step[0] and self.count < count_step[1]:
            # self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
            #                         target_position_h, target_orientation, None)
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
                                    target_position_h, target_orientation, None)
            # target_marker.visualize(target_position_h, target_orientation)

        if self.count >= count_step[1] and self.count < count_step[2]:

            # self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
            #                         target_position_h_down, target_orientation, None)
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
                                    target_position_h_down, target_orientation, None)
            # target_marker.visualize(target_position_h_down, target_orientation)

        # Slightly rotate to fit into the gear
        if gear_id == 4:
            rot_deg = 60
        else:
            rot_deg = 30

        if self.count >= count_step[2] and self.count < count_step[3]:
            # joint_pos = joint_pos[:, arm_joint_ids]
            joint_ids = arm_entity_cfg.joint_ids

            delta_rot_rad = rot_deg / (count_step[3] - count_step[2]) * torch.pi / 180.0
            if self.count == count_step[2]:
                joint_pos = self.scene["robot"].data.joint_pos.clone()
                self.step_initial_joint_pos = joint_pos[:, joint_ids].clone()
            
            self.current_target_joint_pos = self.step_initial_joint_pos.clone()
            self.current_target_joint_pos[:, 5] += delta_rot_rad * (self.count - count_step[2] + 5)
            
            action = self.current_target_joint_pos

        if self.count >= count_step[3] and self.count < count_step[4]:
            gripper_joint_ids = gripper_entity_cfg.joint_ids
            gripper_body_ids = gripper_entity_cfg.body_ids
            num_gripper_joints = len(gripper_joint_ids)

            gripper_joint_pos_des = torch.full(
                    (num_gripper_joints,), 0.04, device=self.device
                )

            action = gripper_joint_pos_des.unsqueeze(0)
            joint_ids = gripper_joint_ids
            
        if self.count >= count_step[4] and self.count < count_step[5]:
            # self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
            #                         target_position_h, target_orientation, None)
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
                                    target_position_h, target_orientation, None)
            # target_marker.visualize(target_position_h, target_orientation)

        return action, joint_ids

    def place_gear_on_table(self,
                                    gear_id: int,
                                    count_step: torch.Tensor,
                                    arm_entity_cfg: SceneEntityCfg,
                                    gripper_entity_cfg: SceneEntityCfg):
        """Place a gear on the table at a side position"""
        
        # Define a position on the table side (away from center)
        if self.fourth_gear_table_position is None:
            # Store the table position for later pickup
            self.fourth_gear_table_position = torch.tensor(
                [0.5, 0.0, self.table_height + 0.01], device=self.device
            ).unsqueeze(0)
        
        target_position = self.fourth_gear_table_position.clone()
        target_position += torch.tensor([self.TCP_offset_x, 0.0, self.TCP_offset_z], device=self.device)
        
        target_position_h = target_position + torch.tensor([0.0, 0.0, self.lifting_height], device=self.device)
        target_orientation = torch.tensor([[0.0, -1.0, 0.0, 0.0]], device=self.device)

        # Step 1: Move to position above table placement point
        if self.count >= count_step[0] and self.count < count_step[1]:
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
                                    target_position_h, target_orientation, None)

        # Step 2: Move down to placement position
        if self.count >= count_step[1] and self.count < count_step[2]:
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
                                    target_position, target_orientation, None)

        # Step 3: Open gripper to release
        if self.count >= count_step[2] and self.count < count_step[3]:
            gripper_joint_ids = gripper_entity_cfg.joint_ids
            num_gripper_joints = len(gripper_joint_ids)
            gripper_joint_pos_des = torch.full(
                    (num_gripper_joints,), 0.04, device=self.device
                )
            action = gripper_joint_pos_des.unsqueeze(0)
            joint_ids = gripper_joint_ids

        # Step 4: Move back up
        if self.count >= count_step[3] and self.count < count_step[4]:
            action, joint_ids = self.move_robot_to_position(arm_entity_cfg, gripper_entity_cfg, self.diff_ik_controller, 
                                    target_position_h, target_orientation, None)

        return action, joint_ids

    def get_action(self):
        action = None
        joint_ids = None

        # Special logic for misplaced_fourth_gear state
        if self.initial_assembly_state == "misplaced_fourth_gear":
            return self._get_action_misplaced_fourth_gear()
        
        # Special logic for lack_fourth_gear state
        if self.initial_assembly_state == "lack_fourth_gear":
            return self._get_action_lack_fourth_gear()

        if self.count < self.count_step_0:
            action = torch.cat([self.initial_pos_left, self.initial_pos_right], dim=0).unsqueeze(0)
            joint_ids = self.left_arm_entity_cfg.joint_ids + self.right_arm_entity_cfg.joint_ids

        # Test
        # Mount the reducer to the gear
            # Pick up the reducer
        # if self.count >= self.count_step_1[0] and self.count < self.count_step_1[-1]:
        #     gear_id = 6

        #     arm = self.gear_to_pin_map['planetary_reducer']['arm']
        #     if arm == 'right':
        #         current_arm = self.right_arm_entity_cfg
        #         current_gripper = self.right_gripper_entity_cfg
        #     else:
        #         current_arm = self.left_arm_entity_cfg
        #         current_gripper = self.left_gripper_entity_cfg

        #     pick_action, pick_joint_ids = self.pick_up_target_gear(gear_id, self.count_step_1, current_arm, current_gripper, self.diff_ik_controller)
        #     # print(f'Pick action: {pick_action}')
        #     # print(f'pick_joint_ids: {pick_joint_ids}')
        #     action = pick_action
        #     joint_ids = pick_joint_ids

        # if self.count >= self.count_step_2[0] and self.count < self.count_step_2[-1]:
        #     gear_id = 6
        #     # Reducer location
        #     # pos = self.initial_root_state["planetary_reducer"][:, :3].clone()
        #     arm = self.gear_to_pin_map['planetary_reducer']['arm']
        #     if arm == 'right':
        #         current_arm = self.right_arm_entity_cfg
        #         current_gripper = self.right_gripper_entity_cfg
        #     else:
        #         current_arm = self.left_arm_entity_cfg
        #         current_gripper = self.left_gripper_entity_cfg
        #     action, joint_ids = self.mount_gear_to_target(gear_id, self.count_step_2, current_arm, current_gripper)


        # Pick up the 1st gear
        if self.count >= self.count_step_1[0] and self.count < self.count_step_1[-1]:
        
            gear_id = 1
            current_arm_str = self.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]['arm']
            if current_arm_str == 'left':
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            else:
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg

            action, joint_ids = self.pick_up_target_gear(gear_id, self.count_step_1, current_arm, current_gripper, self.diff_ik_controller)

        # Mount the 1st gear to the planetary_carrier
        if self.count >= self.count_step_2[0] and self.count < self.count_step_2[-1]:
            gear_id = 1
            current_arm_str = self.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]['arm']
            if current_arm_str == 'left':
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            else:
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
                
            action, joint_ids = self.mount_gear_to_target(gear_id, self.count_step_2, current_arm, current_gripper)


        # Pick up the 2nd gear
        if self.count >= self.count_step_3[0] and self.count < self.count_step_3[-1]:
            gear_id = 2
            current_arm_str = self.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]['arm']
            if current_arm_str == 'left':
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            else:
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.pick_up_target_gear(gear_id, self.count_step_3, current_arm, current_gripper, self.diff_ik_controller)

        # Mount the 2nd gear to the planetary_carrier
        if self.count >= self.count_step_4[0] and self.count < self.count_step_4[-1]:
            gear_id = 2
            current_arm_str = self.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]['arm']
            if current_arm_str == 'left':
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            else:
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.mount_gear_to_target(gear_id, self.count_step_4, current_arm, current_gripper)

        # Reset left arm
        if self.count >= self.count_step_5[0] and self.count < self.count_step_5[-1]:
            action = self.initial_pos_left.unsqueeze(0)
            joint_ids = self.left_arm_entity_cfg.joint_ids

        # Pick up the 3rd gear
        if self.count >= self.count_step_6[0] and self.count < self.count_step_6[-1]:
            gear_id = 3
            current_arm_str = self.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]['arm']
            if current_arm_str == 'left':
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            else:
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.pick_up_target_gear(gear_id, self.count_step_6, current_arm, current_gripper, self.diff_ik_controller)

        # Mount the 3rd gear to the planetary_carrier
        if self.count >= self.count_step_7[0] and self.count < self.count_step_7[-1]:
            gear_id = 3
            current_arm_str = self.gear_to_pin_map[f"sun_planetary_gear_{gear_id}"]['arm']
            if current_arm_str == 'left':
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            else:
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.mount_gear_to_target(gear_id, self.count_step_7, current_arm, current_gripper)

        # Pick up the 4th gear
        if self.count >= self.count_step_8[0] and self.count < self.count_step_8[-1]:
            gear_id = 4
            current_arm = self.right_arm_entity_cfg
            current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.pick_up_target_gear(gear_id, self.count_step_8, current_arm, current_gripper, self.diff_ik_controller)

        # Mount the 4th gear to the planetary_carrier
        if self.count >= self.count_step_9[0] and self.count < self.count_step_9[-1]:
            gear_id = 4
            current_arm = self.right_arm_entity_cfg
            current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.mount_gear_to_target_and_rotate(gear_id, self.count_step_9, current_arm, current_gripper)

        # Reset right arm
        if self.count >= self.count_step_10[0] and self.count < self.count_step_10[-1]:
            action = self.initial_pos_right.unsqueeze(0)
            joint_ids = self.right_arm_entity_cfg.joint_ids

        # Pick up the ring gear
        if self.count >= self.count_step_11[0] and self.count < self.count_step_11[-1]:
            gear_id = 5
            arm = self.gear_to_pin_map['ring_gear']['arm']
            # ring_gear_pos = self.initial_root_state["ring_gear"][:, :3].clone()
            # if ring_gear_pos[0, 1] < 0.0:
            if arm == 'right':
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            else:
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            action, joint_ids = self.pick_up_target_gear(gear_id, self.count_step_11, current_arm, current_gripper, self.diff_ik_controller)

        # Mount the ring gear on the carrier
        if self.count >= self.count_step_12[0] and self.count < self.count_step_12[-1]:
            gear_id = 5
            arm = self.gear_to_pin_map['ring_gear']['arm']
            # ring_gear_pos = self.initial_root_state["ring_gear"][:, :3].clone()
            # if ring_gear_pos[0, 1] < 0.0:
            if arm == 'right':
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            else:
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            action, joint_ids = self.mount_gear_to_target_and_rotate(gear_id, self.count_step_12, current_arm, current_gripper)

        # Pick up the reducer
        if self.count >= self.count_step_13[0] and self.count < self.count_step_13[-1]:
            gear_id = 6

            arm = self.gear_to_pin_map['planetary_reducer']['arm']
            if arm == 'right':
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            else:
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg

            pick_action, pick_joint_ids = self.pick_up_target_gear(gear_id, self.count_step_13, current_arm, current_gripper, self.diff_ik_controller)
            # print(f'Pick action: {pick_action}')
            # print(f'pick_joint_ids: {pick_joint_ids}')
            action = pick_action
            joint_ids = pick_joint_ids

            if self.gear_to_pin_map['planetary_reducer'] != self.gear_to_pin_map['ring_gear']:
                reset_arm = self.gear_to_pin_map['ring_gear']['arm']
                if reset_arm == 'right':
                    reset_action = self.initial_pos_right.unsqueeze(0)
                    reset_joint_ids = self.right_arm_entity_cfg.joint_ids
                else:
                    reset_action = self.initial_pos_left.unsqueeze(0)
                    reset_joint_ids = self.left_arm_entity_cfg.joint_ids

                print(f'Pick action: {pick_action}')
                print(f'pick_joint_ids: {pick_joint_ids}')
                print(f'Reset action: {reset_action}')
                print(f'reset_joint_ids: {reset_joint_ids}')

                # action = torch.cat([pick_action, reset_action], dim=1).unsqueeze(0)
                action = torch.cat([pick_action, reset_action], dim=1)
                joint_ids = pick_joint_ids + reset_joint_ids

                print(f'Action: {action}')
                print(f'joint_ids: {joint_ids}')


        # Mount the reducer to the gear
        if self.count >= self.count_step_14[0] and self.count < self.count_step_14[-1]:
            gear_id = 6
            # Reducer location
            # pos = self.initial_root_state["planetary_reducer"][:, :3].clone()
            arm = self.gear_to_pin_map['planetary_reducer']['arm']
            if arm == 'right':
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            else:
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            action, joint_ids = self.mount_gear_to_target(gear_id, self.count_step_14, current_arm, current_gripper)

        return action, joint_ids

    def _get_action_misplaced_fourth_gear(self):
        """Special action sequence for misplaced_fourth_gear initial state"""
        action = None
        joint_ids = None

        # Initial stabilization
        if self.count < self.count_step_0:
            action = torch.cat([self.initial_pos_left, self.initial_pos_right], dim=0).unsqueeze(0)
            joint_ids = self.left_arm_entity_cfg.joint_ids + self.right_arm_entity_cfg.joint_ids

        # Step 1: Pick up the misplaced 4th gear from on top of other gears
        if self.count >= self.count_step_misplaced_1[0] and self.count < self.count_step_misplaced_1[-1]:
            gear_id = 4
            current_arm = self.right_arm_entity_cfg
            current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.pick_up_target_gear(gear_id, self.count_step_misplaced_1, current_arm, current_gripper, self.diff_ik_controller)

        # Step 2: Mount the 4th gear to the carrier
        if self.count >= self.count_step_misplaced_2[0] and self.count < self.count_step_misplaced_2[-1]:
            gear_id = 4
            current_arm = self.right_arm_entity_cfg
            current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.mount_gear_to_target_and_rotate(gear_id, self.count_step_misplaced_2, current_arm, current_gripper)

        # Step 3: Reset right arm
        if self.count >= self.count_step_misplaced_3[0] and self.count < self.count_step_misplaced_3[-1]:
            action = self.initial_pos_right.unsqueeze(0)
            joint_ids = self.right_arm_entity_cfg.joint_ids

        # Step 4: Pick up the ring gear
        if self.count >= self.count_step_misplaced_4[0] and self.count < self.count_step_misplaced_4[-1]:
            gear_id = 5
            arm = self.gear_to_pin_map['ring_gear']['arm']
            if arm == 'right':
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            else:
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            action, joint_ids = self.pick_up_target_gear(gear_id, self.count_step_misplaced_4, current_arm, current_gripper, self.diff_ik_controller)

        # Step 5: Mount the ring gear on the carrier
        if self.count >= self.count_step_misplaced_5[0] and self.count < self.count_step_misplaced_5[-1]:
            gear_id = 5
            arm = self.gear_to_pin_map['ring_gear']['arm']
            if arm == 'right':
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            else:
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            action, joint_ids = self.mount_gear_to_target_and_rotate(gear_id, self.count_step_misplaced_5, current_arm, current_gripper)

        return action, joint_ids

    def _get_action_lack_fourth_gear(self):
        """Special action sequence for lack_fourth_gear initial state - skips first 3 gears"""
        action = None
        joint_ids = None

        # Initial stabilization
        if self.count < self.count_step_0:
            action = torch.cat([self.initial_pos_left, self.initial_pos_right], dim=0).unsqueeze(0)
            joint_ids = self.left_arm_entity_cfg.joint_ids + self.right_arm_entity_cfg.joint_ids

        # Pick up the 4th gear (directly from step 8)
        if self.count >= self.count_step_8[0] and self.count < self.count_step_8[-1]:
            gear_id = 4
            current_arm = self.right_arm_entity_cfg
            current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.pick_up_target_gear(gear_id, self.count_step_8, current_arm, current_gripper, self.diff_ik_controller)

        # Mount the 4th gear to the planetary_carrier
        if self.count >= self.count_step_9[0] and self.count < self.count_step_9[-1]:
            gear_id = 4
            current_arm = self.right_arm_entity_cfg
            current_gripper = self.right_gripper_entity_cfg
            action, joint_ids = self.mount_gear_to_target_and_rotate(gear_id, self.count_step_9, current_arm, current_gripper)

        # Reset right arm
        if self.count >= self.count_step_10[0] and self.count < self.count_step_10[-1]:
            action = self.initial_pos_right.unsqueeze(0)
            joint_ids = self.right_arm_entity_cfg.joint_ids

        # Pick up the ring gear
        if self.count >= self.count_step_11[0] and self.count < self.count_step_11[-1]:
            gear_id = 5
            arm = self.gear_to_pin_map['ring_gear']['arm']
            if arm == 'right':
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            else:
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            action, joint_ids = self.pick_up_target_gear(gear_id, self.count_step_11, current_arm, current_gripper, self.diff_ik_controller)

        # Mount the ring gear on the carrier
        if self.count >= self.count_step_12[0] and self.count < self.count_step_12[-1]:
            gear_id = 5
            arm = self.gear_to_pin_map['ring_gear']['arm']
            if arm == 'right':
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            else:
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            action, joint_ids = self.mount_gear_to_target_and_rotate(gear_id, self.count_step_12, current_arm, current_gripper)

        # Pick up the reducer
        if self.count >= self.count_step_13[0] and self.count < self.count_step_13[-1]:
            gear_id = 6
            arm = self.gear_to_pin_map['planetary_reducer']['arm']
            if arm == 'right':
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            else:
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg

            pick_action, pick_joint_ids = self.pick_up_target_gear(gear_id, self.count_step_13, current_arm, current_gripper, self.diff_ik_controller)
            action = pick_action
            joint_ids = pick_joint_ids

            if self.gear_to_pin_map['planetary_reducer'] != self.gear_to_pin_map['ring_gear']:
                reset_arm = self.gear_to_pin_map['ring_gear']['arm']
                if reset_arm == 'right':
                    reset_action = self.initial_pos_right.unsqueeze(0)
                    reset_joint_ids = self.right_arm_entity_cfg.joint_ids
                else:
                    reset_action = self.initial_pos_left.unsqueeze(0)
                    reset_joint_ids = self.left_arm_entity_cfg.joint_ids

                action = torch.cat([pick_action, reset_action], dim=1)
                joint_ids = pick_joint_ids + reset_joint_ids

        # Mount the reducer to the gear
        if self.count >= self.count_step_14[0] and self.count < self.count_step_14[-1]:
            gear_id = 6
            arm = self.gear_to_pin_map['planetary_reducer']['arm']
            if arm == 'right':
                current_arm = self.right_arm_entity_cfg
                current_gripper = self.right_gripper_entity_cfg
            else:
                current_arm = self.left_arm_entity_cfg
                current_gripper = self.left_gripper_entity_cfg
            action, joint_ids = self.mount_gear_to_target(gear_id, self.count_step_14, current_arm, current_gripper)

        return action, joint_ids