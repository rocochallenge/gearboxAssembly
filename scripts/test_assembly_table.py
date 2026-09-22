"""Check that contact loads cannot move the assembly work surface.

Run after sourcing scripts/env.sh:
    python scripts/test_assembly_table.py --headless
"""

import argparse
import unittest

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from Galaxea_Lab_External.tasks.direct.galaxea_lab_external.galaxea_lab_external_env_cfg import GalaxeaLabExternalEnvCfg


class AssemblyTableTests(unittest.TestCase):
    def test_contact_load_cannot_move_the_work_surface(self):
        cfg = GalaxeaLabExternalEnvCfg()
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
        try:
            sim_utils.spawn_ground_plane("/World/Ground", sim_utils.GroundPlaneCfg())
            table = RigidObject(cfg.table_cfg.replace(prim_path="/World/Table"))
            # A ring-sized mass lands on the actual production table asset.
            # Velocity caps alone previously allowed millimetres of drift.
            load = RigidObject(RigidObjectCfg(
                prim_path="/World/Load",
                spawn=sim_utils.CuboidCfg(
                    size=(0.08, 0.08, 0.08),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.27),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.98)),
            ))
            sim.reset()
            initial = table.root_physx_view.get_transforms().clone()
            max_motion = 0.0
            for _ in range(200):
                sim.step(render=False)
                # Read PhysX directly; this object is not in an updated scene.
                pose = table.root_physx_view.get_transforms()
                max_motion = max(max_motion, float((pose[:, :3] - initial[:, :3]).norm()))
                self.assertLess(float((pose[:, 3:] - initial[:, 3:]).norm()), 1e-5)
            self.assertLess(max_motion, 1e-5, f"Table moved {max_motion * 1000:.3f} mm under load")
            self.assertLess(float(load.root_physx_view.get_transforms()[0, 2]), 0.975)
        finally:
            sim.clear_all_callbacks()
            sim.clear_instance()


result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AssemblyTableTests))
print(f"ASSEMBLY_TABLE_TESTS_PASSED={result.wasSuccessful()}", flush=True)
if not result.wasSuccessful():
    raise AssertionError("Assembly table checks failed")
app.close(wait_for_replicator=False)
