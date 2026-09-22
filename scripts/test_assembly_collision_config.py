"""Check the contact settings on spawned USD collision shapes, including instances.

Run after sourcing scripts/env.sh:
    python scripts/test_assembly_collision_config.py --headless
"""

import argparse
import unittest

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics
from Galaxea_Lab_External.robots.galaxea_robots import GALAXEA_R1_LITE_CFG
from Galaxea_Lab_External.robots.gears_assets import (
    RING_GEAR_CFG, SUN_PLANETARY_GEAR_CFG, PLANETARY_CARRIER_CFG, PLANETARY_REDUCER_CFG,
)


class AssemblyCollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        omni.usd.get_context().new_stage()
        cls.stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(cls.stage, "/World")

    def colliders(self, path):
        return [p for p in Usd.PrimRange(self.stage.GetPrimAtPath(path), Usd.TraverseInstanceProxies())
                if p.HasAPI(UsdPhysics.CollisionAPI)]

    def test_lite_margin_reaches_every_spawned_collision_shape(self):
        cfg = GALAXEA_R1_LITE_CFG.spawn
        cfg.func("/World/Robot", cfg)
        shapes = self.colliders("/World/Robot")
        self.assertTrue(any("torso_link3" in str(p.GetPath()) for p in shapes))
        for shape in shapes:
            with self.subTest(shape=str(shape.GetPath())):
                contact = shape.GetAttribute("physxCollision:contactOffset").Get()
                self.assertIsNotNone(contact, "Configured margin did not reach an instanced collision mesh")
                self.assertAlmostEqual(contact, cfg.collision_props.contact_offset, places=6)

    def test_assembly_parts_have_valid_contact_offsets(self):
        for name, obj in (("Ring", RING_GEAR_CFG), ("Sun", SUN_PLANETARY_GEAR_CFG),
                          ("Carrier", PLANETARY_CARRIER_CFG), ("Reducer", PLANETARY_REDUCER_CFG)):
            path = f"/World/{name}"
            obj.spawn.func(path, obj.spawn)
            shapes = self.colliders(path)
            self.assertTrue(shapes)
            for shape in shapes:
                with self.subTest(part=name, shape=str(shape.GetPath())):
                    contact = shape.GetAttribute("physxCollision:contactOffset").Get()
                    rest = shape.GetAttribute("physxCollision:restOffset").Get()
                    self.assertGreater(contact, 0.0, "PhysX rejects zero contact offsets")
                    self.assertGreater(contact, rest)


result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AssemblyCollisionTests))
print(f"ASSEMBLY_COLLISION_TESTS_PASSED={result.wasSuccessful()}", flush=True)
if not result.wasSuccessful():
    raise AssertionError("Assembly collision-configuration checks failed")
app.close(wait_for_replicator=False)
