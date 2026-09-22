"""Record one fresh assembly attempt; used by collect_assembly_data.py."""

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import time

from isaaclab.app import AppLauncher

from assembly_dataset import source_manifest, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--robot", choices=("r1_pro", "r1_lite"), required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--source-commit", required=True)
parser.add_argument("--runtime-label", required=True)
parser.add_argument("--fast_physics", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.enable_cameras:
    parser.error("Recorded collection requires --enable_cameras")
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=True)
if any((args.output / name).exists() for name in ("episode.json", "episode.partial.hdf5", "episode.hdf5")):
    parser.error("Output already contains an episode; use a new attempt directory")
os.environ["ROCO_ROBOT_BUNDLE"] = args.robot
os.environ["ROCO_DATA_DIR"] = str(args.output)
app = AppLauncher(args).app

import h5py
import torch
from isaacsim.core.version import get_version

from Galaxea_Lab_External.robots.physics_profiles import use_fast_physics
from Galaxea_Lab_External.tasks.direct.galaxea_lab_external.galaxea_lab_external_env import GalaxeaLabExternalEnv
from Galaxea_Lab_External.tasks.direct.galaxea_lab_external.galaxea_lab_external_env_cfg import GalaxeaLabExternalEnvCfg

loaded_source_manifest = source_manifest(Path(__file__).resolve().parents[1])


def snapshot(env):
    policy = env.rule_policy
    return {
        "simulation_seconds": round(policy.count * policy.sim_dt, 5),
        "state": getattr(policy, "_planetary_state", None),
        "active_gear": getattr(policy, "_planetary_gear", None),
        "assembly_score": int(env.score_tensor.item()),
        "failure": getattr(policy, "planetary_failure", None),
        "verified_first_three": bool(getattr(policy, "planetary_complete", False)),
        "verified_fourth": bool(getattr(policy, "sun_complete", False)),
        "verified_fifth": bool(getattr(policy, "assembly_complete", False)),
        "object_poses": {key: obj.data.root_state_w[0, :7].tolist() for key, obj in env.obj_dict.items()},
    }


class RecordingEnv(GalaxeaLabExternalEnv):
    def _get_dones(self):
        done = super()._get_dones()
        self.collection_snapshot = snapshot(self)
        return done

    def _get_rewards(self):
        return self.score_tensor.to(torch.float32)

    def _write_hdf5_episode(self, success):
        if not success:
            return
        self.save_hdf5_file_name = str(args.output / "episode.partial.hdf5")
        super()._write_hdf5_episode(success)
        self.collection_recording = "episode.partial.hdf5"


cfg = GalaxeaLabExternalEnvCfg()
cfg.seed = args.seed
cfg.sim.device = args.device
cfg.record_data = True
cfg.keep_failed = None
cfg.num_rerenders_on_reset = 5
if args.fast_physics:
    use_fast_physics(cfg)

env = None
try:
    with (args.output / "simulation.log").open("w") as log, (args.output / "trace.jsonl").open("w") as trace:
        with torch.inference_mode():
            with contextlib.redirect_stdout(log):
                env = RecordingEnv(cfg=cfg)
                env.reset(seed=args.seed)
            initial_poses = snapshot(env)["object_poses"]
            started = time.perf_counter()
            previous = None
            for _ in range(math.ceil(cfg.episode_length_s / env.step_dt)):
                with contextlib.redirect_stdout(log):
                    _, _, terminated, truncated, _ = env.step(torch.zeros((1, cfg.action_space), device=env.device))
                record = env.collection_snapshot
                trace.write(json.dumps(record) + "\n")
                trace.flush()
                state = (record["active_gear"], record["state"])
                if state != previous:
                    print(
                        json.dumps(
                            {k: record[k] for k in ("simulation_seconds", "active_gear", "state", "assembly_score")}
                        ),
                        flush=True,
                    )
                    previous = state
                if bool(terminated.item() or truncated.item()) or record["failure"]:
                    break
            recording = getattr(env, "collection_recording", None)
            success = record["assembly_score"] == 5 and record["verified_fifth"] and recording is not None
            result = {
                **record,
                "assembly_success": bool(success),
                "recording": recording,
                "robot": args.robot,
                "seed": args.seed,
                "hostname": socket.gethostname(),
                "runtime_label": args.runtime_label,
                "source_commit": args.source_commit,
                "source_manifest": loaded_source_manifest,
                "python_version": platform.python_version(),
                "torch_version": str(torch.__version__),
                "isaac_sim_version": list(get_version()),
                "gpu": torch.cuda.get_device_name(),
                "fast_physics": args.fast_physics,
                "physics_dt": env.physics_dt,
                "control_dt": env.step_dt,
                "robot_solver_position_iterations": (
                    cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count
                ),
                "robot_solver_velocity_iterations": (
                    cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count
                ),
                "wall_seconds_excluding_startup": time.perf_counter() - started,
                "initial_object_poses": initial_poses,
            }
            if recording:
                with h5py.File(args.output / recording, "r+") as h:
                    for key in ("robot", "seed", "hostname", "runtime_label", "source_commit"):
                        h.attrs[key] = result[key]
                    h.attrs["isaac_sim_version"] = json.dumps(result["isaac_sim_version"])
                    h.attrs["verified_after_release_and_parking"] = bool(record["verified_fifth"])
            write_json(args.output / "episode.json", result)
            print(json.dumps({"assembly_success": bool(success), "output": str(args.output)}), flush=True)
finally:
    if env is not None:
        env.close()
    app.close()
