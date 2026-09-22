"""Recording integrity checks shared by assembly collection entrypoints."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def source_manifest(root):
    root = Path(root)
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "source/Galaxea_Lab_External/Galaxea_Lab_External").rglob("*.py"))
    }


def verify_recording(directory):
    """Check the actual saved arrays against the pre-reset outcome and trace."""
    directory = Path(directory)
    result = json.loads((directory / "episode.json").read_text())
    assert result["assembly_success"] and result["assembly_score"] == 5
    assert result["failure"] is None
    assert all(result[k] for k in ("verified_first_three", "verified_fourth", "verified_fifth"))
    path = directory / result["recording"]
    dt = result["control_dt"]
    scores_by_tick = {}
    for line in (directory / "trace.jsonl").read_text().splitlines():
        row = json.loads(line)
        tick = round(row["simulation_seconds"] / dt)
        if tick in scores_by_tick:
            assert scores_by_tick[tick] == row["assembly_score"]
        scores_by_tick[tick] = row["assembly_score"]
    arm_count = 7 if result["robot"] == "r1_pro" else 6
    torso_count = 4 if result["robot"] == "r1_pro" else 3
    with h5py.File(path, "r") as h:
        assert bool(h.attrs["success"]) and bool(h.attrs["sim"])
        for key in ("seed", "robot", "source_commit", "runtime_label"):
            assert h.attrs[key] == result[key]
        for key in ("physics_dt", "control_dt", "robot_solver_position_iterations", "robot_solver_velocity_iterations"):
            assert h.attrs[key] == result[key]
        times, scores = h["current_time"][:], h["score"][:]
        assert times.ndim == 1 and len(times) > 10 and np.isfinite(times).all()
        np.testing.assert_allclose(np.diff(times), dt, atol=2e-5, rtol=0)
        ticks = np.rint(times / dt).astype(int)
        np.testing.assert_allclose(times, ticks * dt, atol=2e-5, rtol=0)
        assert scores.shape == times.shape and scores.dtype.kind in "iu"
        expected = [scores_by_tick[int(tick)] for tick in ticks]
        np.testing.assert_array_equal(scores, expected)
        assert scores[0] == 0 and scores[-1] == 5 and np.all((scores >= 0) & (scores <= 5))
        count = len(times)
        for camera in ("head", "left_hand", "right_hand"):
            rgb, depth = h[f"observations/{camera}_rgb"], h[f"observations/{camera}_depth"]
            assert rgb.shape == (count, 240, 320, 3) and rgb.dtype == np.dtype("uint8")
            assert depth.shape == (count, 240, 320) and depth.dtype == np.dtype("float32")
            for index in np.unique(np.linspace(0, count - 1, 5, dtype=int)):
                frame, distance = rgb[index], depth[index]
                assert int(frame.max()) > int(frame.min())
                assert np.any(np.isfinite(distance) & (distance > 0))
            assert not np.array_equal(rgb[0], rgb[-1])
        for side in ("left", "right"):
            for kind in ("pos", "vel"):
                values = h[f"observations/{side}_arm_joint_{kind}"][:]
                assert values.shape == (count, arm_count) and np.isfinite(values).all()
                values = h[f"observations/{side}_gripper_joint_{kind}"][:]
                assert values.shape == (count,) and np.isfinite(values).all()
            values = h[f"actions/{side}_arm_action"][:]
            assert values.shape == (count, arm_count) and np.isfinite(values).all()
            values = h[f"actions/{side}_gripper_action"][:]
            assert values.shape == (count,) and np.isfinite(values).all()
        for key in ("actions/torso_action", "observations/torso_joint_pos", "observations/torso_joint_vel"):
            values = h[key][:]
            assert values.shape == (count, torso_count) and np.isfinite(values).all()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "passed": True,
        "seed": result["seed"],
        "robot": result["robot"],
        "recording": path.name,
        "frames_per_camera": count,
        "final_score": 5,
        "all_scores_match_trace": True,
        "finite_states_and_actions": True,
        "camera_content_samples_per_camera": 5,
        "bytes": path.stat().st_size,
        "sha256": digest,
    }
