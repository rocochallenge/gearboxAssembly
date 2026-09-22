"""Index accepted episodes after collecting or copying worker output folders."""

import argparse
import hashlib
import json
from pathlib import Path

from assembly_dataset import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--verify-hashes", action="store_true", help="Re-read every accepted HDF5, e.g. after transfer")
    args = parser.parse_args()
    root = args.directory.resolve()
    episodes = []
    identities = set()
    for outcome_path in sorted(root.glob("*/*/seed*/outcome.json")):
        if ".interrupted-" in outcome_path.parent.name:
            continue
        outcome = json.loads(outcome_path.read_text())
        if outcome["status"] != "accepted":
            continue
        directory = outcome_path.parent
        result = json.loads((directory / "episode.json").read_text())
        verification = json.loads((directory / "verification.json").read_text())
        path = directory / "episode.hdf5"
        assert result["assembly_success"] and result["verified_fifth"] and verification["passed"]
        assert path.stat().st_size == outcome["bytes"] == verification["bytes"]
        assert outcome["sha256"] == verification["sha256"]
        if args.verify_hashes:
            with path.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == outcome["sha256"]
        identity = (result["robot"], result["seed"])
        if identity in identities:
            raise ValueError(f"Duplicate robot/seed identity: {identity}")
        identities.add(identity)
        episodes.append({
            "robot": result["robot"],
            "seed": result["seed"],
            "recording": str(path.relative_to(root)),
            "metadata": str((directory / "episode.json").relative_to(root)),
            "workstation": outcome_path.relative_to(root).parts[0],
            "hostname": result["hostname"],
            "runtime_label": result["runtime_label"],
            "isaac_sim_version": result["isaac_sim_version"],
            "source_commit": result["source_commit"],
            "frames": verification["frames_per_camera"],
            "simulation_seconds": result["simulation_seconds"],
            "bytes": verification["bytes"],
            "sha256": verification["sha256"],
        })
    summary = {
        "episodes": episodes,
        "counts": {robot: sum(e["robot"] == robot for e in episodes) for robot in ("r1_pro", "r1_lite")},
        "total_bytes": sum(e["bytes"] for e in episodes),
        "hashes_rechecked": args.verify_hashes,
        "paths_relative_to": ".",
    }
    write_json(root / "dataset.json", summary)
    temporary = root / "episodes.jsonl.tmp"
    temporary.write_text("".join(json.dumps(episode) + "\n" for episode in episodes))
    temporary.replace(root / "episodes.jsonl")
    print(json.dumps({key: summary[key] for key in ("counts", "total_bytes", "hashes_rechecked")}))


if __name__ == "__main__":
    main()
