"""Collect a resumable quota of verified camera episodes for one robot/host.

Source scripts/env.sh first. A fresh simulator process is used for each seed.
Failed attempts retain small logs and traces, but no demonstration HDF5.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from assembly_dataset import source_manifest, verify_recording, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("r1_pro", "r1_lite"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--successes", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--episode-timeout", type=float, default=1800)
    parser.add_argument("--runtime-label", required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--fast_physics", action="store_true")
    parser.add_argument("--experience", default="")
    args = parser.parse_args()
    if args.successes < 1 or args.max_attempts < args.successes or args.episode_timeout <= 0:
        parser.error("Require positive quota/timeout and max-attempts >= successes")
    root = Path(__file__).resolve().parents[1]
    commit = args.source_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".collection.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    expected = {
        "robot": args.robot,
        "successes_requested": args.successes,
        "seed_start": args.seed_start,
        "max_attempts": args.max_attempts,
        "runtime_label": args.runtime_label,
        "source_commit": commit,
        "source_manifest": source_manifest(root),
        "fast_physics": args.fast_physics,
        "hostname": socket.gethostname(),
        "experience": args.experience,
    }
    config = output / "collection.json"
    if config.exists():
        if json.loads(config.read_text()) != expected:
            parser.error("Existing collection configuration differs; choose a new output directory")
    else:
        write_json(config, expected)
    accepted = []
    attempts = []
    stop_requested = False

    def stop_after_episode(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested; finishing the current attempt before exit.", flush=True)

    signal.signal(signal.SIGTERM, stop_after_episode)
    signal.signal(signal.SIGINT, stop_after_episode)

    def status(state):
        write_json(
            output / "status.json",
            {
                **expected,
                "state": state,
                "pid": os.getpid(),
                "updated_unix": time.time(),
                "successful_episodes": len(accepted),
                "attempts": attempts,
                "episodes": accepted,
            },
        )

    errors = 0
    for index in range(args.max_attempts):
        if len(accepted) >= args.successes or stop_requested or (output / "STOP").exists():
            break
        seed = args.seed_start + index
        directory = output / f"seed{seed}"
        outcome_path = directory / "outcome.json"
        if outcome_path.exists():
            outcome = json.loads(outcome_path.read_text())
            if outcome["status"] == "accepted":
                recording = directory / "episode.hdf5"
                with recording.open("rb") as stream:
                    assert hashlib.file_digest(stream, "sha256").hexdigest() == outcome["sha256"]
                accepted.append(outcome)
            attempts.append(outcome)
            continue
        if source_manifest(root) != expected["source_manifest"]:
            raise RuntimeError("Production source changed during collection; start a new collection")
        if shutil.disk_usage(output).free < 20 * 1024**3:
            raise RuntimeError("Collection stopped with less than 20 GiB free")
        if directory.exists():
            # Preserve an interrupted attempt; never overwrite partial output.
            abandoned = output / f"seed{seed}.interrupted-{time.time_ns()}"
            directory.rename(abandoned)
        directory.mkdir()
        command = [
            sys.executable,
            "-u",
            str(root / "scripts/record_assembly_episode.py"),
            "--robot",
            args.robot,
            "--seed",
            str(seed),
            "--output",
            str(directory),
            "--source-commit",
            commit,
            "--runtime-label",
            args.runtime_label,
            "--headless",
            "--enable_cameras",
        ]
        if args.fast_physics:
            command.append("--fast_physics")
        if args.experience:
            command.extend(["--experience", args.experience])
        write_json(directory / "command.json", command)
        status("running")
        print(f"{args.robot}: starting seed {seed}; {len(accepted)}/{args.successes} accepted", flush=True)
        started = time.perf_counter()
        with (directory / "process.log").open("w") as log:
            try:
                process = subprocess.run(
                    command,
                    cwd=root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.episode_timeout,
                    start_new_session=True,
                )
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                returncode = "timeout"
        outcome = {
            "seed": seed,
            "directory": directory.name,
            "returncode": returncode,
            "wall_seconds": time.perf_counter() - started,
        }
        result_path = directory / "episode.json"
        if returncode != 0 or not result_path.exists():
            outcome.update(status="runtime_error", error="Simulator failed; inspect process.log")
            errors += 1
        else:
            result = json.loads(result_path.read_text())
            if not result["assembly_success"]:
                outcome.update(status="assembly_failed", score=result["assembly_score"], failure=result["failure"])
                errors = 0
            else:
                try:
                    assert result["source_manifest"] == expected["source_manifest"]
                    verification = verify_recording(directory)
                    partial = directory / result["recording"]
                    partial.replace(directory / "episode.hdf5")
                    result["recording"] = "episode.hdf5"
                    verification["recording"] = "episode.hdf5"
                    write_json(result_path, result)
                    write_json(directory / "verification.json", verification)
                    outcome.update(status="accepted", **verification)
                    accepted.append(outcome)
                    errors = 0
                except (AssertionError, KeyError, ValueError, OSError) as exc:
                    outcome.update(status="recording_invalid", error=str(exc))
                    errors += 1
        write_json(outcome_path, outcome)
        attempts.append(outcome)
        status("running")
        print(json.dumps(outcome), flush=True)
        if errors >= 2:
            status("runtime_error")
            raise RuntimeError("Two consecutive runtime/recording errors; stopped to preserve resources")
    state = "complete" if len(accepted) >= args.successes else "stopped"
    status(state)
    print(f"{state}: {len(accepted)}/{args.successes} verified episodes in {output}", flush=True)
    return 0 if state == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
