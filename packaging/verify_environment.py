#!/usr/bin/env python3
"""Verify pinned versions and dependency metadata without importing Isaac Sim.

--simulation starts a small, finite GPU smoke test explicitly; not a task gate.
Arena has three EXACT upstream override exceptions, not a blanket ignore.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
ARENA_EXCEPTIONS = {
    ("openpi-client", "numpy", "2.3.1"),
    ("isaaclab", "coverage", "7.4.4"),
    ("isaacsim-kernel", "websockets", "16.1.1"),
}


def dependency_issues(profile: str) -> tuple[list[str], list[str]]:
    installed = {canonicalize_name(d.metadata["Name"]): d for d in metadata.distributions()}
    errors, warnings = [], []
    for name, dist in installed.items():
        for text in dist.requires or []:
            req = Requirement(text)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            target = installed.get(canonicalize_name(req.name))
            if target is None:
                errors.append(f"{name}: missing {req}")
            elif req.specifier and not req.specifier.contains(target.version, prereleases=True):
                message = f"{name} requires {req}, installed {target.version}"
                key = (name, canonicalize_name(req.name), target.version)
                (warnings if profile == "arena" and key in ARENA_EXCEPTIONS else errors).append(message)
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("main", "arena"), default="main")
    parser.add_argument("--assets", action="store_true")
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--wbc", action="store_true", help="Initialize the isolated Arena worker and verify 43 named joints.")
    parser.add_argument("--full", action="store_true", help="Compare every package in the version snapshot/lock.")
    args = parser.parse_args()
    sources = json.loads((ROOT / "packaging/sources.json").read_text())
    profile = sources[args.profile]
    expected = {"isaacsim": profile["isaacsim"], "torch": profile["torch"]}
    if args.profile == "main":
        expected.update({"isaaclab": "0.54.3", "numpy": "1.26.0", "rsl-rl-lib": "3.1.2", **sources["compatibility_changes"]})
        expected.update({"isaaclab-assets": "0.2.4", "isaaclab-contrib": "0.0.2", "isaaclab-mimic": "1.0.16",
                         "isaaclab-rl": "0.4.7", "isaaclab-tasks": "0.11.13", "unitree-rl-lab": "0.2.1"})
    else:
        expected.update({"isaaclab": "6.1.17", "isaaclab-arena": "0.3.0", "pin-pink": "3.3.0", "pin": "3.9.0", "numpy": "2.3.1"})
    if args.full:
        snapshot = ROOT / "requirements" / ("main.lock.txt" if args.profile == "main" else "arena.snapshot.txt")
        for line in snapshot.read_text().splitlines():
            value = line.split(" #", 1)[0].strip()
            if value and not value.startswith(("#", "--")):
                req = Requirement(value)
                for spec in req.specifier:
                    if spec.operator in ("==", "==="):
                        expected[req.name] = spec.version
    errors, warnings = dependency_issues(args.profile)
    if platform.python_version() != profile["python"]:
        errors.append(f"Python {platform.python_version()} != {profile['python']}")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        errors.append("This environment profile supports Linux x86_64 only.")
    libc_name, libc_version = platform.libc_ver()
    if libc_name != "glibc" or tuple(int(v) for v in libc_version.split(".")) < (2, 35):
        errors.append(f"Need GLIBC >=2.35, found {libc_name} {libc_version}")
    for name, version in expected.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = "MISSING"
        if installed != version:
            errors.append(f"{name}: {installed} != {version}")
    contaminated = [p for p in sys.path if "/opt/ros/" in p or "/isaacgym/" in p]
    if contaminated:
        errors.append(f"Foreign Python paths: {contaminated}; start a clean shell / use activate.sh")
    if args.assets:
        ros = Path(os.environ.get("UNITREE_ROS_DIR", ROOT / "external/unitree_ros"))
        urdf = ros / "robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf"
        if not urdf.is_file():
            errors.append(f"Missing 43-joint robot description: {urdf}")
    result = {"profile": args.profile, "python": platform.python_version(), "prefix": sys.prefix,
              "errors": errors, "upstream_override_warnings": warnings, "metadata_pass": not errors,
              "simulation_test": "not requested", "wbc_test": "not requested"}
    if args.wbc and not errors:
        if args.profile != "main":
            errors.append("WBC bridge check must be requested from the main environment.")
        else:
            try:
                from unitree_rl_lab.utils.arena_wbc_bridge import ArenaWBCBridge

                with ArenaWBCBridge(startup_timeout_s=180.0) as bridge:
                    if len(bridge.joint_names) != 43:
                        raise ValueError(f"Expected 43 joints, got {len(bridge.joint_names)}")
                result["wbc_test"] = "passed: 43 named joints"
            except Exception as error:
                result["wbc_test"] = "failed"
                errors.append(f"WBC initialization: {error}")
    if args.simulation and not errors:
        if args.profile != "main":
            errors.append("Use the main environment for the standalone simulation smoke test.")
        else:
            # Official docs warn that first-run registry downloads can take
            # >10 min; a short fixed deadline misclassifies a clean install.
            try:
                process = subprocess.run([sys.executable, str(ROOT / "packaging/smoke_simulation.py")], timeout=900)
            except subprocess.TimeoutExpired:
                errors.append("Simulation startup timed out after 900 seconds; check NVIDIA registry/network and GPU logs.")
                result["simulation_test"] = "timed out"
                print(json.dumps(result, indent=2))
                return 1
            result["simulation_test"] = "passed" if process.returncode == 0 else "failed"
            if process.returncode:
                errors.append(f"Simulation subprocess exited {process.returncode}")
    result["metadata_pass"] = not errors
    print(json.dumps(result, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
