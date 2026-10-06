"""Small CLI that does not import Isaac Sim before its app is initialized."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


COMMANDS = {
    "doctor": "packaging/verify_environment.py",
    "train-bc": "scripts/teacher/phase_bc.py",
    "fsm": "scripts/teacher/play_bc_skill_fsm.py",
    "teacher": "scripts/teacher/collect_carry_place_teacher.py",
    "dagger": "scripts/teacher/collect_dagger_corrections.py",
}


def find_workspace(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "scripts/teacher/phase_bc.py").is_file():
            return path
    raise FileNotFoundError("Run inside the cloned repository or set UNITREE_WORKSPACE.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Unitree G1 workspace commands; provision Isaac dependencies first.")
    parser.add_argument("command", choices=COMMANDS)
    # Only parse our first token. Forward '--help' and all options after the
    # command to the selected training/verification script, unchanged.
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    try:
        root = find_workspace(Path(os.environ.get("UNITREE_WORKSPACE", Path.cwd())).resolve())
    except FileNotFoundError as error:
        parser.error(str(error))
    return subprocess.call([sys.executable, str(root / COMMANDS[args.command]), *remaining], cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
