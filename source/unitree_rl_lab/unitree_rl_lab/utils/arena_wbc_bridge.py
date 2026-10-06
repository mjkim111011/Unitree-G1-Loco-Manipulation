"""Process bridge to Arena's G1 Pink/Homie teacher controller.

The bridge keeps Isaac Lab 0.54/Isaac Sim 5.1 in the caller and Arena 0.3/
Isaac Sim 6.0 in an isolated Python process.  It exchanges only JSON numeric
arrays, avoiding imports across incompatible simulator ABIs.
"""

from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from unitree_rl_lab.runtime_paths import default_arena_root


PROTOCOL_PREFIX = "ARENA_WBC_JSON:"


class ArenaWBCBridgeError(RuntimeError):
    """Raised when the isolated Arena controller fails or times out."""


class ArenaWBCBridge:
    """Persistent JSON-lines client for :mod:`scripts.teacher.arena_wbc_worker`."""

    def __init__(
        self,
        arena_root: str | os.PathLike[str] | None = None,
        worker_path: str | os.PathLike[str] | None = None,
        startup_timeout_s: float = 45.0,
        step_timeout_s: float = 5.0,
    ) -> None:
        self.arena_root = Path(arena_root or default_arena_root()).expanduser().resolve()
        self.python = self.arena_root / ".venv" / "bin" / "python"
        if worker_path is None:
            repository_root = Path(os.environ.get("UNITREE_WORKSPACE", Path(__file__).resolve().parents[4]))
            worker_path = repository_root / "scripts" / "teacher" / "arena_wbc_worker.py"
        self.worker_path = Path(worker_path).expanduser().resolve()
        if not math.isfinite(startup_timeout_s) or startup_timeout_s <= 0.0:
            raise ValueError("startup_timeout_s must be finite and greater than zero")
        if not math.isfinite(step_timeout_s) or step_timeout_s <= 0.0:
            raise ValueError("step_timeout_s must be finite and greater than zero")
        self.step_timeout_s = float(step_timeout_s)

        if not self.python.is_file():
            raise FileNotFoundError(f"Arena Python was not found: {self.python}")
        if not self.worker_path.is_file():
            raise FileNotFoundError(f"Arena WBC worker was not found: {self.worker_path}")

        environment = os.environ.copy()
        # Do not leak the caller's Isaac Lab source paths into Arena's Python.
        # The worker must resolve every simulator/controller dependency from
        # Arena's own venv and checkout for the process boundary to be useful.
        environment["PYTHONPATH"] = str(self.arena_root)
        environment["PATH"] = os.pathsep.join(
            (str(self.python.parent), environment.get("PATH", os.defpath))
        )
        environment.pop("PYTHONHOME", None)
        # The caller commonly has Isaac Sim, ROS, or an older Isaac Gym on its
        # native-library search path.  Those libraries are just as ABI-sensitive
        # as Python modules, so let Arena's venv/package rpaths resolve them.
        environment.pop("LD_LIBRARY_PATH", None)
        environment.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        environment.setdefault("ACCEPT_EULA", "Y")
        self._process = subprocess.Popen(
            [str(self.python), str(self.worker_path)],
            cwd=str(self.arena_root),
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()
        self._stdout_thread = threading.Thread(target=self._drain_stdout, daemon=True)
        self._stdout_thread.start()
        try:
            ready = self._read_protocol_message(float(startup_timeout_s))
            if not ready.get("ok") or not ready.get("ready"):
                raise ArenaWBCBridgeError(f"Arena controller initialization failed: {ready}")
            ready_names = self._validate_joint_names(ready.get("joint_names"), "startup")
        except BaseException:
            self.close()
            raise
        self.joint_names = ready_names

    @property
    def is_alive(self) -> bool:
        return self._process.poll() is None

    def _drain_stdout(self) -> None:
        """Continuously drain worker output without TextIO/select buffering races."""
        if self._process.stdout is None:
            self._stdout_queue.put(None)
            return
        try:
            for line in self._process.stdout:
                self._stdout_queue.put(line.rstrip())
        finally:
            self._stdout_queue.put(None)

    def _read_protocol_message(self, timeout_s: float) -> dict[str, Any]:
        if self._process.stdout is None:
            raise ArenaWBCBridgeError("Arena worker stdout is unavailable")
        deadline = time.monotonic() + timeout_s
        diagnostic_lines: list[str] = []
        while time.monotonic() < deadline:
            if self._process.poll() is not None and self._stdout_queue.empty():
                raise ArenaWBCBridgeError(
                    f"Arena worker exited with code {self._process.returncode}: "
                    + "\n".join(diagnostic_lines[-20:])
                )
            wait_s = max(0.0, min(0.25, deadline - time.monotonic()))
            try:
                line = self._stdout_queue.get(timeout=wait_s)
            except queue.Empty:
                continue
            if line is None:
                continue
            if line.startswith(PROTOCOL_PREFIX):
                try:
                    payload = json.loads(line[len(PROTOCOL_PREFIX) :])
                except json.JSONDecodeError as exc:
                    raise ArenaWBCBridgeError(f"Arena worker emitted invalid protocol JSON: {line}") from exc
                if not isinstance(payload, dict):
                    raise ArenaWBCBridgeError(
                        f"Arena worker protocol payload must be an object, got {type(payload).__name__}"
                    )
                return payload
            diagnostic_lines.append(line)
        raise ArenaWBCBridgeError(
            f"Timed out after {timeout_s:.1f}s waiting for Arena worker. "
            f"Last output: {diagnostic_lines[-10:]}"
        )

    @staticmethod
    def _validate_joint_names(value: Any, stage: str) -> list[str]:
        if not isinstance(value, list) or len(value) != 43:
            length = len(value) if isinstance(value, list) else "not-a-list"
            raise ArenaWBCBridgeError(f"Expected 43 joint names at {stage}, got {length}")
        if any(not isinstance(name, str) or not name for name in value):
            raise ArenaWBCBridgeError(f"Arena worker returned a non-string or empty joint name at {stage}")
        names = list(value)
        if len(set(names)) != len(names):
            raise ArenaWBCBridgeError(f"Arena worker returned duplicate joint names at {stage}")
        return names

    def step(self, request: dict[str, Any]) -> tuple[list[str], list[float]]:
        """Return an absolute 43-D named joint target for one high-level action."""
        if not self.is_alive:
            raise ArenaWBCBridgeError(f"Arena worker is not running (code={self._process.returncode})")
        self._write_message(json.dumps(request, separators=(",", ":")) + "\n", "step")
        response = self._read_protocol_message(self.step_timeout_s)
        if not response.get("ok"):
            raise ArenaWBCBridgeError(
                f"Arena controller step failed: {response.get('error')}\n{response.get('traceback', '')}"
            )
        names = self._validate_joint_names(response.get("joint_names"), "step")
        if set(names) != set(self.joint_names):
            raise ArenaWBCBridgeError("Arena worker changed its joint-name set after startup")
        target_value = response.get("joint_target")
        if not isinstance(target_value, list) or len(target_value) != 43:
            length = len(target_value) if isinstance(target_value, list) else "not-a-list"
            raise ArenaWBCBridgeError(f"Expected 43 joint targets, got {length}")
        try:
            target = [float(value) for value in target_value]
        except (TypeError, ValueError) as exc:
            raise ArenaWBCBridgeError("Arena worker returned a non-numeric joint target") from exc
        if not all(math.isfinite(value) for value in target):
            raise ArenaWBCBridgeError("Arena worker returned a non-finite joint target")
        return names, target

    def reset(self) -> None:
        """Reset Arena's recurrent locomotion history and Pink warm-up state."""
        if not self.is_alive:
            raise ArenaWBCBridgeError("Arena worker is not running")
        self._write_message('{"command":"reset"}\n', "reset")
        response = self._read_protocol_message(self.step_timeout_s)
        if not response.get("ok") or not response.get("reset"):
            raise ArenaWBCBridgeError(f"Arena controller reset failed: {response}")

    def _write_message(self, message: str, stage: str) -> None:
        """Write one protocol line while preserving the bridge error contract."""
        stdin = self._process.stdin
        if stdin is None:
            raise ArenaWBCBridgeError("Arena worker stdin is unavailable")
        try:
            stdin.write(message)
            stdin.flush()
        except (OSError, ValueError) as exc:
            raise ArenaWBCBridgeError(
                f"Could not write Arena controller {stage} request "
                f"(worker code={self._process.poll()})"
            ) from exc

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write('{"command":"close"}\n')
                process.stdin.flush()
            process.wait(timeout=3.0)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            try:
                process.terminate()
            except OSError:
                pass
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
                process.wait(timeout=3.0)

    def __enter__(self) -> "ArenaWBCBridge":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
