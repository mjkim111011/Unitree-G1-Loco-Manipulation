"""Duration of the loaded WBC bridge between BC2 (grasp-lift) and BC3 (carry).

The baseline bridge is 7 s: a 2 s smoothstep from BC2's last target into the
Arena WBC, then a 5 s loaded stand under the WBC.  A shorter bridge first
shortens the stand and then the WBC interpolation itself:

    bridge   WBC interpolation   loaded stand
    7.0 s    2.0 s               5.0 s        (baseline)
    2.0 s    2.0 s               0.0 s
    0.5 s    0.5 s               0.0 s
    0.0 s    -                   -            (no WBC: BC3 blends directly from BC2's last target)

With a zero bridge the WBC is never queried; the BC3 blend
(``--carry_blend_steps``) then starts from BC2's final target instead of the
WBC's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


BASELINE_BRIDGE_SECONDS = 7.0
BASELINE_INTERPOLATION_SECONDS = 2.0


@dataclass(frozen=True)
class WbcBridgeSchedule:
    interpolation_steps: int
    stand_steps: int

    @property
    def total_steps(self) -> int:
        return self.interpolation_steps + self.stand_steps

    @property
    def uses_wbc(self) -> bool:
        return self.total_steps > 0

    def as_dict(self, step_dt: float) -> dict:
        return {
            "wbc_bridge_seconds": self.total_steps * step_dt,
            "wbc_interpolation_steps": self.interpolation_steps,
            "wbc_stand_steps": self.stand_steps,
            "uses_wbc": self.uses_wbc,
        }


def schedule_for(bridge_seconds: float, step_dt: float,
                 interpolation_seconds: float = BASELINE_INTERPOLATION_SECONDS) -> WbcBridgeSchedule:
    """Split a total bridge time into WBC interpolation and loaded-stand control steps."""

    if not math.isfinite(bridge_seconds) or bridge_seconds < 0.0:
        raise ValueError("bridge_seconds must be finite and nonnegative")
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError("step_dt must be finite and positive")
    if not math.isfinite(interpolation_seconds) or interpolation_seconds <= 0.0:
        raise ValueError("interpolation_seconds must be finite and positive")
    total = int(round(bridge_seconds / step_dt))
    if total and not math.isclose(total * step_dt, bridge_seconds, abs_tol=1e-9):
        raise ValueError(f"bridge_seconds {bridge_seconds} is not a whole number of {step_dt} s control steps")
    interpolation = min(total, int(round(interpolation_seconds / step_dt)))
    return WbcBridgeSchedule(interpolation_steps=interpolation, stand_steps=total - interpolation)


def wbc_interpolation_weight(step: int, schedule: WbcBridgeSchedule) -> float:
    """Smoothstep weight of the WBC target over BC2's last target at a bridge step (1 once in the stand)."""

    if step < 0:
        raise ValueError("step must be nonnegative")
    if step >= schedule.interpolation_steps:
        return 1.0
    fraction = (step + 1) / float(schedule.interpolation_steps)
    return fraction * fraction * (3.0 - 2.0 * fraction)
