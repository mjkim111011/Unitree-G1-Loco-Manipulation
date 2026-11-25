"""Experimental capture-point-conditioned walking-command attenuation.

Keeps fixed joint interpolation and grasp targets intact. The two-foot
rectangle hull is a geometric envelope, NOT a proof of dynamic stability.
Measurement remains in feedback; there is no BC3 posture extrapolation/lead.
"""
from dataclasses import dataclass
import math

import torch

from .com_stabilizer import ComStabilizer
from .com_predictive_blend import signed_support_margin
from .transition_diagnostics import support_polygon


@dataclass(frozen=True)
class CommandGuardCfg:
    step_dt: float = .02
    active_seconds: float = 3.0
    outside_deadband_m: float = .02
    full_attenuation_outside_m: float = .10
    minimum_scale: float = .2
    attenuation_rate_per_second: float = 4.0
    recovery_rate_per_second: float = 1.0

    def validate(self):
        if any(not math.isfinite(x) or x <= 0 for x in (self.step_dt, self.active_seconds,
                self.full_attenuation_outside_m, self.attenuation_rate_per_second, self.recovery_rate_per_second)):
            raise ValueError('Guard time, distance, and rates must be positive')
        if not 0 <= self.outside_deadband_m < self.full_attenuation_outside_m:
            raise ValueError('Guard deadband must be below full attenuation distance')
        if not 0 <= self.minimum_scale <= 1:
            raise ValueError('Guard command scale must be in [0,1]')
        return self


def command_scale(previous, margin, carry_step, cfg):
    """No changes inside the envelope; bounded attenuation and recovery outside."""
    cfg.validate()
    active = (carry_step*cfg.step_dt < cfg.active_seconds and margin is not None and math.isfinite(margin))
    risk = (max(0., min(1., (-margin-cfg.outside_deadband_m)/
            (cfg.full_attenuation_outside_m-cfg.outside_deadband_m))) if active else 0.)
    target = 1.-(1.-cfg.minimum_scale)*risk
    rate = cfg.attenuation_rate_per_second if target < previous else cfg.recovery_rate_per_second
    delta = max(-rate*cfg.step_dt, min(rate*cfg.step_dt, target-previous))
    return max(cfg.minimum_scale, min(1., previous+delta)), target


class CaptureCommandGuard:
    def __init__(self, env, locked, cfg=None):
        self.cfg = (cfg or CommandGuardCfg(step_dt=env.step_dt)).validate()
        self.model = ComStabilizer(env, locked, active=False)
        self.scale = 1.

    def step(self, carry_step):
        model = self.model
        state = model.measure()
        feet = model.robot.data.body_link_pos_w[0, model.feet]
        origin = (state['weights'][:, None]*feet).sum(0)
        # Include both feet so ordinary single-support stepping does not
        # instantly trigger from the smaller current stance polygon alone.
        polygon = support_polygon(feet, model.robot.data.body_link_quat_w[0, model.feet],
            [True, True], origin, state['rotation'])
        margin = float(signed_support_margin(state['capture_xy'][None], polygon)[0])
        self.scale, target = command_scale(self.scale, margin, carry_step, self.cfg)
        return self.scale, {'capture_xy_m': state['capture_xy'].tolist(),
            'two_foot_envelope_margin_m': margin, 'command_scale': self.scale,
            'target_scale': target, 'mode': state['mode'], 'phase': 'pre_step_measurement',
            'active_window': carry_step*self.cfg.step_dt < self.cfg.active_seconds}
