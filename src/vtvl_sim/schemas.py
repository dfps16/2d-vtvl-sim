"""Pydantic models for validating scenario JSON files before they reach the solver.

These validate scenario *setup* once per run (initial state, targets, gains,
solver options) — not the per-step ODE state passed through solve_ivp, which
stays a plain list/array for speed.
"""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    model_validator,
)

from src.controllers import CONTROLLER_REGISTRY


class LanderState(BaseModel):
    model_config = ConfigDict(extra='forbid')

    x: float
    z: NonNegativeFloat
    xdot: float
    zdot: float
    theta: float
    thetadot: float

    def to_list(self) -> list[float]:
        return [self.x, self.z, self.xdot, self.zdot, self.theta, self.thetadot]


class Phase(BaseModel):
    model_config = ConfigDict(extra='forbid')

    x_target: float
    z_target: NonNegativeFloat
    t_end: PositiveFloat


class ParamsSchema(BaseModel):
    model_config = ConfigDict(extra='forbid')

    m: PositiveFloat
    I: PositiveFloat
    L: PositiveFloat
    g: PositiveFloat
    T_max: PositiveFloat
    T_min: NonNegativeFloat
    isp: PositiveFloat
    delta_max_deg: float = Field(gt=0, le=90)
    tilt_limit_deg: float = Field(gt=0, le=90)

    @model_validator(mode='after')
    def check_thrust_bounds(self):
        if self.T_min >= self.T_max:
            raise ValueError(f'T_min ({self.T_min}) must be < T_max ({self.T_max})')
        return self


class ScenarioSetup(BaseModel):
    model_config = ConfigDict(extra='forbid')

    params: ParamsSchema
    controller_name: str
    gains: dict[str, float]
    phases: list[Phase] = Field(min_length=1)
    initial_state: LanderState
    landing_tolerance: PositiveFloat

    @model_validator(mode='after')
    def check_controller_and_gains(self):
        if self.controller_name not in CONTROLLER_REGISTRY:
            raise ValueError(
                f'unknown controller_name {self.controller_name!r}, '
                f'available: {list(CONTROLLER_REGISTRY)}'
            )
        required = set(CONTROLLER_REGISTRY[self.controller_name]['gain_fields'])
        missing = required - self.gains.keys()
        if missing:
            raise ValueError(
                f'gains missing required fields for {self.controller_name!r}: {sorted(missing)}'
            )
        return self


class SolverSetup(BaseModel):
    model_config = ConfigDict(extra='forbid')

    max_step: PositiveFloat
    method: Literal['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']


class Outputs(BaseModel):
    model_config = ConfigDict(extra='forbid')

    trajectory: Literal[1, 0]
    state: Literal[1, 0]
    animation: Literal[1, 0]
    report: Literal[1, 0]
    csv: Literal[1, 0]