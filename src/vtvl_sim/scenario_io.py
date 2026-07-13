"""Load and validate scenario JSON files into the sim_setup/solver_setup dicts
that sim_run/vtvl_solver/write_sim_report expect.
"""

import json

import numpy as np

from vtvl_sim.controllers import CONTROLLER_REGISTRY
from vtvl_sim.schemas import Outputs, ScenarioSetup, SolverSetup


def build_setup(raw_sim_setup, raw_solver_setup):
    """Validate raw dicts (from JSON or a GUI) into the sim_setup/solver_setup
    shape that sim_run expects. No file I/O — usable from anywhere."""
    scenario = ScenarioSetup.model_validate(raw_sim_setup)
    solver = SolverSetup.model_validate(raw_solver_setup)

    params = scenario.params.model_dump(exclude={'delta_max_deg', 'tilt_limit_deg'})
    params['delta_max'] = np.radians(scenario.params.delta_max_deg)
    params['tilt_limit'] = np.radians(scenario.params.tilt_limit_deg)

    sim_setup = {
        'params': params,
        'gains': scenario.gains,
        'controller': CONTROLLER_REGISTRY[scenario.controller_name],
        'phases': [(p.x_target, p.z_target, p.t_end) for p in scenario.phases],
        'initial_state': scenario.initial_state.to_list(),
        'landing_tolerance': scenario.landing_tolerance,
    }
    return sim_setup, solver.model_dump()


def load_scenario(path):
    with open(path) as f:
        raw = json.load(f)

    sim_setup, solver_setup = build_setup(raw['sim_setup'], raw['solver_setup'])
    outputs_setup = Outputs.model_validate(raw['outputs']).model_dump()
    return sim_setup, solver_setup, outputs_setup