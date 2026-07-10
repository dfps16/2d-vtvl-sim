"""Load and validate scenario JSON files into the sim_setup/solver_setup dicts
that sim_run/vtvl_solver/write_sim_report expect.
"""

import json

import numpy as np

from src.controllers import CONTROLLER_REGISTRY
from src.schemas import Outputs, ScenarioSetup, SolverSetup


def load_scenario(path):
    with open(path) as f:
        raw = json.load(f)

    scenario = ScenarioSetup.model_validate(raw['sim_setup'])
    solver = SolverSetup.model_validate(raw['solver_setup'])
    outputs = Outputs.model_validate(raw['outputs'])

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
    solver_setup = solver.model_dump()

    outputs_setup = {
        'trajectory': outputs.trajectory,
        'state': outputs.state,
        'animation': outputs.animation,
        'report': outputs.report,
        'csv': outputs.csv,
    }

    return sim_setup, solver_setup, outputs_setup