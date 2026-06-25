import numpy as np
from scipy.integrate import solve_ivp

from src.dynamics import lander_eom


def closed_loop_rhs(t, state, params):
    """RHS passed to solve_ivp: query controller, saturate actuators, return state derivatives."""
    T, delta = params['controller'](t, state, params)

    # Clamp magnitude while preserving sign, then clamp gimbal angle
    T = np.sign(T) * np.clip(np.abs(T), params['T_min'], params['T_max'])
    delta = np.clip(delta, -params['delta_max'], params['delta_max'])
    state = lander_eom(t, state, T, delta, params)
    return state

def touchdown_event(t, state, params):
    # z = state[1]; integration stops when z crosses zero from above
    return state[1]

def run_sim(state_0, params, t_span, max_step, controller):
    # Terminal + direction=-1: stop only when z is decreasing through zero (descending touchdown)
    touchdown_event.terminal = True
    touchdown_event.direction = -1

    sol = solve_ivp(
        closed_loop_rhs,
        t_span,
        state_0,
        args=(params, controller),  # must be a tuple matching (params, controller) after (t, state)
        events=touchdown_event,
        max_step=max_step
    )

    return sol