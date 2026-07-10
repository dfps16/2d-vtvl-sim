import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

from vtvl_sim.dynamics import lander_eom

# Make ``src`` importable when run as a script from any cwd.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def closed_loop_rhs(t, state, sim_setup, controller):
    """RHS passed to solve_ivp: query controller, saturate actuators, return state derivatives."""
    params = sim_setup['params']
    T, delta = controller(t, state, params)
    # Clamp magnitude, then clamp gimbal angle
    T = np.clip(T, params['T_min'], params['T_max'])
    delta = np.clip(delta, -params['delta_max'], params['delta_max'])
    state_dot = lander_eom(t, state, T, delta, params)
    return state_dot

def touchdown_event(t, state, sim_setup, controller):
    z = state[1] # integration stops when z crosses zero from above
    touchdown = z - sim_setup['landing_tolerance']
    return touchdown
 
def vtvl_solver(state_0, x_target, z_target, t_end, sim_setup, solver_setup):
    
    # Extracting the simulation setup
    PARAMS = sim_setup['params']
    GAINS = sim_setup['gains']
    SELECTED_CONTROLLER = sim_setup['controller']

    # Extracting the solver setup
    max_step = solver_setup['max_step']
    method = solver_setup['method']

    controller = SELECTED_CONTROLLER['build'](GAINS, x_target, z_target)

    # Setting up termination event(s)
    # Terminal + direction=-1: stop only when z is decreasing through zero (descending touchdown)
    touchdown_event.terminal = True
    touchdown_event.direction = -1
    
    sol = solve_ivp(
        closed_loop_rhs,
        (0.0, t_end),
        state_0,
        args=(sim_setup, controller),  # must be a tuple matching (params, controller) after (t, state)
        events=touchdown_event,
        max_step=max_step,
        method=method
    )

    t = sol.t
    x = sol.y[0]
    z = sol.y[1]
    xdot = sol.y[2]
    zdot = sol.y[3]
    theta = sol.y[4]
    thetadot = sol.y[5]

    theta_cmd = np.empty(t.size)
    u_T = np.empty(t.size)
    delta = np.empty(t.size)

    xddot_des = np.empty(t.size)
    zddot_des = np.empty(t.size)
    thetaddot_des = np.empty(t.size)

    for i in range(t.size):
        s = sol.y[:, i]
        theta_cmd[i], xddot_des[i] = controller.commanded_tilt(s, PARAMS)
        u_T[i], zddot_des[i] = controller.commanded_thrust(s, PARAMS)
        delta[i], thetaddot_des[i] = controller.commanded_thrust_vector(s, PARAMS, theta_cmd[i], u_T[i])

    results = {
        't': t,
        'x': x,
        'z': z,
        'xdot': xdot,
        'zdot': zdot,
        'theta': theta,
        'thetadot': thetadot,
        'theta_cmd': theta_cmd,
        'u_T': u_T,
        'delta': delta,
        'xddot_des': xddot_des,
        'zddot_des': zddot_des,
        'thetaddot_des': thetaddot_des
    }

    return results

def sim_run(sim_setup, solver_setup):
    segments = []  # collects results after each phase
    state_n = list(sim_setup['initial_state'])  # state after each phase
    time_elapsed = 0.0  # time elapsed since start of sim

    # Loop through each phase, solve the dynamics, append the results
    for x_target, z_target, t_end in sim_setup['phases']:
        seg = vtvl_solver(state_n, x_target, z_target, t_end, sim_setup, solver_setup)
        seg['t'] = seg['t'] + time_elapsed
        segments.append(seg)
        state_n = [seg[k][-1] for k in ('x', 'z', 'xdot', 'zdot', 'theta', 'thetadot')]
        time_elapsed = seg['t'][-1]

    return {
        key: np.concatenate([segments[0][key], *(s[key][1:] for s in segments[1:])])
        
        for key in segments[0]
        }