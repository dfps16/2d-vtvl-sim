import numpy as np


def lander_eom(t, state, T, delta, params):
    """Equations of motion for a 2D VTVL lander (planar, rigid body).

    Thrust T acts at the base, gimbal angle delta rotates it relative to body axis.
    State: [x, z, xdot, zdot, theta, thetadot] — position, velocity, attitude, rate.
    Returns state derivatives for use with an ODE integrator (e.g. scipy solve_ivp).
    """
    x, z, xdot, zdot, theta, thetadot = state
    m, I, L, g = params['m'], params['I'], params['L'], params['g']

    # Inertial frame accelerations from thrust projected through body + gimbal angles
    xddot = - (T * np.sin(theta - delta)) / m
    zddot = (T * np.cos(theta - delta) - m * g) / m
    # Torque from off-axis thrust: moment arm L from CoM to nozzle
    thetaddot = - T * np.sin(delta) * L / I
    return [xdot, zdot, xddot, zddot, thetadot, thetaddot]