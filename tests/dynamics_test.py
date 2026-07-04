import numpy as np
from scipy.integrate import solve_ivp
from src.dynamics import lander_eom
from src.params import PARAMS


def test_free_fall():
    """Zero thrust should reduce exactly to projectile motion."""
    T, delta = 0.0, 0.0
    x0, z0 = 0.0, 100.0
    xdot0, zdot0 = 5.0, 0.0
    state0 = [x0, z0, xdot0, zdot0, 0.0, 0.0]

    t_span = (0, 4.0)
    t_eval = np.linspace(*t_span, 200)

    sol = solve_ivp(
        lander_eom, t_span, state0,
        args=(T, delta, PARAMS),
        t_eval=t_eval, rtol=1e-9, atol=1e-9
    )

    t = sol.t
    x_an = x0 + xdot0 * t
    z_an = z0 + zdot0 * t - 0.5 * PARAMS['g'] * t**2

    assert np.max(np.abs(sol.y[0] - x_an)) < 1e-6
    assert np.max(np.abs(sol.y[1] - z_an)) < 1e-6


def test_hover_equilibrium():
    """T = mg, theta = delta = 0 should hold the state constant."""
    m, g = PARAMS['m'], PARAMS['g']
    T, delta = m * g, 0.0
    state0 = [0.0, 100.0, 0.0, 0.0, 0.0, 0.0]

    sol = solve_ivp(
        lander_eom, (0, 10.0), state0,
        args=(T, delta, PARAMS),
        t_eval=np.linspace(0, 10.0, 200),
        rtol=1e-9, atol=1e-9
    )

    drift = np.abs(sol.y[:, -1] - np.array(state0))
    assert np.max(drift) < 1e-4