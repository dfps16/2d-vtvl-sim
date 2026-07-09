"""Inner attitude-loop regression tests.

Closed-loop counterpart to dynamics_test.py: dynamics_test validates the open-loop
plant, this validates the AttitudePDController wrapped around it. Encodes the Week 2
acceptance criteria as pass/fail assertions. See tests/attitude_test_plan.md for the
architecture and rationale.

Fixture (matches notebooks/attitude_loop.py):
  - throttle pinned at hover (T = m*g); vehicle floats while attitude is exercised
  - level, at-rest start; a pitch-step command
  - delta recomputed post-hoc from the solution states (controller is stateless -> exact)
  - gains/targets pulled from params.py (single source of truth)
"""

import numpy as np

from src.controllers import AttitudePDController
from src.params import PARAMS, PD_GAINS
from src.sim import run_sim

# Design targets recovered from the gains: omega_n = sqrt(kp), zeta = kd/(2*sqrt(kp)).
ZETA = PD_GAINS['kd'] / (2 * np.sqrt(PD_GAINS['kp']))
PREDICTED_OS = np.exp(-ZETA * np.pi / np.sqrt(1 - ZETA**2)) * 100  # % overshoot

T_HOVER = PARAMS['m'] * PARAMS['g']
DELTA_MAX_DEG = np.degrees(PARAMS['delta_max'])

# Small step stays in the linear regime (no gimbal saturation) so linear-theory
# predictions apply; large step deliberately saturates the gimbal.
SMALL_STEP_DEG = 1.5
LARGE_STEP_DEG = 8.0


def _run_step(theta_cmd_deg, t_end=6.0, max_step=0.005):
    """Simulate one fixed-throttle pitch step; return (t, theta_deg, delta_deg, cmd_deg)."""
    theta_cmd = np.radians(theta_cmd_deg)
    attitude = AttitudePDController(PD_GAINS, theta_cmd)

    def controller(t, state, params):
        return (T_HOVER, attitude(theta_cmd, state, params, T_HOVER))

    state_0 = [0.0, 100.0, 0.0, 0.0, 0.0, 0.0]  # level, at rest
    sol = run_sim(state_0, PARAMS, (0.0, t_end), max_step, controller)

    t = sol.t
    theta_deg = np.degrees(sol.y[4])
    # Stateless controller -> recomputing delta from each state is exact.
    delta_deg = np.degrees(
        [attitude(theta_cmd, sol.y[:, i], PARAMS, T_HOVER) for i in range(t.size)]
    )
    return t, theta_deg, delta_deg, theta_cmd_deg


def test_no_steady_state_error():
    """Criterion 1: final pitch reaches the command (PD, no offset)."""
    _, theta_deg, _, cmd = _run_step(SMALL_STEP_DEG)
    assert abs(theta_deg[-1] - cmd) < 0.05  # deg


def test_overshoot_matches_linear_design():
    """Criterion 2: overshoot equals the second-order prediction for the design zeta.

    The strong check: stability alone is weak; matching the predicted overshoot
    confirms the gains realise the intended pole locations.
    """
    _, theta_deg, _, cmd = _run_step(SMALL_STEP_DEG)
    overshoot = (theta_deg.max() - cmd) / cmd * 100
    assert abs(overshoot - PREDICTED_OS) < 0.5  # percentage points


def test_gimbal_sign_convention():
    """Criterion 3: a +theta_cmd step drives delta negative first (plant-sign guard)."""
    _, _, delta_deg, _ = _run_step(SMALL_STEP_DEG)
    first = delta_deg[np.abs(delta_deg) > 1e-6][0]
    assert first < 0


def test_gimbal_saturates_on_large_step():
    """Criterion 4: a large step drives the gimbal onto its travel limit."""
    _, _, delta_deg, _ = _run_step(LARGE_STEP_DEG)
    assert np.isclose(np.abs(delta_deg).max(), DELTA_MAX_DEG, atol=1e-3)


def test_gimbal_never_exceeds_limit():
    """Actuator model: commanded delta is always within +/- delta_max."""
    _, _, delta_deg, _ = _run_step(LARGE_STEP_DEG)
    assert np.all(np.abs(delta_deg) <= DELTA_MAX_DEG + 1e-6)
