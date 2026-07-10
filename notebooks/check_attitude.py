"""Inner attitude-loop validation harness.

Isolates the gimbal -> pitch PD loop: throttle is pinned at hover so the lander
floats vertically while we command a pitch step and watch theta track it. The
outer position loop is NOT closed here, so horizontal drift is expected and
ignored — we judge attitude tracking only.

Plots are built around the four acceptance criteria for the inner loop:
  1. theta settles to theta_cmd, overshoot consistent with the design zeta (~1.5%)
  2. no steady-state theta error
  3. delta swings NEGATIVE first on a +theta_cmd step (plant sign check)
  4. on a large step, delta clips at -delta_max then comes off saturation

Two scenarios are run: a 2 deg step (stays linear) and an 8 deg step (saturates).
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Make ``vtvl_sim`` (under src/) importable when run as a script from any cwd.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from vtvl_sim.controllers import AttitudePDController
from vtvl_sim.params import PARAMS, PD_GAINS
from vtvl_sim.sim import run_sim

# Recover the design targets from the gains (single source of truth in params):
#   omega_n = sqrt(kp),  zeta = kd / (2*sqrt(kp))
OMEGA_N = np.sqrt(PD_GAINS['kp'])
ZETA = PD_GAINS['kd'] / (2 * np.sqrt(PD_GAINS['kp']))

# Predicted linear-design step metrics, drawn as reference lines.
PREDICTED_OS = np.exp(-ZETA * np.pi / np.sqrt(1 - ZETA**2)) * 100  # % overshoot
PREDICTED_TS = 4.0 / (ZETA * OMEGA_N)                              # 2% settling time [s]
SETTLE_BAND = 0.02  # +/-2% acceptance band

T_HOVER = PARAMS['m'] * PARAMS['g']
DELTA_MAX_DEG = np.degrees(PARAMS['delta_max'])


def run_step(theta_cmd_deg, t_end=6.0, max_step=0.005):
    """Run one fixed-throttle pitch-step scenario.

    Returns (t, theta_deg, delta_deg, theta_cmd_deg). delta is recomputed
    post-hoc from the solution states — the PD controller is stateless, so this
    is exact and avoids logging inside the RHS.
    """
    theta_cmd = np.radians(theta_cmd_deg)
    attitude = AttitudePDController(PD_GAINS, theta_cmd)

    def controller(t, state, params):
        # Fixed-throttle wrapper adapting AttitudePD to the (t, state, params)
        # contract: hold hover thrust, command gimbal from the inner loop.
        delta = attitude(theta_cmd, state, params, T_HOVER)
        return (T_HOVER, delta)

    state_0 = [0.0, 100.0, 0.0, 0.0, 0.0, 0.0]  # level, at rest
    sol = run_sim(state_0, PARAMS, (0.0, t_end), max_step, controller)

    t = sol.t
    theta = sol.y[4]
    delta = np.array([
        attitude(theta_cmd, sol.y[:, i], PARAMS, T_HOVER) for i in range(t.size)
    ])
    return t, np.degrees(theta), np.degrees(delta), theta_cmd_deg


def metrics(t, theta_deg, delta_deg, cmd_deg):
    """Acceptance numbers for one scenario."""
    peak = theta_deg.max()
    overshoot = (peak - cmd_deg) / cmd_deg * 100 if cmd_deg != 0 else 0.0

    # Settling time: last instant theta is outside the +/-2% band.
    band = SETTLE_BAND * cmd_deg
    outside = np.where(np.abs(theta_deg - cmd_deg) > band)[0]
    settle_t = t[outside[-1] + 1] if outside.size and outside[-1] + 1 < t.size else 0.0

    ss_error = theta_deg[-1] - cmd_deg

    # First commanded deflection of meaningful size — checks the sign convention.
    nz = np.where(np.abs(delta_deg) > 1e-6)[0]
    first_sign = np.sign(delta_deg[nz[0]]) if nz.size else 0.0
    saturated = np.abs(delta_deg).max() >= DELTA_MAX_DEG - 1e-3

    return dict(peak=peak, overshoot=overshoot, settle_t=settle_t,
                ss_error=ss_error, first_sign=first_sign, saturated=saturated)


SCENARIOS = [('2 deg step (linear)', 2.0), ('8 deg step (saturating)', 8.0)]

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

for col, (label, cmd_deg) in enumerate(SCENARIOS):
    t, theta_deg, delta_deg, cmd = run_step(cmd_deg)
    m = metrics(t, theta_deg, delta_deg, cmd)

    # ---- Row 0: pitch tracking + acceptance markers ----
    ax = axes[0, col]
    ax.plot(t, theta_deg, color='tab:blue', label='theta')
    ax.axhline(cmd, color='k', linestyle='--', linewidth=1, label='theta_cmd')
    ax.axhspan(cmd * (1 - SETTLE_BAND), cmd * (1 + SETTLE_BAND),
               color='green', alpha=0.12, label='+/-2% band')
    ax.axhline(cmd * (1 + PREDICTED_OS / 100), color='tab:orange', linestyle=':',
               linewidth=1, label=f'predicted OS {PREDICTED_OS:.1f}%')
    if m['settle_t'] > 0:
        ax.axvline(m['settle_t'], color='tab:red', linestyle='-.', linewidth=1,
                   label=f"settle {m['settle_t']:.2f}s")
    ax.set_title(label)
    ax.set_ylabel('pitch theta [deg]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)
    ax.text(0.02, 0.97,
            f"overshoot {m['overshoot']:.2f}%\nss error {m['ss_error']:+.3f} deg",
            transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    # ---- Row 1: gimbal command + saturation limits ----
    ax = axes[1, col]
    ax.plot(t, delta_deg, color='tab:purple', label='delta')
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.axhline(DELTA_MAX_DEG, color='gray', linestyle=':', linewidth=1, label='+/-delta_max')
    ax.axhline(-DELTA_MAX_DEG, color='gray', linestyle=':', linewidth=1)
    ax.set_ylabel('gimbal delta [deg]')
    ax.set_xlabel('time [s]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    sign_txt = 'negative first OK' if m['first_sign'] < 0 else 'NOT negative first!'
    sat_txt = 'saturates' if m['saturated'] else 'no saturation'
    ax.text(0.02, 0.05, f"{sign_txt}\n{sat_txt}", transform=ax.transAxes,
            va='bottom', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    print(f"[{label}]  peak={m['peak']:.3f} deg  overshoot={m['overshoot']:.2f}%  "
          f"settle={m['settle_t']:.2f}s  ss_err={m['ss_error']:+.3f} deg  "
          f"delta_first_sign={m['first_sign']:+.0f}  saturated={m['saturated']}")

fig.suptitle(f'Inner attitude loop — acceptance criteria '
             f'(omega_n={OMEGA_N:.1f} rad/s, zeta={ZETA:.2f})', fontsize=13)
fig.tight_layout()
fig.savefig('results/check_attitude.png', dpi=150)
plt.close(fig)
