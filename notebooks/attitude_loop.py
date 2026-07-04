import matplotlib.pyplot as plt
import numpy as np
from src.controllers import AttitudePDController
from src.params import PARAMS, SYS_PROP_ATT
from src.paths import result_path
from src.sim import run_sim

# Store design targets from params.py
ZETA = SYS_PROP_ATT['zeta']
OMEGA_N = SYS_PROP_ATT['omega_n']

# Calculate PD gains from target parameters - assumes idealised plant
PD_GAINS = {
    'kp': OMEGA_N ** 2,
    'kd': 2 * ZETA * OMEGA_N,
}

# Predict time properties using linear-design assumptions
OS_PRED = np.exp(- ZETA * np.pi / np.sqrt(1 - ZETA**2)) * 100 # % overshoot
T_S_PRED = 4.0 / (ZETA * OMEGA_N) # (s) settling time to within 2%

# Calculated parameters from vehicle properties
T_HOVER = PARAMS['m'] * PARAMS['g']  # (N), weight of the vehicle hence T_hover
DELTA_MAX_DEG = np.degrees(PARAMS['delta_max'])  # (deg) max gimbal angle

def pitch_step_scenario(theta_target_deg, t_end=6.0, max_step=0.005, T_fixed=None):
    # T_fixed lets us hold a throttle other than hover (used by the thrust-
    # robustness figure); the inversion inside the controller sees the same T.
    theta_target = np.radians(theta_target_deg)
    T_cmd = T_HOVER if T_fixed is None else T_fixed
    attitude = AttitudePDController(PD_GAINS, theta_target) # Initiate controller

    def controller(t, state, params):
        # Wrapper acting as adapter for AttitudePD to fit the (t, state, params)
        # contract.
        delta = attitude(theta_target, state, params, T_cmd)
        return (T_cmd, delta)

    state_0 = [0.0, 100.0, 0.0, 0.0, 0.0, 0.0] # Initial state
    sol = run_sim(state_0, PARAMS, (0.0, t_end), max_step, controller)

    t = sol.t
    theta = sol.y[4]
    thetadot = sol.y[5]
    # Controller is stateless, so delta is recomputed exactly from the states.
    delta = np.array([
        attitude(theta_target, sol.y[:, i], PARAMS, T_cmd) for i in range(t.size)
    ])
    return t, np.degrees(theta), np.degrees(thetadot), np.degrees(delta), theta_target_deg


def performance(t, theta_deg, delta_deg, target_deg):
    peak = theta_deg.max()
    os = (peak - target_deg) / target_deg * 100 if target_deg != 0 else 0.0

    # t_s, (settling time) computed from the last instant at which tehta is outside +- 2%
    band = np.abs(0.02 * target_deg)
    error = np.abs(theta_deg - target_deg)

    outside_indices = np.where(error > band)[0]

    if len(outside_indices) == 0:
        # signal never left the band after entering, settled immediatly
        t_s = 0.0
    elif outside_indices[-1] == len(t) - 1:
        # signal left the bounds and never came back (did not settle)
        t_s = np.nan
    else:
        # time at the time step right after the last time signal was out of bounds
        t_s = t[outside_indices[-1] + 1]

    # t_r (rise time) from 10% to 90% of target
    frac = theta_deg / target_deg

    above_10 = np.where(frac >= 0.1)[0]
    above_90 = np.where(frac >= 0.9)[0]
    if above_10.size and above_90.size:
        t_r = t[above_90[0]] - t[above_10[0]]
    else:
        t_r = np.nan
    return dict(peak=peak, overshoot=os, settling_time=t_s, rise_time=t_r)


SETTLE_BAND = 0.02
SCENARIOS = [('2 deg step (linear)', 2.0), ('8 deg step (saturating)', 8.0)]

# ============================================================================
# Figure A — per-scenario time histories: theta (plot 1), thetadot (plot 4),
# delta (plot 2). Rows = signals, columns = step sizes (2 deg / 8 deg).
# ============================================================================
figA, axA = plt.subplots(3, 2, figsize=(13, 11), sharex=True)

for col, (label, cmd_deg) in enumerate(SCENARIOS):
    t, theta_deg, thetadot_deg, delta_deg, target = pitch_step_scenario(cmd_deg)
    perf = performance(t, theta_deg, delta_deg, target)

    # ---- Row 0: pitch tracking (plot 1) ----
    ax = axA[0, col]
    ax.plot(t, theta_deg, color='tab:blue', label=r'$\theta$')
    ax.axhline(target, color='k', ls='--', lw=1, label=r'$\theta_{cmd}$')
    ax.axhspan(target * (1 - SETTLE_BAND), target * (1 + SETTLE_BAND),
               color='green', alpha=0.12, label=r'$\pm2\%$ band')
    ax.axhline(target * (1 + OS_PRED / 100), color='tab:orange', ls=':', lw=1,
               label=f'pred OS {OS_PRED:.1f}%')
    if np.isfinite(perf['settling_time']) and perf['settling_time'] > 0:
        ax.axvline(perf['settling_time'], color='tab:red', ls='-.', lw=1,
                   label=f"$t_s$ {perf['settling_time']:.2f} s")
    ax.set_title(label)
    ax.set_ylabel(r'pitch $\theta$ [deg]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)
    ax.text(0.02, 0.97,
            f"OS {perf['overshoot']:.2f}%\n"
            f"$t_r$ {perf['rise_time']:.2f} s\n"
            f"$t_s$ {perf['settling_time']:.2f} s",
            transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    # ---- Row 1: angular rate (plot 4) ----
    ax = axA[1, col]
    ax.plot(t, thetadot_deg, color='tab:green')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_ylabel(r'rate $\dot{\theta}$ [deg/s]')
    ax.grid(True, alpha=0.3)

    # ---- Row 2: gimbal command (plot 2) ----
    ax = axA[2, col]
    ax.plot(t, delta_deg, color='tab:purple', label=r'$\delta$')
    ax.axhline(0, color='gray', lw=0.8)
    ax.axhline(DELTA_MAX_DEG, color='gray', ls=':', lw=1, label=r'$\pm\delta_{max}$')
    ax.axhline(-DELTA_MAX_DEG, color='gray', ls=':', lw=1)
    ax.set_ylabel(r'gimbal $\delta$ [deg]')
    ax.set_xlabel('time [s]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    nz = delta_deg[np.abs(delta_deg) > 1e-6]
    sign_ok = nz.size and nz[0] < 0
    saturated = np.abs(delta_deg).max() >= DELTA_MAX_DEG - 1e-3
    ax.text(0.02, 0.05,
            f"{'negative first: OK' if sign_ok else 'negative first: NO'}\n"
            f"{'saturates' if saturated else 'no saturation'}",
            transform=ax.transAxes, va='bottom', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    print(f"[{label}] {perf}")

figA.suptitle(r'Inner attitude loop — step response '
              f'($\\omega_n$={OMEGA_N:.1f} rad/s, $\\zeta$={ZETA:.2f})', fontsize=13)
figA.tight_layout()
figA.savefig(result_path('attitude_loop_response.png'), dpi=150)
plt.close(figA)

# ============================================================================
# Figure B — thrust robustness of the inversion (plot 3).
# Same small step at three throttle settings. The step is kept in the linear
# regime even at the LOWEST thrust (lowest authority) so the gimbal never
# saturates; the inversion then cancels the T-dependence and the curves should
# coincide. (A larger step would saturate and the curves would diverge, since
# saturated authority b*sin(delta_max) scales with T.)
# ============================================================================
ROBUST_STEP_DEG = 1.5
THRUSTS = [('T_min', PARAMS['T_min']), ('hover', T_HOVER), ('T_max', PARAMS['T_max'])]

figB, axB = plt.subplots(figsize=(9, 5))
for name, T_val in THRUSTS:
    t, theta_deg, _, _, target = pitch_step_scenario(ROBUST_STEP_DEG, T_fixed=T_val)
    axB.plot(t, theta_deg, label=f'{name} ({T_val:.0f} N)')
axB.axhline(ROBUST_STEP_DEG, color='k', ls='--', lw=1, label=r'$\theta_{cmd}$')
axB.set_xlabel('time [s]')
axB.set_ylabel(r'pitch $\theta$ [deg]')
axB.set_title(f'Thrust robustness: identical {ROBUST_STEP_DEG} deg step at three throttles\n'
              '(curves coincide -> inversion cancels thrust dependence)')
axB.grid(True, alpha=0.3)
axB.legend()
figB.tight_layout()
figB.savefig(result_path('attitude_loop_robustness.png'), dpi=150)
plt.close(figB)

