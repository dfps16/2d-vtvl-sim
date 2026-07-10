import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Make ``vtvl_sim`` (under src/) importable when run as a script from any cwd.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from vtvl_sim.controllers import AltitudePIDController
from vtvl_sim.params import PARAMS, PID_GAINS
from vtvl_sim.sim import run_sim

z_target = 0  # m
controller = AltitudePIDController(PID_GAINS, z_target)

# Log raw u and saturated T at every RHS call (controller is stateful, can't replay post-hoc)
control_log = {'t': [], 'u': [], 'T': []}

def logging_controller(t, state, params):
    u_T, u_delta = controller(t, state, params)
    control_log['t'].append(t)
    control_log['u'].append(u_T)
    control_log['T'].append(np.clip(u_T, params['T_min'], params['T_max']))
    return (u_T, u_delta)

state_0 = [0.0, 100, 0.0, 0.0, 0.0, 0.0]  # x, z, xdot, zdot, theta, thetadot

sol = run_sim(state_0, PARAMS, (0.0, 50), 0.01, logging_controller)

t = sol.t
z = sol.y[1]
zdot = sol.y[3]

t_ctrl = np.array(control_log['t'])
u_ctrl = np.array(control_log['u'])
T_ctrl = np.array(control_log['T'])

t_td = sol.t_events[0][0] if len(sol.t_events[0]) > 0 else None
zdot_td = sol.y_events[0][0][3] if t_td is not None else None

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axes[0].plot(t_ctrl, u_ctrl, label='u (raw)')
axes[0].plot(t_ctrl, T_ctrl, label='T (saturated)', linestyle='--')
axes[0].axhline(PARAMS['T_min'], color='gray', linestyle=':', linewidth=0.8, label='T_min')
axes[0].axhline(PARAMS['T_max'], color='gray', linestyle=':', linewidth=0.8, label='T_max')
axes[0].set_ylabel('Force [N]')
axes[0].set_title('Control input u and thrust T')
axes[0].legend(loc='upper right')
axes[0].grid(True)

axes[1].plot(t, zdot)
if t_td is not None:
    axes[1].plot(t_td, zdot_td, 'ro', markersize=6)
    axes[1].annotate(
        f'touchdown ż = {zdot_td:.2f} m/s',
        xy=(t_td, zdot_td),
        xytext=(8, 8), textcoords='offset points',
        color='red', fontsize=9
    )
axes[1].set_ylabel('ż [m/s]')
axes[1].set_title('Vertical velocity ż (zdot)')
axes[1].grid(True)

axes[2].plot(t, z)
if t_td is not None:
    axes[2].axvline(t_td, color='red', linestyle='--', label=f'touchdown t={t_td:.2f}s')
    axes[2].legend(loc='upper right')
axes[2].set_ylabel('z [m]')
axes[2].set_xlabel('time [s]')
axes[2].set_title('Altitude z')
axes[2].grid(True)

fig.tight_layout()
fig.savefig('results/check_sim_diagnostics.png', dpi=150)
plt.close(fig)
