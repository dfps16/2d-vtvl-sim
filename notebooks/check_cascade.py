import numpy as np
from src.controllers import CascadedController
from src.params import PARAMS, SYS_PROP_CASC
from src.paths import result_path
from src.sim import run_sim

# Store design targets from params.py
ZETA_X = SYS_PROP_CASC['zeta_x']
OMEGA_X = SYS_PROP_CASC['omega_x']
ZETA_Z = SYS_PROP_CASC['zeta_z']
OMEGA_Z = SYS_PROP_CASC['omega_z']
ZETA_THETA = SYS_PROP_CASC['zeta_theta']
OMEGA_THETA = SYS_PROP_CASC['omega_theta']

# Calculate PD gains from design targets - assumes ideal 2nd order plant
PD_GAINS = {
    'kp_x': OMEGA_X ** 2,
    'kd_x': 2 * ZETA_X * OMEGA_X,
    'kp_z': OMEGA_Z ** 2,
    'kd_z': 2 * ZETA_Z * OMEGA_Z,
    'kp_theta': OMEGA_THETA ** 2,
    'kd_theta': 2 * ZETA_THETA * OMEGA_THETA,
}

def cascade_run(state_0, x_target, altitude_target, t_end=20.0, max_step=0.05):

    controller = CascadedController(PD_GAINS, x_target, altitude_target)
    sol = run_sim(state_0, PARAMS, (0.0, t_end), max_step, controller)

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

    state = [x, z, xdot, zdot, theta, thetadot]
    u = [theta_cmd, u_T, delta]
    accels = [xddot_des, zddot_des, thetaddot_des]

    return [t, state, u, accels]


def plot_cascade(t, state, u, x_target, z_target, save_path=result_path('check_cascade.png')):
    import matplotlib.pyplot as plt

    x, z, _, _, theta, _ = state
    theta_cmd, _, _ = u

    fig, axes = plt.subplots(3, 1, figsize=(5, 9), sharex=True)

    axes[0].plot(t, x, label='x')
    axes[0].axhline(x_target, color='r', linestyle='--', linewidth=0.8, label='x_target')
    axes[0].set_ylabel('x [m]')
    axes[0].set_title('Lateral position over time')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, z, label='z')
    axes[1].axhline(z_target, color='r', linestyle='--', linewidth=0.8, label='z_target')
    axes[1].set_ylabel('z [m]')
    axes[1].set_title('Vertical position over time')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t, np.degrees(theta), label='θ')
    axes[2].plot(t, np.degrees(theta_cmd), linestyle='--', label='θ_cmd')
    axes[2].set_ylabel('θ [deg]')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_title('Attitude over time')
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_rates(t, state, save_path=result_path('check_cascade_rates.png')):
    import matplotlib.pyplot as plt

    _, _, xdot, zdot, _, thetadot = state

    fig, axes = plt.subplots(3, 1, figsize=(5, 9), sharex=True)

    axes[0].plot(t, xdot)
    axes[0].set_ylabel('ẋ [m/s]')
    axes[0].set_title('Lateral velocity over time')
    axes[0].grid(True)

    axes[1].plot(t, zdot)
    axes[1].set_ylabel('ż [m/s]')
    axes[1].set_title('Vertical velocity over time')
    axes[1].grid(True)

    axes[2].plot(t, np.degrees(thetadot))
    axes[2].set_ylabel('θ̇ [deg/s]')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_title('Angular rate over time')
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_trajectory(state, x_target, z_target, save_path=result_path('check_cascade_trajectory.png')):
    import matplotlib.pyplot as plt

    x, z, xdot, zdot, theta, _ = state

    error = np.abs(x_target - x[-1])
    xdot_f = xdot[-1]
    zdot_f = zdot[-1]
    vel_abs = np.sqrt(xdot_f ** 2 + zdot_f ** 2)
    theta_f = np.degrees(theta[-1])

    

    fig, ax = plt.subplots(figsize=(4, 12))
    ax.plot(x, z)
    ax.plot(x[0], z[0], 'go', label='start')
    ax.plot(x[-1], z[-1], 'rs', label='end')
    ax.plot(x_target, z_target, 'r*', markersize=12, label='target')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_title('Trajectory (z vs x)')
    ax.set_xlim(0, x_target * 1.2)
    ax.legend(loc='upper right')
    ax.grid(True)
    ax.set_aspect('equal')

    # Touchdown metrics in a text box, kept out of the legend to save space
    ax.text(
        0.05, 0.05,
        f'e_x = {error:.2g} m\nv_t = {vel_abs:.2g} m/s\nθ_t = {theta_f:.2g}°',
        transform=ax.transAxes,
        va='bottom', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_control_inputs(t, u, save_path=result_path('check_cascade_inputs.png')):
    import matplotlib.pyplot as plt

    _, u_T, delta = u
    delta_max_deg = np.degrees(PARAMS['delta_max'])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, u_T)
    axes[0].set_ylabel('u_T [N]')
    axes[0].set_title('Commanded thrust over time')
    axes[0].grid(True)

    axes[1].plot(t, np.degrees(delta))
    axes[1].axhline( delta_max_deg, color='r', linestyle='--', linewidth=0.8, label='+δ_max')
    axes[1].axhline(-delta_max_deg, color='r', linestyle='--', linewidth=0.8, label='-δ_max')
    axes[1].set_ylabel('δ [deg]')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_title('Thrust vector angle over time')
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def save_csv(t, state, save_path=result_path('check_cascade.csv')):
    import csv
    x, z, xdot, zdot, theta, thetadot = state
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'x', 'z', 'xdot', 'zdot', 'theta', 'thetadot'])
        for row in zip(t, x, z, xdot, zdot, theta, thetadot):
            writer.writerow(row)


if __name__ == '__main__':
    state_0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    diversion_target = 20.0
    altitude_target = 100.0
    t, state, u, accels = cascade_run(state_0, x_target=diversion_target, altitude_target=altitude_target, t_end=30)
    plot_cascade(t, state, u, x_target=diversion_target, z_target=altitude_target)
    plot_trajectory(state, x_target=diversion_target, z_target=altitude_target)
    plot_rates(t, state)
    plot_control_inputs(t, u)
    save_csv(t, state)
