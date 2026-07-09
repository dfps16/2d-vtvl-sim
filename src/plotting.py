import matplotlib.pyplot as plt
import numpy as np

from src.paths import result_path


def plot_state(t, state, x_target, z_target, save_path=result_path('state_last_sim.png')):

    x, z, xdot, zdot, theta, thetadot = state

    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)
    
    axes = axes.flatten()
    
    axes[0].plot(t, x, label='x')
    axes[0].axhline(x_target, color='r', linestyle='--', linewidth=0.8, label='x_target')
    axes[0].set_ylabel('x [m]')
    axes[0].set_title('Lateral position over time')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, xdot)
    axes[1].set_ylabel('ẋ [m/s]')
    axes[1].set_title('Lateral velocity over time')
    axes[1].grid(True)

    axes[2].plot(t, z, label='z')
    axes[2].axhline(z_target, color='r', linestyle='--', linewidth=0.8, label='z_target')
    axes[2].set_ylabel('z [m]')
    axes[2].set_title('Vertical position over time')
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(t, zdot)
    axes[3].set_ylabel('ż [m/s]')
    axes[3].set_title('Vertical velocity over time')
    axes[3].grid(True)

    axes[4].plot(t, np.degrees(theta), label='θ')
    # axes[2].plot(t, np.degrees(theta_cmd), linestyle='--', label='θ_cmd')
    axes[4].set_ylabel('θ [deg]')
    axes[4].set_xlabel('Time [s]')
    axes[4].set_title('Attitude over time')
    axes[4].legend()
    axes[4].grid(True)

    axes[5].plot(t, np.degrees(thetadot))
    axes[5].set_ylabel('θ̇ [deg/s]')
    axes[5].set_xlabel('Time [s]')
    axes[5].set_title('Angular rate over time')
    axes[5].grid(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
