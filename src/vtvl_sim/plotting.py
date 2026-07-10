import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon


def plot_state(sim_results, sim_setup):

    t = sim_results['t']
    x = sim_results['x']
    z = sim_results['z']
    xdot = sim_results['xdot']
    zdot = sim_results['zdot']
    theta = sim_results['theta']
    thetadot = sim_results['thetadot']

    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)
    
    axes = axes.flatten()
    
    axes[0].plot(t, x, label='x')
    # axes[0].axhline(x_target, color='r', linestyle='--', linewidth=0.8, label='x_target')
    axes[0].set_ylabel('x [m]')
    axes[0].set_title('Lateral position over time')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, xdot)
    axes[1].set_ylabel('ẋ [m/s]')
    axes[1].set_title('Lateral velocity over time')
    axes[1].grid(True)

    axes[2].plot(t, z, label='z')
    # axes[2].axhline(z_target, color='r', linestyle='--', linewidth=0.8, label='z_target')
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
    return fig

def plot_trajectory(sim_results):

    x = sim_results['x']
    z = sim_results['z']

    # error = np.abs(x_target - x[-1])
    # xdot_f = xdot[-1]
    # zdot_f = zdot[-1]
    # vel_abs = np.sqrt(xdot_f ** 2 + zdot_f ** 2)
    # theta_f = np.degrees(theta[-1])

    fig, ax = plt.subplots(figsize=(4, 12))
    ax.plot(x, z)
    ax.plot(x[0], z[0], 'go', label='start')
    ax.plot(x[-1], z[-1], 'rs', label='end')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_title('Trajectory (z vs x)')
    ax.legend(loc='upper right')
    ax.grid(True)
    ax.set_aspect('equal')

    fig.tight_layout()
    return fig

def _resample(t, arrays, fps, playback_speed):
    """Interpolate the (non-uniform) integrator output onto a uniform frame grid.

    solve_ivp uses adaptive steps, so playing raw samples at fixed fps would
    distort timing. We build an even time base and linearly interpolate.

    playback_speed is sim-seconds shown per real second: 1.0 = real time,
    <1.0 = slow motion (more frames), >1.0 = time-lapse.
    """
    dt_frame = playback_speed / fps
    t_uniform = np.arange(t[0], t[-1], dt_frame)
    resampled = [np.interp(t_uniform, t, a) for a in arrays]
    return t_uniform, resampled


def _glyph(cx, cz, theta, delta, throttle_frac, glyph_len):
    """Return drawing primitives for the lander at one instant.

    Body 'up'/nose unit vector n = (-sin theta, cos theta) — consistent with the
    thrust direction in dynamics.lander_eom. Lateral unit p = (cos theta, sin theta).
    The exhaust plume points opposite the gimballed thrust axis (theta - delta).
    """
    half = glyph_len / 2.0
    width = glyph_len * 0.35
    leg_len = glyph_len * 0.55
    flame_base = glyph_len * 0.35
    flame_gain = glyph_len * 1.5

    n = np.array([-np.sin(theta), np.cos(theta)])   # body axis, nose direction
    p = np.array([np.cos(theta), np.sin(theta)])    # body lateral axis
    C = np.array([cx, cz])

    nose = C + half * n
    base = C - half * n

    # Body rectangle corners (closed polygon)
    body = np.array([
        nose + (width / 2) * p,
        nose - (width / 2) * p,
        base - (width / 2) * p,
        base + (width / 2) * p,
    ])

    # Two splayed landing legs from the base, drawn as one broken line
    dir_l = (-n - p)
    dir_l = dir_l / np.linalg.norm(dir_l)
    dir_r = (-n + p)
    dir_r = dir_r / np.linalg.norm(dir_r)
    foot_l = base + leg_len * dir_l
    foot_r = base + leg_len * dir_r
    legs_x = [base[0], foot_l[0], np.nan, base[0], foot_r[0]]
    legs_z = [base[1], foot_l[1], np.nan, base[1], foot_r[1]]

    # Exhaust plume: opposite the thrust direction (-sin(theta-delta), cos(theta-delta))
    thrust_dir = np.array([-np.sin(theta - delta), np.cos(theta - delta)])
    plume_dir = -thrust_dir
    flame_len = flame_base + flame_gain * throttle_frac
    flame_tip = base + flame_len * plume_dir
    flame_x = [base[0], flame_tip[0]]
    flame_z = [base[1], flame_tip[1]]

    return body, (legs_x, legs_z), (flame_x, flame_z)



def animate_descent(sim_results, sim_setup,
                    fps=30, playback_speed=1.0, glyph_len=8.0,
                    save_path=None):
    """Render the trajectory to a video file. Returns the FuncAnimation object.

    targets: list of (x, z) waypoints to mark (one per mission phase).
    """
    t = sim_results['t']
    x = sim_results['x']
    z = sim_results['z']
    xdot = sim_results['xdot']
    zdot = sim_results['zdot']
    theta = sim_results['theta']
    delta = sim_results['delta']
    u_T = sim_results['u_T']
    zddot_des = sim_results['zddot_des']

    params = sim_setup['params']
    T_max = params['T_max']
    thr_act = u_T / T_max
    T_cmd = zddot_des * params['m']
    thr_cmd = T_cmd / T_max
    speed = np.sqrt(xdot ** 2 + zdot ** 2)

    targets = [(x_t, z_t) for x_t, z_t, _ in sim_setup['phases']]

    t_u, (x_u, z_u, th_u, dl_u, act_u, cmd_u, sp_u) = _resample(
        t, [x, z, theta, delta, thr_act, thr_cmd, speed], fps, playback_speed
    )
    z_u_pos = np.interp(t_u, t, z)  # altitude for HUD (same as z_u, kept explicit)

    # Fixed axis limits (equal aspect) from trajectory + target extents, padded.
    tx = [p[0] for p in targets]
    tz = [p[1] for p in targets]
    pad = glyph_len * 1.5
    x_lo = min(x.min(), *tx) - pad
    x_hi = max(x.max(), *tx) + pad
    z_hi = max(z.max(), *tz) + pad
    z_lo = -pad

    fig, ax = plt.subplots(figsize=(6, 11))
    ax.set_aspect('equal')
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(z_lo, z_hi)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('z [m]')
    ax.set_title('VTVL trajectory (lander glyph not to scale)')
    ax.grid(True, alpha=0.3)

    # Static scene
    ax.axhline(0.0, color='saddlebrown', linewidth=2, zorder=0)          # ground
    for k, (tx_, tz_) in enumerate(targets):
        ax.plot(tx_, tz_, 'r*', markersize=16, zorder=1,
                label='target' if k == 0 else None)
    ax.plot(x[0], z[0], 'go', markersize=6, label='start', zorder=1)

    # Dynamic artists
    (trail,) = ax.plot([], [], '-', color='tab:blue', linewidth=1.0, alpha=0.7, label='path')
    (flame,) = ax.plot([], [], '-', color='orange', linewidth=6, solid_capstyle='round', zorder=2)
    (legs,) = ax.plot([], [], '-', color='dimgray', linewidth=2, zorder=3)
    body = Polygon(np.zeros((4, 2)), closed=True, facecolor='lightsteelblue',
                   edgecolor='black', linewidth=1.2, zorder=4)
    ax.add_patch(body)
    hud = ax.text(0.03, 0.97, '', transform=ax.transAxes, va='top', ha='left',
                  family='monospace', fontsize=9,
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.legend(loc='upper right')

    def update(i):
        # Plume length tracks the actual (applied) throttle — the physical thrust.
        body_xy, (lx, lz), (fx, fz) = _glyph(
            x_u[i], z_u[i], th_u[i], dl_u[i], act_u[i], glyph_len
        )
        body.set_xy(body_xy)
        legs.set_data(lx, lz)
        flame.set_data(fx, fz)
        trail.set_data(x_u[:i + 1], z_u[:i + 1])
        hud.set_text(
            f't      = {t_u[i]:6.2f} s\n'
            f'alt    = {z_u_pos[i]:6.1f} m\n'
            f'speed  = {sp_u[i]:6.2f} m/s\n'
            f'theta  = {np.degrees(th_u[i]):6.2f} deg\n'
            f'thrott   cmd={cmd_u[i] * 100:6.1f} %  act={act_u[i] * 100:6.1f} %'
        )
        return body, legs, flame, trail, hud

    anim = animation.FuncAnimation(
        fig, update, frames=len(t_u), interval=1000.0 / fps, blit=False
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if shutil.which('ffmpeg'):
            writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
            out = save_path
        else:
            print('[animate_descent] ffmpeg not found — falling back to GIF.')
            writer = animation.PillowWriter(fps=fps)
            out = os.path.splitext(save_path)[0] + '.gif'
        anim.save(out, writer=writer)
        print(f'[animate_descent] wrote {out}  ({len(t_u)} frames @ {fps} fps)')
    return fig, anim