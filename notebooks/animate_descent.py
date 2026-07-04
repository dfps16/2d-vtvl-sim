"""Animate the 2D VTVL lander descent.

Reuses the closed-loop solution from ``cascade_run`` and renders a schematic
side-view animation: body attitude (theta), throttle-scaled exhaust plume,
gimballed thrust direction (theta - delta), and the path travelled so far.

Output is MP4 when ffmpeg is available, otherwise it falls back to an animated
GIF (Pillow) with a warning. The lander glyph is drawn at an exaggerated, fixed
visual size (NOT to physical scale) so attitude is legible against a ~100 m
trajectory.

Run from the ``vtvl-descent-control`` directory:
    python notebooks/animate_descent.py
"""

import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon

# Make ``src`` and ``notebooks`` importable when run as a script from any cwd.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.params import PARAMS
from src.paths import result_path

from notebooks.check_cascade import cascade_run


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


def chain_phases(state_0, phases):
    """Run a sequence of cascade phases into one continuous trajectory.

    Each phase's final state seeds the next phase's initial state, so an ascent
    and a descent (or any waypoint sequence) join seamlessly.

    phases: list of (x_target, altitude_target, t_end) tuples.
    Returns combined [t, state, u, accels] in the same layout as cascade_run,
    with time made monotonic across phases. A phase that ends early (e.g. the
    descent touchdown event firing at z=0) is honoured — the next phase starts
    from wherever the previous one actually stopped.
    """
    segments = []
    s0 = list(state_0)
    t_offset = 0.0
    for x_target, altitude_target, t_end in phases:
        t, state, u, accels = cascade_run(s0, x_target, altitude_target, t_end=t_end)
        segments.append((t + t_offset, state, u, accels))
        s0 = [comp[-1] for comp in state]   # last column -> next phase's initial state
        t_offset += t[-1]

    def _cat(pick):
        # Drop the duplicated first sample of every phase after the first so the
        # joined time base stays strictly increasing (np.interp needs that).
        parts = [pick(seg) if k == 0 else pick(seg)[1:] for k, seg in enumerate(segments)]
        return np.concatenate(parts)

    t_c = _cat(lambda s: s[0])
    state_c = [_cat(lambda s, i=i: s[1][i]) for i in range(6)]
    u_c = [_cat(lambda s, i=i: s[2][i]) for i in range(3)]
    accels_c = [_cat(lambda s, i=i: s[3][i]) for i in range(3)]
    return [t_c, state_c, u_c, accels_c]


def animate_descent(t, state, u, accels, targets,
                    fps=30, playback_speed=1.0, glyph_len=8.0,
                    save_path=result_path('check_cascade_descent.mp4')):
    """Render the trajectory to a video file. Returns the FuncAnimation object.

    targets: list of (x, z) waypoints to mark (one per mission phase).
    """
    x, z, xdot, zdot, theta, _ = state
    _, u_T, delta = u
    _, zddot_des, _ = accels

    T_max = PARAMS['T_max']
    # Throttle as a fraction of max thrust, so T_min (=0.4*T_max) reads 40%.
    # Actual (applied) throttle: u_T is already saturated to [T_min, T_max].
    thr_act = u_T / T_max
    # Commanded throttle: raw controller demand T_cmd = zddot_des * m, pre-saturation.
    # Left unclipped so saturation is visible (can read <40% or >100%).
    T_cmd = zddot_des * PARAMS['m']
    thr_cmd = T_cmd / T_max
    speed = np.sqrt(xdot ** 2 + zdot ** 2)

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

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if shutil.which('ffmpeg'):
        writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
        out = save_path
    else:
        print('[animate_descent] ffmpeg not found on PATH — falling back to GIF. '
              'Install ffmpeg (e.g. `brew install ffmpeg`) for MP4 output.')
        writer = animation.PillowWriter(fps=fps)
        out = os.path.splitext(save_path)[0] + '.gif'

    anim.save(out, writer=writer)
    plt.close(fig)
    print(f'[animate_descent] wrote {out}  ({len(t_u)} frames @ {fps} fps)')
    return anim


if __name__ == '__main__':
    # Full mission: lift off from the ground, then descend with a lateral divert.
    ground = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    phases = [
        # (x_target, altitude_target, t_end)
        (10.0, 100.0, 25.0),    # ascent to 100 m
        (20.0, 0.0, 25.0),     # descent + divert to x = 20 m (ends at touchdown)
    ]
    t, state, u, accels = chain_phases(ground, phases)
    targets = [(x_t, z_t) for x_t, z_t, _ in phases]
    animate_descent(t, state, u, accels, targets)
