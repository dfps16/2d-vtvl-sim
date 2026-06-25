# 2D VTVL Descent Simulation

Planar (3-DOF) thrust-vectoring lander simulation with cascaded PD, LQR, and (stretch) convex G-FOLD guidance. Built as a GNC portfolio piece with direct VTVL relevance.

---

## Problem

A rigid-body lander in a vertical plane has state `(x, z, ẋ, ż, θ, θ̇)` and two controls: thrust magnitude `T` and gimbal angle `δ`. The plant is underactuated — three translational/rotational DOF driven by two inputs. Horizontal translation is controlled through attitude, exactly as in a quadrotor. The goal is a soft, accurate touchdown from an offset initial condition, with honest comparison of controllers on fuel, accuracy, and constraint satisfaction.

### Equations of motion

```
m ẍ  =  -T sin(θ - δ)
m z̈  =   T cos(θ - δ) - mg
I θ̈  =  -T sin(δ) · L
```

`θ` is pitch from vertical, `δ` is gimbal deflection from body axis, `L` is the moment arm from CoM to gimbal pivot. Sign convention verified against free-body diagram — the torque opposes positive `δ` and the lateral force coupling is consistent.

---

## Repository layout

```
vtvl-descent-control/
├── src/
│   ├── dynamics.py       — 3-DOF EOM
│   ├── sim.py            — solve_ivp wrapper, touchdown event
│   ├── controllers.py    — altitude PID (baseline), cascaded PD, LQR (in progress)
│   ├── guidance.py       — convex G-FOLD reference (stretch, not started)
│   ├── run_scenarios.py  — divert-and-land, dispersion sweep (not started)
│   └── plotting.py       — trajectories, fuel, animation (not started)
├── results/
├── tests/
│   └── dynamics_test.py  — free-fall and hover equilibrium
└── requirements.txt
```

---

## Progress

### Week 1 — dynamics + simulator (complete)

**`dynamics.py`** — `lander_eom(t, state, T, delta, params)` implements the full 3-DOF nonlinear EOM. State vector `[x, z, ẋ, ż, θ, θ̇]`; params dict carries `m, I, L, g`.

**`sim.py`** — `run_sim` wraps `solve_ivp` with a terminal touchdown event (`z = 0`, descending). `closed_loop_rhs` queries the controller, saturates actuator commands to `[T_min, T_max]` and `±δ_max`, then calls the EOM.

**`controllers.py`** — `AltitudePIDController` (Baseline 0): 1-DOF hover/descent with `mg` feedforward. Gimbal fixed at zero; no lateral or attitude control. Sanity-checks the hover equilibrium and integrator.

**Tests** (both passing):
- `test_free_fall` — zero thrust reduces to analytic projectile motion, error < 1e-6 m
- `test_hover_equilibrium` — `T = mg, θ = δ = 0` holds state constant over 10 s, drift < 1e-4

### Week 2 — cascaded PD (next)

- Inner loop: gimbal `δ` controls pitch `θ`
- Outer loop: commanded tilt `θ_cmd` controls horizontal position; throttle controls altitude
- Divert-and-land scenario: offset start → vertical soft landing
- Plot initial-undershoot to confirm understanding of the underactuation coupling

### Week 3 — LQR (planned)

Linearise about hover, controllability check, full-state feedback design, mass depletion, disturbance injection, comparison against cascaded PD.

### Week 4 — guidance stretch + write-up (planned)

Convex G-FOLD reference tracked by LQR, or Monte Carlo dispersion analysis if skipping guidance. Animation. README becomes the short technical report.

---

## Parameters (nominal)

| Symbol | Value | Description |
|--------|-------|-------------|
| `m` | 1500 kg | Dry mass |
| `I` | 2000 kg·m² | Pitch moment of inertia |
| `L` | 1.5 m | CoM-to-gimbal moment arm |
| `g` | 9.81 m/s² | Gravitational acceleration |
| `T_min` | TBD | Minimum throttle (non-zero — the non-convex constraint) |
| `T_max` | TBD | Maximum thrust |
| `δ_max` | TBD | Gimbal deflection limit |

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r vtvl-descent-control/requirements.txt
```

Run tests from the repo root:

```bash
python -m pytest vtvl-descent-control/tests/ -v
```
