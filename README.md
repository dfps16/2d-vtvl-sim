# 2D VTVL Descent Simulation

Planar (3-DOF) thrust-vectoring lander simulation with cascaded PD, LQR, and (stretch) convex G-FOLD guidance.

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
2d-vtvl-sim/
├── pyproject.toml        — project metadata + pinned dependencies (uv-managed)
├── uv.lock               — resolved dependency lockfile for reproducible installs
├── test_scenarios/
│   └── scenario1.json    — example scenario: physical params, controller/gains, phases, outputs
├── src/
│   └── vtvl_sim/
│       ├── params.py          — single source of truth: physical params, gains, design targets
│       ├── paths.py           — centralised results-directory paths (no hardcoded absolute paths)
│       ├── dynamics.py        — 3-DOF EOM
│       ├── sim.py             — vtvl_solver/sim_run: solve_ivp wrapper, phase chaining, touchdown event
│       ├── controllers.py     — altitude PID, attitude PD, cascaded PD + CONTROLLER_REGISTRY; LQR not yet added
│       ├── schemas.py         — Pydantic models validating scenario JSON files
│       ├── scenario_io.py     — load_scenario: JSON -> validated sim/solver/output setup
│       ├── post_processing.py — CSV export, touchdown report
│       ├── guidance.py        — convex G-FOLD reference (stretch, empty stub)
│       ├── run_scenarios.py   — CLI entrypoint: run a scenario JSON end to end
│       └── plotting.py        — state/trajectory plots, descent animation
├── notebooks/
│   ├── check_sim.py         — baseline altitude-PID diagnostics
│   ├── attitude_loop.py     — inner attitude-loop response + robustness
│   ├── check_attitude.py    — inner-loop verification against design targets
│   ├── check_cascade.py     — full cascade divert-and-land scenario
│   └── animate_descent.py   — schematic side-view descent animation (MP4/GIF)
├── results/              — saved diagnostic plots (generated, gitignored)
└── tests/
    ├── dynamics_test.py       — free-fall and hover equilibrium
    └── attitude_test_plan.md  — inner-loop regression spec (planned)
```

---

## Progress

### Week 1 — dynamics + simulator (complete)

**`dynamics.py`** — `lander_eom(t, state, T, delta, params)` implements the full 3-DOF nonlinear EOM. State vector `[x, z, ẋ, ż, θ, θ̇]`; params dict carries `m, I, L, g`.

**`sim.py`** — `run_sim` wraps `solve_ivp` with a terminal touchdown event (`z = 0`, descending). `closed_loop_rhs` queries the controller, saturates actuator commands to `[T_min, T_max]` and `±δ_max`, then calls the EOM.

**`controllers.py`** — `AltitudePIDController` (Baseline 0): 1-DOF hover/descent with `mg` feedforward. Gimbal fixed at zero; no lateral or attitude control. Sanity-checks the hover equilibrium and integrator.

**Tests** (both passing, logic re-verified via independent RK4 integration):
- `test_free_fall` — zero thrust reduces to analytic projectile motion, error < 1e-6 m
- `test_hover_equilibrium` — `T = mg, θ = δ = 0` holds state constant over 10 s, drift < 1e-4

**Closed-loop baseline result.** PID tuned to `kp=3.0, ki=0.0, kd=30.0` with conditional integration (integral active only when `|e| < 5`). From a 100 m drop at rest, the lander reaches a soft vertical touchdown at **ż ≈ -0.66 m/s** in ≈26 s — vertical channel confirmed working before adding lateral/attitude control.

> **Resolved in Week 2:** the earlier split between `tests/dynamics_test.py` (`m=1500, I=2000, L=1.5`) and the closed-loop sim (`m=120, I=200, L=0.5`) is gone — both now import a single `src/params.py` source of truth.

### Week 2 — cascaded PD (substantially complete)

Parameters centralised into `src/params.py` (physical constants, gains, and design targets), so the simulator, notebooks, and tests can no longer drift apart. `sim.py` now takes the controller as an explicit argument rather than through the params dict.

**`AttitudePDController`** (inner loop) — PD in angular-acceleration space with dynamic inversion. From the rotational EOM `I·θ̈ = -T·L·sin(δ)`, a PD law sets `θ̈_des`, inverted exactly to `δ = arcsin(-θ̈_des / b)` with `b = T·L/I`. Stateless (rate measured), with arcsin-domain clipping and hard δ saturation.

**`CascadedController`** — three nested loops, each generating the next loop's reference:
- *Outer (position → tilt):* horizontal PD → pitch reference `θ_ref = -ẍ_des / g` (small-angle inversion), clipped to `tilt_limit`.
- *Middle (altitude → thrust):* PD with exact hover feedforward `mg/cos(θ)`, saturated to `[T_min, T_max]`.
- *Inner (attitude → gimbal):* EOM inversion as above, using post-saturation thrust for `b`.

Gains are derived from design targets `(ζ, ωₙ)` assuming an ideal 2nd-order plant (`kp = ωₙ²`, `kd = 2ζωₙ`), so retuning happens at the physics level. Bandwidth separation ω_x ≪ ω_θ (0.4 vs 4 rad/s) keeps the inner loop quasi-static from the outer loop's perspective.

**Divert-and-land scenario running** (`notebooks/check_cascade.py`): from 100 m altitude with a 20 m lateral offset target, the cascade drives a soft vertical touchdown, producing position/attitude tracking, trajectory (z-vs-x), rate, and control-input plots plus CSV export.

**Remaining:** write the inner-loop regression suite (`tests/attitude_test_plan.md` specs it — steady-state error, overshoot vs linear prediction, sign convention, saturation) and extend regression coverage to the closed-loop lateral channel.

### Week 3 — LQR (planned)

Linearise about hover, controllability check, full-state feedback design, mass depletion, disturbance injection, comparison against cascaded PD.

### Week 4 — guidance stretch + write-up (planned)

Convex G-FOLD reference tracked by LQR, or Monte Carlo dispersion analysis if skipping guidance. Animation. README becomes the short technical report.

---

## Parameters (nominal)

Defined in `src/params.py`, the single source of truth imported by the simulator, notebooks, and tests:

| Symbol | Value | Description |
|--------|-------|-------------|
| `m` | 120 kg | Dry mass |
| `I` | 200 kg·m² | Pitch moment of inertia |
| `L` | 0.5 m | CoM-to-gimbal moment arm |
| `g` | 9.81 m/s² | Gravitational acceleration |
| `T_min` | 1000 N | Minimum throttle (0.4·T_max, non-zero) |
| `T_max` | 2500 N | Maximum thrust (≈2.1× hover weight) |
| `δ_max` | 12° | Gimbal deflection limit |
| `tilt_limit` | 10° | Pitch reference clamp (outer-loop θ_cmd limit) |

Mass depletion is deferred to Week 3 (known technical debt); LQR gains computed at fixed mass will need revisiting before the controller comparison is final.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Run tests from the repo root:

```bash
uv run pytest tests/ -v
```

Run a scenario:

```bash
uv run python -m vtvl_sim.run_scenarios test_scenarios/scenario1.json
```
