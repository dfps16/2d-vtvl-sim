import numpy as np


class AltitudePIDController:
    """Altitude-only 1-DOF controller; gimbal fixed at δ=0, single actuator: thrust T.

    Intended as a baseline without attitude control — not suitable for horizontal station-keeping.
    """

    def __init__(self, gains, z_target):
        self.kp = gains['kp']
        self.ki = gains['ki']
        self.kd = gains['kd']
        self.r = z_target

        self.integral_sum = 0
        self.e_prev = None  # used by windup guard on the following call
        self.t_prev = None  # used to compute Δt on the following call

    def __call__(self, t, state, params):
        z = state[1]
        zdot = state[3]
        m = params['m']
        g = params['g']

        error = self.r - z

        # Skip derivative term on first call; Δt is undefined before t_prev is set
        if self.t_prev is not None:
            dt = t - self.t_prev
            d_term = zdot  # derivative-on-measurement: eliminates derivative kick on setpoint steps
        else:
            dt = 0
            d_term = 0

        # Conditional integration: integrator active only within |e| < 5 m to prevent windup
        # during large-error transients; resets to zero outside this band
        if self.e_prev is not None and np.abs(self.e_prev) < 5:
            self.integral_sum += error * dt
        else:
            self.integral_sum = 0

        self.e_prev = error
        self.t_prev = t

        # Gravity feedforward shifts the operating point to hover; PID corrects only residual deviations.
        # Eliminates steady-state thrust error without relying solely on the integrator.
        u_T = self.kp * error + self.ki * self.integral_sum - self.kd * d_term + m * g
        u_delta = 0.0  # gimbal fixed at zero; no lateral or attitude correction
        return (u_T, u_delta)

class AttitudePDController:
        """Attitude inner loop: PD in angular acceleration space with dynamic inversion.

        From the rotational EOM: I·θ̈ = -T·L·sin(δ). The PD law specifies θ̈_des,
        which is then inverted analytically to recover δ. Stateless — θ̇ is measured,
        so no integral action or state history is required.
        """

        def __init__(self, gains, theta_target):
            self.kp = gains['kp']
            self.kd = gains['kd']

            self.r = theta_target

        def __call__(self, theta_target, state, params, T):
            theta = state[4]
            thetadot = state[5]

            # Control effectiveness: scalar b = T·L/I relates sin(δ) to θ̈ via the rotational EOM
            b = T * params['L'] / params['I']

            # PD law defines desired angular acceleration; derivative-on-measurement avoids kick
            thetaddot_des = self.kp * (theta_target - theta) - self.kd * thetadot

            # Exact inversion of EOM: δ = arcsin(-θ̈_des / b)
            # Clip to [-1, 1] guards the arcsin domain when b is small (low thrust)
            delta = np.arcsin(np.clip( - thetaddot_des / b, - 1, 1))

            # Hard saturation at mechanical TVC travel limit; applied after inversion
            delta = np.clip(delta, - params['delta_max'], params['delta_max'])

            return delta
        
class CascadedController:
    """Three-loop cascade: horizontal position (outer) → pitch attitude (middle) → TVC angle (inner).

    Each loop generates a reference for the next:
      1. Horizontal PD  →  pitch reference θ_ref  (via small-angle linearisation of the horizontal EOM)
      2. Altitude PD    →  thrust command T         (with gravity feedforward)
      3. Attitude PD    →  gimbal angle δ           (via exact inversion of the rotational EOM)

    Validity assumption: bandwidth separation ω_x << ω_θ, so the attitude loop tracks θ_ref
    fast enough to appear quasi-static from the horizontal loop's perspective.
    All PD loops use velocity/rate measurements on the derivative term (derivative-on-measurement).
    No integral action — steady-state z error is eliminated by feedforward; steady-state x error
    is rejected by gain tuning alone.
    """

    def __init__(self, gains, x_target, z_target):
        self.kp_x = gains['kp_x']
        self.kd_x = gains['kd_x']
        self.kp_z = gains['kp_z']
        self.kd_z = gains['kd_z']
        self.kp_theta = gains['kp_theta']
        self.kd_theta = gains['kd_theta']

        self.r_z = z_target
        self.r_x = x_target

    def commanded_tilt(self, state, params):
        # ── Outer loop: horizontal position → pitch reference ──────────────────
        # Small-angle linearisation of horizontal EOM: ẍ ≈ -g·θ (for |θ|, |δ| << 1).
        # Invert to map desired ẍ to a pitch setpoint: θ_ref = -ẍ_des / g.
        # State vector: [x, z, ẋ, ż, θ, θ̇]
        x = state[0]
        xdot = state[2]

        g = params['g']
        theta_max = params['tilt_limit']

        e_x = self.r_x - x
        xddot_des = self.kp_x * e_x - self.kd_x * xdot  # PD in acceleration space
        theta_des = - xddot_des / g
        r_theta = np.clip(theta_des, - theta_max, theta_max)
        return r_theta, xddot_des

    def commanded_thrust(self, state, params):
        # State vector: [x, z, ẋ, ż, θ, θ̇]
        z = state[1]
        zdot = state[3]
        theta = state[4]

        m = params['m']
        g = params['g']
        T_min = params['T_min']
        T_max = params['T_max']

        # ── Middle loop: altitude → thrust ─────────────────────────────────────
        e_z = self.r_z - z

        # Exact hover feedforward at tilt θ: vertical thrust component is T·cos(θ), so T_ff = mg/cos(θ).
        # However by linearising we remove the m, and put it back after
        # Uses θ rather than the net gimbal angle (θ−δ) — valid approximation for small δ.
        feedforward_z = g / np.cos(theta)
        # cos(θ) → 0 as |θ| → 90°; TVC saturation keeps this well-conditioned in practice.

        zddot_des = self.kp_z * e_z - self.kd_z * zdot + feedforward_z
        T_des = zddot_des * m
        u_T = np.clip(T_des, T_min, T_max)  # actuator saturation applied before attitude inversion
        return u_T, zddot_des
    
    def commanded_thrust_vector(self, state, params, r_theta, u_T):
        # State vector: [x, z, ẋ, ż, θ, θ̇]
        theta = state[4]
        thetadot = state[5]

        L = params['L']
        I = params['I']

        delta_max = params['delta_max']
        

        # ── Inner loop: pitch attitude → TVC angle ─────────────────────────────
        # Rotational EOM: I·θ̈ = -T·L·sin(δ)  →  control effectiveness b = T·L/I.
        # u_T (post-saturation) is used so that b reflects the torque actually available.
        b = u_T * L / I

        e_theta = r_theta - theta
        # PD law specifies desired angular acceleration
        thetaddot_des = self.kp_theta * e_theta - self.kd_theta * thetadot

        # Exact inversion of rotational EOM: δ = arcsin(-θ̈_des / b)
        # Clip to [-1, 1] guards the arcsin domain; applies when θ̈_des exceeds available authority b
        delta_des = np.arcsin(np.clip(- thetaddot_des / b, -1, 1))

        u_delta = np.clip(delta_des, - delta_max, delta_max)  # mechanical TVC travel limit
        return u_delta, thetaddot_des


    def __call__(self, t, state, params):
        
        r_theta, xddot_des = self.commanded_tilt(state, params)
        u_T, zddot_des = self.commanded_thrust(state, params)
        u_delta, thetaddot_des = self.commanded_thrust_vector(state, params, r_theta=r_theta, u_T=u_T)

        return (u_T, u_delta)



