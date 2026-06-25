

class AltitudePIDController:
    """PID controller for vertical descent. Commands thrust only — no attitude control."""

    def __init__(self, gains, z_target):
        self.kp = gains['kp']
        self.ki = gains['ki']
        self.kd = gains['kd']
        self.r = z_target

        self.integral_sum = 0
        self.e_prev = None
        self.t_prev = None

    def __call__(self, t, state, params):
        z = state[1]
        m = params['m']
        g = params['g']

        error = self.r - z

        # Skip derivative on first call; no dt available yet
        if self.t_prev is not None:
            dt = t - self.t_prev
            d_term = (error - self.e_prev) / dt
        else:
            dt = 0
            d_term = 0

        self.integral_sum += error * dt
        self.e_prev = error
        self.t_prev = t

        # m*g feedforward: PID only corrects deviations from hover, not the full weight
        u_T = self.kp * error + self.ki * self.integral_sum + self.kd * d_term + m * g
        u_delta = 0.0  # gimbal fixed; no lateral or attitude correction
        return (u_T, u_delta)
