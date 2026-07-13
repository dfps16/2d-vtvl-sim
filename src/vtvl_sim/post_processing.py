
import numpy as np

from vtvl_sim.paths import result_path


def save_csv(sim_results, save_path=result_path('last_sim_data.csv')):
    import csv
    t = sim_results['t']
    x = sim_results['x']
    z = sim_results['z']
    xdot = sim_results['xdot']
    zdot = sim_results['zdot']
    theta = sim_results['theta']
    thetadot = sim_results['thetadot']

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t', 'x', 'z', 'xdot', 'zdot', 'theta', 'thetadot'])
        for row in zip(t, x, z, xdot, zdot, theta, thetadot):
            writer.writerow(row)

def compute_touchdown_metrics(sim_setup, sim_results):
    """Touchdown/usage metrics shared by write_sim_report and the GUI summary."""
    phases = sim_setup['phases']
    x_target, z_target, _ = phases[-1]
    params = sim_setup['params']

    thrust = sim_results['u_T']
    isp = params['isp']

    t = sim_results['t']
    x = sim_results['x']
    z = sim_results['z']
    xdot = sim_results['xdot']
    zdot = sim_results['zdot']
    theta = sim_results['theta']

    vel_touchdown = np.sqrt(xdot[-1] ** 2 + zdot[-1] ** 2)
    angle_touchdown_deg = np.degrees(theta[-1])

    altitude_error = z_target - z[-1]
    touchdown_error = x_target - x[-1]
    touchdown_time = t[-1]
    landed = altitude_error < sim_setup['landing_tolerance']

    total_impulse = np.sum(thrust * t)
    propellant_mass = total_impulse / (params['g'] * isp)

    return {
        'landed': landed,
        'touchdown_time': touchdown_time,
        'touchdown_error': touchdown_error,
        'vel_touchdown': vel_touchdown,
        'angle_touchdown_deg': angle_touchdown_deg,
        'altitude_error': altitude_error,
        'total_impulse': total_impulse,
        'propellant_mass': propellant_mass,
    }


def write_sim_report(sim_setup, sim_results, save_path=result_path('last_sim_report.txt')):
    m = compute_touchdown_metrics(sim_setup, sim_results)

    with open(save_path, 'w') as f:
        f.write("====== SIM REPORT ======\n")
        if m['landed']:
            f.write("Touchdown state --\n")
            f.write(f"Touchdown time {m['touchdown_time']:.3g} sec --\n")
            f.write(f"Touchdown x-error {m['touchdown_error']:.3g} m --\n")
            f.write(f"Touchdown velocity {m['vel_touchdown']:.3g} m/s --\n")
            f.write(f"Touchdown angle {m['angle_touchdown_deg']:.3g} deg --\n")
        else:
            f.write("No touchdown achieved --\n")
            f.write(f"Final altitude: {m['altitude_error']:.3g} m --\n")
            f.write(f"Final x error: {m['touchdown_error']:.3g} m --\n")
        f.write("Usage --\n")
        f.write(f"Total impulse required {m['total_impulse']:.2f} Ns\n")
        f.write(f"Propellant mass required {m['propellant_mass']:.2f} kg\n")