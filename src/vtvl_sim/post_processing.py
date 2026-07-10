
import numpy as np

from src.paths import result_path


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

def write_sim_report(sim_setup, sim_results, save_path=result_path('last_sim_report.txt')):
    phases = sim_setup['phases']
    x_target, z_target, _ = phases[-1]
    params = sim_setup['params']

    thrust = sim_results['u_T']
    isp = params['isp']

    # Stub for fuel consumption

    t = sim_results['t']
    x = sim_results['x']
    z = sim_results['z']
    xdot = sim_results['xdot']
    zdot = sim_results['zdot']
    theta = sim_results['theta']
    thetadot = sim_results['thetadot']

    xdot_touchdown = xdot[-1]
    zdot_touchdown = zdot[-1]
    vel_touchdown = np.sqrt(xdot_touchdown**2 + zdot_touchdown**2)
    angle_touchdown = theta[-1]

    altitude_error = z_target - z[-1]
    touchdown_error = x_target - x[-1]

    touchdown_time = t[-1]

    total_impulse = np.sum(thrust * t)
    propellant_mass = total_impulse / (params['g'] * isp)
    
    with open(save_path, 'w') as f:
        f.write("====== SIM REPORT ======\n")
        if altitude_error < sim_setup['landing_tolerance']:
            f.write("Touchdown state --\n")
            f.write(f"Touchdown time {touchdown_time:.3g} sec --\n")
            f.write(f"Touchdown x-error {touchdown_error:.3g} m --\n")
            f.write(f"Touchdown velocity {vel_touchdown:.3g} m/s --\n")
            f.write(f"Touchdown angle {angle_touchdown:.3g} deg --\n")
        else:
            f.write("No touchdown achieved --\n")
            f.write(f"Final altitude: {altitude_error:.3g} m --\n")
            f.write(f"Final x error: {touchdown_error:.3g} m --\n")
        f.write("Usage --\n")
        f.write(f"Total impulse required {total_impulse:.2f} Ns\n")
        f.write(f"Propellant mass required {propellant_mass:.2f} kg\n")