import time

from nicegui import app, run, ui
from pydantic import ValidationError

from vtvl_sim.controllers import CONTROLLER_REGISTRY
from vtvl_sim.paths import result_path
from vtvl_sim.plotting import animate_descent, plot_state, plot_trajectory
from vtvl_sim.post_processing import compute_touchdown_metrics
from vtvl_sim.scenario_io import build_setup
from vtvl_sim.sim import sim_run

app.add_static_files('/results', 'results')

ui.label('Lander Control Simulation').classes('text-2xl')

DEFAULT_PARAMS = {
    'm': 150.0, 'I': 200.0, 'L': 0.5, 'g': 9.81,
    'T_max': 2500.0, 'T_min': 1000.0, 'isp': 200,
    'delta_max_deg': 12, 'tilt_limit_deg': 10,
}
DEFAULT_INITIAL_STATE = {'x': 0.0, 'z': 0.0, 'xdot': 0.0, 'zdot': 0.0, 'theta': 0.0, 'thetadot': 0.0}
DEFAULT_LANDING_TOLERANCE = 1.0
DEFAULT_T_END = 30.0
DEFAULT_SOLVER_SETUP = {'max_step': 0.05, 'method': 'RK45'}
DEFAULT_GAINS = {
    'kp_x': 0.16, 'kd_x': 0.8,
    'kp_z': 0.16, 'kd_z': 0.8,
    'kp_theta': 16.0, 'kd_theta': 6.4,
}

phase_rows = []

def add_phase_row():
    with phases_container:
        with ui.row() as row:
            with ui.row():
                with ui.column():
                    x_input = ui.number(label='x_target [m]', value=0.0)
                with ui.column():
                    z_input = ui.number(label='z_target [m]', value=0.0)
                with ui.column():
                    t_end_input = ui.number(label='t_end [s]', value=DEFAULT_T_END)
            entry = {'row': row, 'x_target': x_input, 'z_target': z_input, 't_end': t_end_input}
            ui.button(icon='delete', on_click=lambda: remove_phase_row(entry)).props('flat dense')
    phase_rows.append(entry)

def remove_phase_row(entry):
    entry['row'].delete()
    phase_rows.remove(entry)

def _placeholder_box(text):
    with ui.element('div').classes(
        'w-64 h-48 border-2 border-dashed border-gray-400 rounded '
        'flex items-center justify-center'
    ):
        ui.label(text).classes('text-gray-400 text-center px-2')

def show_placeholders():
    output_area.clear()
    with output_area:
        with ui.row():
            _placeholder_box('State plot will appear here')
            _placeholder_box('Trajectory plot will appear here')
            _placeholder_box('Animation will appear here (if enabled)')

def _metric_tile(label, value):
    with ui.column().classes('items-center px-4'):
        ui.label(value).classes('text-xl font-bold')
        ui.label(label).classes('text-xs text-grey-6')

def _show_metrics(metrics):
    with ui.row().classes('w-full justify-center gap-6'):
        if metrics['landed']:
            _metric_tile('Touchdown time', f"{metrics['touchdown_time']:.2f} s")
            _metric_tile('x error', f"{metrics['touchdown_error']:.2f} m")
            _metric_tile('Touchdown speed', f"{metrics['vel_touchdown']:.2f} m/s")
            _metric_tile('Touchdown angle', f"{metrics['angle_touchdown_deg']:.1f}°")
        else:
            _metric_tile('Status', 'No touchdown')
            _metric_tile('Final altitude error', f"{metrics['altitude_error']:.2f} m")
            _metric_tile('Final x error', f"{metrics['touchdown_error']:.2f} m")
        _metric_tile('Impulse', f"{metrics['total_impulse']:.0f} Ns")
        _metric_tile('Propellant mass', f"{metrics['propellant_mass']:.2f} kg")

async def on_run_click():
    gains = {field: inp.value for field, inp in gain_inputs.items()}

    raw_sim_setup = {
        'params': DEFAULT_PARAMS,
        'controller_name': controller_select.value,
        'gains': gains,
        'phases': [
            {
                'x_target': entry['x_target'].value,
                'z_target': entry['z_target'].value,
                't_end': entry['t_end'].value,
            }
            for entry in phase_rows
        ],
        'initial_state': DEFAULT_INITIAL_STATE,
        'landing_tolerance': DEFAULT_LANDING_TOLERANCE,
    }

    try:
        sim_setup, solver_setup = build_setup(raw_sim_setup, DEFAULT_SOLVER_SETUP)
    except ValidationError as e:
        ui.notify(str(e), type='negative', multi_line=True)
        return

    spinner.visible = True
    try:
        sim_results = await run.io_bound(sim_run, sim_setup, solver_setup)

        traj_fig = plot_trajectory(sim_results)
        traj_fig.savefig(result_path('last_gui_trajectory.png'), dpi=300)

        state_fig = plot_state(sim_results, sim_setup)
        state_fig.savefig(result_path('last_gui_state.png'), dpi=300)

        output_area.clear()
        with output_area:
            with ui.row():
                ui.image(f'/results/last_gui_state.png?v={time.time()}').classes('w-[40em]')
                ui.image(f'/results/last_gui_trajectory.png?v={time.time()}').classes('w-64')

                if animation_checkbox.value:
                    await run.io_bound(
                        animate_descent, sim_results, sim_setup,
                        save_path=result_path('last_gui_animation.mp4'),
                    )
                    ui.video(f'/results/last_gui_animation.mp4?v={time.time()}').classes(
                        'w-96 h-[32rem] object-contain'
                    )
                else:
                    _placeholder_box('Animation not generated — enable "Generate animation" and Run again')

            metrics = compute_touchdown_metrics(sim_setup, sim_results)
            _show_metrics(metrics)
    finally:
        spinner.visible = False

with ui.column().classes('w-full items-center'):
    output_area = ui.column()

show_placeholders()

ui.separator()

with ui.row():
    with ui.card():
        ui.label('Controller').classes('text-lg')
        controller_select = ui.select(
            list(CONTROLLER_REGISTRY.keys()),
            value='Cascaded PD',
            label='Controller',
        )
        gain_inputs = {}
        with ui.expansion('Gains', icon='tune').classes('w-full'):
            with ui.column():
                ui.label('Horizontal position (outer loop)').classes('text-sm text-grey-7')
                with ui.row():
                    gain_inputs['kp_x'] = ui.number(label='kp_x', value=DEFAULT_GAINS['kp_x'], format='%.3f')
                    gain_inputs['kd_x'] = ui.number(label='kd_x', value=DEFAULT_GAINS['kd_x'], format='%.3f')

                ui.label('Altitude (middle loop)').classes('text-sm text-grey-7')
                with ui.row():
                    gain_inputs['kp_z'] = ui.number(label='kp_z', value=DEFAULT_GAINS['kp_z'], format='%.3f')
                    gain_inputs['kd_z'] = ui.number(label='kd_z', value=DEFAULT_GAINS['kd_z'], format='%.3f')

                ui.label('Attitude (inner loop)').classes('text-sm text-grey-7')
                with ui.row():
                    gain_inputs['kp_theta'] = ui.number(label='kp_theta', value=DEFAULT_GAINS['kp_theta'], format='%.3f')
                    gain_inputs['kd_theta'] = ui.number(label='kd_theta', value=DEFAULT_GAINS['kd_theta'], format='%.3f')

    with ui.card():
        ui.label('Phases').classes('text-lg')
        phases_container = ui.column()
        ui.button('+ Add phase', icon='add', on_click=add_phase_row).props('flat')

add_phase_row()  # seed with one phase so it's not empty on load

with ui.row():
    animation_checkbox = ui.checkbox('Generate animation', value=False)
    ui.button('Run', on_click=on_run_click)
    spinner = ui.spinner(size='lg')
    spinner.visible = False

ui.run()
