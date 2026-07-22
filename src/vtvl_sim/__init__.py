"""vtvl_sim — planar VTVL trajectory simulation engine.

Public API for downstream tools (CLI, GUI). Import from the package root:

    from vtvl_sim import sim_run, build_setup, CONTROLLER_REGISTRY
"""

from importlib.metadata import PackageNotFoundError, version

from vtvl_sim.controllers import CONTROLLER_REGISTRY
from vtvl_sim.plotting import (
    animate_descent,
    plot_engine,
    plot_state,
    plot_trajectory,
)
from vtvl_sim.post_processing import (
    compute_engine_metrics,
    compute_state_metrics,
    compute_trajectory_metrics,
)
from vtvl_sim.scenario_io import build_setup, load_scenario
from vtvl_sim.sim import sim_run

try:
    __version__ = version("vtvl_sim")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
      "sim_run", "build_setup", "load_scenario", "CONTROLLER_REGISTRY",
      "plot_trajectory", "plot_state", "plot_engine", "animate_descent",
      "compute_state_metrics", "compute_trajectory_metrics",
  "compute_engine_metrics",
      "__version__",
]