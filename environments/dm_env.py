from abc import ABC
import sys
import numpy as np

import py_dismech

from environments.base_env import BaseEnv
from environments.utils.common import NUM_NODES, LENGTH, DENSITY, RADIUS, YOUNG_MOD, POISSON, MU


class DisMechEnv(BaseEnv, ABC):
    """
    An abstract base class for constructing DisMech environments for RL.
    """

    def __init__(self,
                 sim_timestep: float,
                 control_interval: int,
                 timeout_steps: int,
                 render: bool = False):
        """
        Args:
            sim_timestep: Simulation timestep.
            control_interval: The number of sim steps for every control.
            timeout_steps: Number of steps before timing out an episode.
        """
        super().__init__(sim_timestep, control_interval, timeout_steps, render)

    def _create_new_sim(self):
        """
        This function can be called to create a new simulation.
        This is useful for episode resets.
        """
        self._simulator = py_dismech.SimulationManager()

        self._simulator.sim_params.dt = self._sim_timestep
        self._simulator.sim_params.sim_time = 1000000
        self._simulator.sim_params.ftol = 1e-3
        self._simulator.sim_params.max_iter.num_iters = 2
        self._simulator.sim_params.max_iter.terminate_at_max = False

        self._simulator.render_params.renderer = py_dismech.HEADLESS
        self._simulator.render_params.cmd_line_per = 0

        # Create the rod
        self._simulator.soft_robots.addLimb(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, LENGTH]),
            num_nodes=NUM_NODES,
            rho=DENSITY,
            rod_radius=RADIUS,
            youngs_modulus=YOUNG_MOD,
            poisson_ratio=POISSON,
            mu=MU,
            col_group=0xFFFF  # have manipulator collide with everything
        )

        # Fix the bottom end.
        self._simulator.soft_robots.lockEdge(0, 0)

        # Child classes can set up custom sim params
        self._custom_sim_params()

        # We always include gravity
        gravity_force = py_dismech.GravityForce(self._simulator.soft_robots,
                                                np.array([0.0, 0.0, -9.8]))
        self._simulator.forces.addForce(gravity_force)

        self._arm = self._simulator.soft_robots.limbs[0]

        self._simulator.initialize(sys.argv)
