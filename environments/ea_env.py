from abc import ABC
import numpy as np

import elastica as ea
from elastica.timestepper import extend_stepper_interface

from environments.base_env import BaseEnv
from environments.utils.common import NUM_EDGES, LENGTH, DENSITY, RADIUS, YOUNG_MOD, SHEAR_MOD


class BaseSimulator(ea.BaseSystemCollection, ea.Constraints, ea.Connections,
                    ea.Forcing, ea.CallBacks):
    pass


class ElasticaEnv(BaseEnv, ABC):
    """
    An abstract base class for constructing Elastica environments for RL.
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

        self._step_env_func = None
        self._stages_and_update = None
        self._time_tracker = np.float64(0.0)
        self._stepper = ea.PositionVerlet()

    def _get_arm_pos(self) -> np.ndarray:
        return self._arm.position_collection[[1, 0, 2]].T

    def _get_arm_vel(self) -> np.ndarray:
        return self._arm.velocity_collection[[1, 0, 2]].T

    def _create_new_sim(self):
        """
        This function can be called to create a new simulation.
        This is useful for episode resets.
        """
        self._simulator = BaseSimulator()

        self._arm = ea.CosseratRod.straight_rod(
            n_elements=NUM_EDGES,
            start=np.array([0.0, 0.0, 0.0]),
            direction=np.array([0.0, 0.0, 1.0]),
            normal=np.array([1.0, 0.0, 0.0]),
            base_length=LENGTH,
            density=DENSITY,
            base_radius=RADIUS,
            shear_modulus=SHEAR_MOD,
            nu=10,
            youngs_modulus=YOUNG_MOD)

        self._simulator.append(self._arm)

        # Fix the bottom end.
        self._simulator.constrain(self._arm).using(
            ea.OneEndFixedRod,
            constrained_position_idx=(0, ),
            constrained_director_idx=(0, ))

        # We always include gravity
        self._simulator.add_forcing_to(self._arm).using(ea.GravityForces,
                                                        acc_gravity=np.array(
                                                            [0.0, 0.0, -9.8]))

        # Child classes can setup custom sim params
        self._custom_sim_params()

        self._simulator.finalize()

        self._step_env_func, self._stages_and_update = extend_stepper_interface(
            self._stepper, self._simulator)
