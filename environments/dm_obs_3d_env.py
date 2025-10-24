from functools import partial
import numpy as np

import py_dismech

from environments.dm_env import DisMechEnv
from environments.utils.common import DENSITY, YOUNG_MOD, POISSON, MU
from environments.utils.obs_3d_common import (OBSTACLE_RADIUS, TARGET_POS,
                                              obstacle_3d_task_reward,
                                              Obstacle3DObservation,
                                              Obstacle3DTaskVisualizer)
from utils.obstacle_creator import create_obstacles

import alf
from alf.data_structures import TimeStep


@alf.configurable
class DisMechObstacle3DEnv(DisMechEnv):

    def __init__(self,
                 sim_timestep: float,
                 control_interval: int,
                 timeout_steps: int,
                 render: bool = False):

        super().__init__(sim_timestep=sim_timestep,
                         control_interval=control_interval,
                         timeout_steps=timeout_steps,
                         render=render)

        self._sym_eqs = py_dismech.SymbolicEquations()
        self._sym_eqs.generateContactPotentialPiecewiseFunctions()
        self._sym_eqs.generateFrictionJacobianPiecewiseFunctions()

        self._observation_spec = Obstacle3DObservation(
            state_pos=alf.TensorSpec((self._n_state_points * 3, )),
            state_vel=alf.TensorSpec((self._n_state_points * 3, )),
            curr_kappa_bar=alf.TensorSpec((self._n_ctrl_points * 2, )),
            target_pos=alf.TensorSpec((3, )),
            obstacle_pos=alf.TensorSpec((8 * 2 * 3, )),
        )
        self._obstacles = None
        self._action_spec = self._action_converter.action_spec()

        if self._render:
            self._arm_pos_history = []
            self._renderer = Obstacle3DTaskVisualizer()

        self._create_new_sim()

    def _custom_sim_params(self) -> None:
        self._simulator.sim_params.max_iter.num_iters = 5

        # Create rigid obstacles. We need to use a minimum of 3 nodes for a rod to be properly defined.
        create_obstacle = partial(self._simulator.soft_robots.addLimb,
                                  num_nodes=3,
                                  rho=DENSITY,
                                  rod_radius=OBSTACLE_RADIUS,
                                  youngs_modulus=YOUNG_MOD,
                                  poisson_ratio=POISSON,
                                  mu=MU)

        # Create obstacles
        self._obstacles = create_obstacles()

        for i, (start, end) in enumerate(self._obstacles):
            create_obstacle(
                start=start, end=end,
                col_group=1 << i)  # have obstacles only contact manipulator
            self._simulator.soft_robots.lockEdge(i + 1, 0)
            self._simulator.soft_robots.lockEdge(i + 1, 1)

        if self._render:
            self._renderer.set_static_obstacles(self._obstacles)

        contact_force = py_dismech.ContactForce(
            self._simulator.soft_robots,
            col_limit=5e-2,
            delta=5e-3,
            k_scaler=1e6,
            friction=MU != 0,
            nu=1e-2,
            self_contact=False,
            symbolic_equations=self._sym_eqs)
        self._simulator.forces.addForce(contact_force)

    def _generate_observation(self) -> Obstacle3DObservation:
        positions = self._arm.getVertices()[self._state_indices].ravel()
        velocities = self._arm.getVelocities()[self._state_indices].ravel()

        return Obstacle3DObservation(
            state_pos=positions.astype(np.float32),
            state_vel=velocities.astype(np.float32),
            curr_kappa_bar=self._action_converter.curr_kappa_bar,
            target_pos=TARGET_POS,
            obstacle_pos=self._obstacles.ravel())

    def _compute_reward(self, obs: Obstacle3DObservation) -> float:
        return obstacle_3d_task_reward(obs)

    def _custom_step(self, action: np.ndarray) -> TimeStep:
        delta_action = self._action_converter.transform_action(
            action=action,
            output_type="dismech",
            interpolate_steps=self._control_interval)

        for _ in range(self._control_interval):
            self._simulator.step_simulation(delta_action)
            if self._render:
                self._arm_pos_history.append(self._arm.getVertices())

    def render(self, mode='rgb_array'):
        if self._render:
            for arm_pos in self._arm_pos_history:
                self._renderer.render(arm_pos=arm_pos)

            self._arm_pos_history.clear()
