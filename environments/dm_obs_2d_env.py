from functools import partial
import numpy as np
import alf

import py_dismech

from environments.dm_env import DisMechEnv
from environments.utils.common import DENSITY, YOUNG_MOD, POISSON, MU
from environments.utils.obs_2d_common import (OBSTACLE_RADIUS, OBSTACLES_2D,
                                              TARGET_POS,
                                              Obstacle2DObservation,
                                              obstacles_2d_task_reward,
                                              Obstacle2DTaskVisualizer)


@alf.configurable
class DisMechObstacle2DEnv(DisMechEnv):

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

        self._observation_spec = Obstacle2DObservation(
            state_pos=alf.TensorSpec((self._n_state_points * 2, )),
            state_vel=alf.TensorSpec((self._n_state_points * 2, )),
            curr_kappa_bar=alf.TensorSpec((self._n_ctrl_points, )),
            target_pos=alf.TensorSpec((2, )),
        )

        self._action_spec = self._action_converter.action_spec()

        if self._render:
            self._arm_pos_history = []
            self._renderer = Obstacle2DTaskVisualizer(
                ctrl_indices=self._ctrl_indices)

        self._create_new_sim()

    def _custom_sim_params(self):
        self._simulator.sim_params.enable_2d_sim = True
        self._simulator.sim_params.max_iter.num_iters = 5

        # Create rigid obstacles. We need to use a minimum of 3 nodes for a rod to be properly defined.
        # We reuse the params, but it doesn't really matter.
        create_obstacle = partial(self._simulator.soft_robots.addLimb,
                                  num_nodes=3,
                                  rho=DENSITY,
                                  rod_radius=OBSTACLE_RADIUS,
                                  youngs_modulus=YOUNG_MOD,
                                  poisson_ratio=POISSON,
                                  mu=MU)

        # Create obstacles
        for i, (x, z) in enumerate(OBSTACLES_2D):
            create_obstacle(
                start=np.array([x, -0.5, z]),
                end=np.array([x, 0.5, z]),
                col_group=1 << i)  # have obstacles only contact manipulator

            # Lock the edges to make it "rigid".
            self._simulator.soft_robots.lockEdge(i + 1, 0)
            self._simulator.soft_robots.lockEdge(i + 1, 1)

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

    def _generate_observation(self) -> Obstacle2DObservation:
        positions = self._arm.getVertices()[self._state_indices, ::2].ravel()
        velocities = self._arm.getVelocities()[
            self._state_indices, ::2].ravel()

        return Obstacle2DObservation(
            state_pos=positions.astype(np.float32),
            state_vel=velocities.astype(np.float32),
            curr_kappa_bar=self._action_converter.curr_kappa_bar,
            target_pos=TARGET_POS,
        )

    def _compute_reward(self, obs: Obstacle2DObservation) -> float:
        return obstacles_2d_task_reward(obs)

    def _custom_step(self, action: np.ndarray):
        delta_action = self._action_converter.transform_action(
            action=action,
            output_type="dismech",
            interpolate_steps=self._control_interval)

        for _ in range(self._control_interval):
            self._simulator.step_simulation(delta_action)
            if self._render:
                self._arm_pos_history.append(self._arm.getVertices()[:, ::2])

    def render(self, mode='rgb_array'):
        if self._render:
            self._renderer.render(arm_pos_history=self._arm_pos_history)
            self._arm_pos_history.clear()
