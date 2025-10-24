import numpy as np

import elastica as ea

from environments.ea_env import ElasticaEnv
from environments.utils.common import DENSITY
from environments.utils.obs_2d_common import (OBSTACLE_RADIUS, OBSTACLES_2D,
                                              TARGET_POS,
                                              Obstacle2DObservation,
                                              obstacles_2d_task_reward,
                                              Obstacle2DTaskVisualizer)

import alf


@alf.configurable
class ElasticaObstacle2DEnv(ElasticaEnv):

    def __init__(self,
                 sim_timestep: float,
                 control_interval: int,
                 timeout_steps: int,
                 render: bool = False):

        super().__init__(sim_timestep=sim_timestep,
                         control_interval=control_interval,
                         timeout_steps=timeout_steps,
                         render=render)

        self._observation_spec = Obstacle2DObservation(
            state_pos=alf.TensorSpec((self._n_state_points * 2, )),
            state_vel=alf.TensorSpec((self._n_state_points * 2, )),
            curr_kappa_bar=alf.TensorSpec((self._n_ctrl_points, )),
            target_pos=alf.TensorSpec((2, )),
        )

        self._action_spec = self._action_converter.action_spec()

        if self._render:
            self._steps_per_render = round(0.05 / self._sim_timestep)
            self._arm_pos_history = []
            self._renderer = Obstacle2DTaskVisualizer(
                ctrl_indices=self._ctrl_indices)

        self._create_new_sim()

    def _custom_sim_params(self):
        for x, z in OBSTACLES_2D:
            # Add the obstacles
            obstacle = ea.Cylinder(start=np.array([-0.5, x, z]),
                                   direction=np.array([1., 0., 0.]),
                                   normal=np.array([0., 1., 0.]),
                                   base_length=1.0,
                                   base_radius=OBSTACLE_RADIUS,
                                   density=DENSITY)
            self._simulator.append(obstacle)

            # Fix the obstacles as rigid
            self._simulator.constrain(obstacle).using(
                ea.OneEndFixedRod,
                constrained_position_idx=(0, ),
                constrained_director_idx=(0, ))

            # Add a contact force between the arm and the obstacle
            self._simulator.connect(self._arm,
                                    obstacle).using(ea.ExternalContact,
                                                    k=2 * 8e4,
                                                    nu=4.0)

    def _generate_observation(self) -> Obstacle2DObservation:
        pos = self._get_arm_pos()[self._state_indices, ::2].ravel()
        vel = self._get_arm_vel()[self._state_indices, ::2].ravel()

        return Obstacle2DObservation(
            state_pos=pos.astype(np.float32),
            state_vel=vel.astype(np.float32),
            curr_kappa_bar=self._action_converter.curr_kappa_bar,
            target_pos=TARGET_POS,
        )

    def _compute_reward(self, obs: Obstacle2DObservation) -> float:
        return obstacles_2d_task_reward(obs)

    def _custom_step(self, action: np.ndarray):
        delta_action = self._action_converter.transform_action(
            action=action,
            output_type="elastica",
            interpolate_steps=self._control_interval,
            voronoi_lengths=self._arm.rest_voronoi_lengths)

        delta_curvature = delta_action["delta_curvature"]

        for i in range(self._control_interval):
            self._arm.rest_kappa[:2] += delta_curvature
            self._time_tracker = self._step_env_func(self._stepper,
                                                     self._stages_and_update,
                                                     self._simulator,
                                                     self._time_tracker,
                                                     self._sim_timestep)

            if self._render and i % self._steps_per_render == 0:
                self._arm_pos_history.append(self._get_arm_pos()[:, ::2])

    def render(self, mode='rgb_array'):
        if self._render:
            self._renderer.render(arm_pos_history=self._arm_pos_history)
            self._arm_pos_history.clear()
