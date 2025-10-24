import numpy as np

import elastica as ea

from environments.ea_env import ElasticaEnv
from environments.utils.common import DENSITY
from environments.utils.obs_3d_common import (OBSTACLE_RADIUS, TARGET_POS,
                                              obstacle_3d_task_reward,
                                              Obstacle3DObservation,
                                              Obstacle3DTaskVisualizer)
from utils.obstacle_creator import create_obstacles

import alf


@alf.configurable
class ElasticaObstacle3DEnv(ElasticaEnv):

    def __init__(self,
                 sim_timestep: float,
                 control_interval: int,
                 timeout_steps: int,
                 render: bool = False):

        super().__init__(sim_timestep=sim_timestep,
                         control_interval=control_interval,
                         timeout_steps=timeout_steps,
                         render=render)

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
            self._steps_per_render = round(0.05 / self._sim_timestep)
            self._arm_pos_history = []
            self._renderer = Obstacle3DTaskVisualizer()

        self._create_new_sim()

    def _custom_sim_params(self):
        # obstacles_3d = OBSTACLES_3D[..., [1, 0, 2]]

        self._obstacles = create_obstacles()
        if self._render:
            self._renderer.set_static_obstacles(self._obstacles)

        for start, end in self._obstacles[..., [1, 0, 2]]:
            # Add the obstacles
            direction = end - start
            direction /= np.linalg.norm(direction)

            # Choose any arbitrary normal vector
            if abs(direction[0]) < abs(direction[1]):
                normal = np.array([0, -direction[2], direction[1]])
            else:
                normal = np.array([-direction[2], 0, direction[0]])
            normal /= np.linalg.norm(normal)

            obstacle = ea.Cylinder(start=start,
                                   direction=direction,
                                   normal=normal,
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

    def _generate_observation(self) -> Obstacle3DObservation:
        pos = self._get_arm_pos()[self._state_indices].ravel()
        vel = self._get_arm_vel()[self._state_indices].ravel()

        return Obstacle3DObservation(
            state_pos=pos.astype(np.float32),
            state_vel=vel.astype(np.float32),
            curr_kappa_bar=self._action_converter.curr_kappa_bar,
            target_pos=TARGET_POS,
            obstacle_pos=self._obstacles.ravel())

    def _compute_reward(self, obs: Obstacle3DObservation) -> float:
        return obstacle_3d_task_reward(obs)

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
                self._arm_pos_history.append(self._get_arm_pos())

    def render(self, mode='rgb_array'):
        if self._render:
            for arm_pos in self._arm_pos_history:
                self._renderer.render(arm_pos=arm_pos)

            self._arm_pos_history.clear()
