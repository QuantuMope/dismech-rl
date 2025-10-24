import numpy as np

import alf

from environments.ea_env import ElasticaEnv
from environments.utils.common import MovingTarget
from environments.utils.follow_common import (FollowTaskObservation,
                                              FOLLOW_TARGET_BOUNDS,
                                              follow_task_reward,
                                              FollowTaskVisualizer)


@alf.configurable
class ElasticaFollowEnv(ElasticaEnv):

    def __init__(self,
                 sim_timestep: float,
                 control_interval: int,
                 timeout_steps: int,
                 render: bool = False):

        super().__init__(sim_timestep=sim_timestep,
                         control_interval=control_interval,
                         timeout_steps=timeout_steps,
                         render=render)

        self._n_state_points = len(self._state_indices)

        self._observation_spec = FollowTaskObservation(
            state_pos=alf.TensorSpec((self._n_state_points * 3, )),
            state_vel=alf.TensorSpec((self._n_state_points * 3, )),
            curr_kappa_bar=alf.TensorSpec((self._n_ctrl_points * 2, )),
            target_pos=alf.TensorSpec((3, )),
            target_vel=alf.TensorSpec((3, )),
        )

        self._action_spec = self._action_converter.action_spec()

        control_dt = sim_timestep * control_interval
        self._target = MovingTarget(boundary=FOLLOW_TARGET_BOUNDS,
                                    dt=control_dt)

        if self._render:
            self._renderer = FollowTaskVisualizer()

        self._create_new_sim()

    def _generate_observation(self) -> FollowTaskObservation:
        positions = self._get_arm_pos()[self._state_indices].ravel()
        velocities = self._get_arm_vel()[self._state_indices].ravel()

        return FollowTaskObservation(
            state_pos=positions.astype(np.float32),
            state_vel=velocities.astype(np.float32),
            curr_kappa_bar=self._action_converter.curr_kappa_bar,
            target_pos=self._target.pos,
            target_vel=self._target.vel,
        )

    def _compute_reward(self, obs: FollowTaskObservation) -> float:
        return follow_task_reward(obs)

    def _custom_step(self, action: np.ndarray):
        delta_action = self._action_converter.transform_action(
            action=action,
            output_type="elastica",
            interpolate_steps=self._control_interval,
            voronoi_lengths=self._arm.rest_voronoi_lengths)

        delta_curvature = delta_action["delta_curvature"]

        for _ in range(self._control_interval):
            self._arm.rest_kappa[:2] += delta_curvature
            self._time_tracker = self._step_env_func(self._stepper,
                                                     self._stages_and_update,
                                                     self._simulator,
                                                     self._time_tracker,
                                                     self._sim_timestep)
        self._target.step()

    def _custom_reset(self):
        self._target.reset()

    def render(self, mode='rgb_array'):
        if self._render:
            self._renderer.render(self._get_arm_pos(), self._target.pos)
