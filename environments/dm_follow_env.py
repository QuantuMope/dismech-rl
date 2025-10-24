import numpy as np
from typing import Literal

import alf

from environments.dm_env import DisMechEnv
from environments.utils.common import MovingTarget
from environments.utils.follow_common import (FollowTaskObservation,
                                              FOLLOW_TARGET_BOUNDS,
                                              follow_task_reward,
                                              FollowTaskVisualizer)


@alf.configurable
class DisMechFollowEnv(DisMechEnv):

    def __init__(self,
                 sim_timestep: float,
                 control_interval: int,
                 timeout_steps: int,
                 workspace_dim: Literal[2, 3] = 3,
                 render: bool = False):

        super().__init__(sim_timestep=sim_timestep,
                         control_interval=control_interval,
                         timeout_steps=timeout_steps,
                         render=render)

        self._ws_dim = workspace_dim

        self._observation_spec = FollowTaskObservation(
            state_pos=alf.TensorSpec((self._n_state_points * self._ws_dim, )),
            state_vel=alf.TensorSpec((self._n_state_points * self._ws_dim, )),
            curr_kappa_bar=alf.TensorSpec(
                (self._n_ctrl_points * (self._ws_dim - 1), )),
            target_pos=alf.TensorSpec((self._ws_dim, )),
            target_vel=alf.TensorSpec((self._ws_dim, )),
        )

        self._action_spec = self._action_converter.action_spec()

        control_dt = sim_timestep * control_interval
        boundary = FOLLOW_TARGET_BOUNDS
        if self._ws_dim == 2:
            boundary = boundary[::2]
        self._target = MovingTarget(boundary=boundary,
                                    dt=control_dt,
                                    workspace_dim=self._ws_dim)

        if self._render:
            self._renderer = FollowTaskVisualizer(workspace_dim=workspace_dim)

        self._create_new_sim()

    def _generate_observation(self) -> FollowTaskObservation:
        positions = self._arm.getVertices()[self._state_indices]
        velocities = self._arm.getVelocities()[self._state_indices]

        if self._ws_dim == 2:
            positions = positions[:, ::2]
            velocities = velocities[:, ::2]

        positions = positions.ravel()
        velocities = velocities.ravel()

        return FollowTaskObservation(
            state_pos=positions.astype(np.float32),
            state_vel=velocities.astype(np.float32),
            curr_kappa_bar=self._action_converter.curr_kappa_bar,
            target_pos=self._target.pos,
            target_vel=self._target.vel,
        )

    def _compute_reward(self, obs: FollowTaskObservation) -> float:
        return follow_task_reward(obs, ws_dim=self._ws_dim)

    def _custom_sim_params(self) -> None:
        if self._ws_dim == 2:
            self._simulator.sim_params.enable_2d_sim = True

    def _custom_step(self, action: np.ndarray):
        delta_action = self._action_converter.transform_action(
            action=action,
            output_type="dismech",
            interpolate_steps=self._control_interval)

        for _ in range(self._control_interval):
            self._simulator.step_simulation(delta_action)
        self._target.step()

    def _custom_reset(self):
        self._target.reset()

    def render(self, mode='rgb_array'):
        if self._render:
            self._renderer.render(self._arm.getVertices(), self._target.pos)
