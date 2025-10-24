import numpy as np
from scipy.spatial.transform import Rotation as R

from environments.dm_env import DisMechEnv
from environments.utils.common import StationaryTarget, NUM_EDGES
from environments.utils.ik_common import (IK_TARGET_BOUNDS, InvKinObservation,
                                          ik_task_reward, InvKinTaskVisualizer)

import alf


@alf.configurable
class DisMechInvKinEnv(DisMechEnv):

    def __init__(self,
                 sim_timestep: float,
                 control_interval: int,
                 timeout_steps: int,
                 render: bool = False):

        super().__init__(sim_timestep=sim_timestep,
                         control_interval=control_interval,
                         timeout_steps=timeout_steps,
                         render=render)

        self._observation_spec = InvKinObservation(
            state_pos=alf.TensorSpec((self._n_state_points * 3, )),
            state_vel=alf.TensorSpec((self._n_state_points * 3, )),
            tip_orientation=alf.TensorSpec((4, )),
            curr_kappa_bar=alf.TensorSpec((self._n_ctrl_points * 2, )),
            curr_twist_bar=alf.TensorSpec((self._n_ctrl_points, )),
            target_pos=alf.TensorSpec((3, )),
            target_quat=alf.TensorSpec((4, )),
        )

        self._action_spec = self._action_converter.action_spec()

        self._target = StationaryTarget(boundary=IK_TARGET_BOUNDS)

        if self._render:
            self._renderer = InvKinTaskVisualizer()

        self._create_new_sim()

    def _generate_observation(self) -> InvKinObservation:
        all_positions = self._arm.getVertices()

        tip_tangent = all_positions[-1] - all_positions[-2]
        tip_tangent /= np.linalg.norm(tip_tangent)

        positions = all_positions[self._state_indices]
        velocities = self._arm.getVelocities()[self._state_indices]

        positions = positions.ravel()
        velocities = velocities.ravel()

        m1 = self._arm.getM1(NUM_EDGES - 1)
        m2 = self._arm.getM2(NUM_EDGES - 1)

        rot_mat = np.array([m1, m2, tip_tangent]).T
        tip_orientation = R.from_matrix(rot_mat).as_quat()

        return InvKinObservation(
            state_pos=positions.astype(np.float32),
            state_vel=velocities.astype(np.float32),
            tip_orientation=tip_orientation.astype(np.float32),
            curr_kappa_bar=self._action_converter.curr_kappa_bar,
            curr_twist_bar=self._action_converter.curr_theta_bar,
            target_pos=self._target.pos,
            target_quat=self._target.orientation,
        )

    def _compute_reward(self, obs: InvKinObservation) -> float:
        return ik_task_reward(obs)

    def _custom_step(self, action: np.ndarray):
        delta_action = self._action_converter.transform_action(
            action=action,
            output_type="dismech",
            interpolate_steps=self._control_interval)

        for _ in range(self._control_interval):
            self._simulator.step_simulation(delta_action)

    def _custom_reset(self):
        self._target.reset()

    def render(self, mode='rgb_array'):
        if self._render:
            self._renderer.render(arm_pos=self._arm.getVertices(),
                                  target_pos=self._target.pos,
                                  target_quat=self._target.orientation,
                                  m1_getter=self._arm.getM1,
                                  m2_getter=self._arm.getM2)
