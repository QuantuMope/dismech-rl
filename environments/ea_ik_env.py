import numpy as np

from scipy.spatial.transform import Rotation as R

from environments.ea_env import ElasticaEnv
from environments.utils.common import StationaryTarget
from environments.utils.ik_common import (IK_TARGET_BOUNDS, InvKinObservation,
                                          ik_task_reward, InvKinTaskVisualizer)

import alf


@alf.configurable
class ElasticaInvKinEnv(ElasticaEnv):

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
        pos = self._get_arm_pos()[self._state_indices].ravel()
        vel = self._get_arm_vel()[self._state_indices].ravel()

        body_axes = self._arm.director_collection[..., -1][:, [1, 0, 2]]
        m1 = -body_axes[1]
        m2 = -body_axes[0]
        tip_tangent = body_axes[2]

        rot_mat = np.array([m1, m2, tip_tangent]).T
        tip_orientation = R.from_matrix(rot_mat).as_quat()

        return InvKinObservation(
            state_pos=pos.astype(np.float32),
            state_vel=vel.astype(np.float32),
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
            output_type="elastica",
            interpolate_steps=self._control_interval,
            voronoi_lengths=self._arm.rest_voronoi_lengths)

        delta_curvature = delta_action["delta_curvature"]
        delta_theta = delta_action["delta_theta"]

        for _ in range(self._control_interval):
            self._arm.rest_kappa[:2] += delta_curvature
            self._arm.rest_kappa[2] += delta_theta
            self._time_tracker = self._step_env_func(self._stepper,
                                                     self._stages_and_update,
                                                     self._simulator,
                                                     self._time_tracker,
                                                     self._sim_timestep)

    def _custom_reset(self):
        self._target.reset()

    def _get_m1(self, i: int):
        body_axes = self._arm.director_collection[..., i][:, [1, 0, 2]]
        return -body_axes[1]

    def _get_m2(self, i: int):
        body_axes = self._arm.director_collection[..., i][:, [1, 0, 2]]
        return -body_axes[0]

    def render(self, mode='rgb_array'):
        if self._render:
            self._renderer.render(arm_pos=self._get_arm_pos(),
                                  target_pos=self._target.pos,
                                  target_quat=self._target.orientation,
                                  m1_getter=self._get_m1,
                                  m2_getter=self._get_m2)
