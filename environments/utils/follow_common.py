import numpy as np
from typing import Literal, NamedTuple

import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from environments.utils.common import array

FOLLOW_TARGET_BOUNDS = np.array([[-0.6, 0.6], [-0.6, 0.6], [0.3, 0.9]],
                                dtype=np.float32)


class FollowTaskObservation(NamedTuple):
    """
    Observation of the environment.

    Attributes:
        state_pos: The position of the soft manipulator.
        state_vel: The velocity of the soft manipulator.
        curr_kappa_bar: The current kappa_bar of the soft manipulator's control points.
        target_pos: The position of the target.
        target_vel: The velocity of the target.

    Note that the values should only be an alf.TensorSpec
    when used to specify the observation spec.
    """
    state_pos: array
    state_vel: array
    curr_kappa_bar: array
    target_pos: array
    target_vel: array


def follow_task_reward(obs: FollowTaskObservation,
                       ws_dim: Literal[2, 3] = 3) -> float:
    distance = np.linalg.norm(obs.target_pos - obs.state_pos[-ws_dim:])

    reward = -distance**2

    bonus_dist_threshold = 0.05

    if bonus_dist_threshold < distance < 2 * bonus_dist_threshold:
        reward += 0.5
    elif distance < bonus_dist_threshold:
        reward += 2.0

    return reward


class FollowTaskVisualizer:

    def __init__(self, workspace_dim: Literal[2, 3] = 3):
        plt.ion()
        self._ws_dim = workspace_dim
        if workspace_dim == 3:
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(
                111, projection='3d')  # Use 3D projection
            self._ax.set_xlim(-1, 1)  # Set appropriate limits for your data
            self._ax.set_ylim(-1, 1)
            self._ax.set_zlim(0, 1)
            self._state_line_render, = self._ax.plot([], [], [],
                                                     'r-')  # 3D line plot
            self._target_pos_render, = self._ax.plot([], [], [],
                                                     'bo')  # 3D scatter plot
        else:
            self._fig, self._ax = plt.subplots()
            self._ax.set_xlim(-1, 1)  # Set appropriate limits for your data
            self._ax.set_ylim(0, 1)
            self._state_line_render, = self._ax.plot([], [],
                                                     'r-')  # 3D line plot
            self._target_pos_render, = self._ax.plot([], [],
                                                     'bo')  # 3D scatter plot

    def render(self,
               arm_pos: np.ndarray,
               target_pos: np.ndarray,
               pause_time: float = 0.05):
        if self._ws_dim == 3:
            self._state_line_render.set_data([0, *arm_pos[:, 0]],
                                             [0, *arm_pos[:, 1]])
            self._state_line_render.set_3d_properties([0, *arm_pos[:, 2]
                                                       ])  # Set z-coordinates
            self._target_pos_render.set_data([target_pos[0]], [target_pos[1]])
            self._target_pos_render.set_3d_properties([target_pos[2]
                                                       ])  # Set z-coordinate
        else:
            self._state_line_render.set_data([0, *arm_pos[:, 0]],
                                             [0, *arm_pos[:, 2]])
            self._target_pos_render.set_data([target_pos[0]], [target_pos[1]])
        plt.draw()
        plt.pause(pause_time)
