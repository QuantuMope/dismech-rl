import numpy as np
from typing import Callable, NamedTuple
from scipy.spatial.transform import Rotation as R

import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from environments.utils.common import array, NUM_EDGES

IK_TARGET_BOUNDS = np.array([[-0.6, 0.6], [-0.6, 0.6], [0.3, 0.9]],
                            dtype=np.float32)


class InvKinObservation(NamedTuple):
    """
    Observation of the environment.

    Attributes:
        state_pos: The position of the soft manipulator.
        state_vel: The velocity of the soft manipulator.
        tip_orientation: The orientation of the tip as a quaternion.
        curr_kappa_bar: The current kappa_bar of the soft manipulator's control points.
        curr_twist_bar: The current twist_bar of the soft manipulator's control points.
        target_pos: The position of the target.
        target_quat: The orientation of the target as a quaternion.

    Note that the values should only be an alf.TensorSpec
    when used to specify the observation spec.
    """
    state_pos: array
    state_vel: array
    tip_orientation: array
    curr_kappa_bar: array
    curr_twist_bar: array
    target_pos: array
    target_quat: array


def ik_task_reward(obs: InvKinObservation) -> float:
    dist = np.linalg.norm(obs.target_pos - obs.state_pos[-3:])
    distance_reward = -dist**2

    ori_dist = 1 - np.dot(obs.tip_orientation, obs.target_quat)**2
    orientation_reward = -ori_dist**2

    reward = distance_reward + 0.5 * orientation_reward

    bonus_dist_threshold = 0.05
    if bonus_dist_threshold < dist < 2 * bonus_dist_threshold:
        if ori_dist > 0.1:
            reward += 1.0 - 0.5 * ori_dist
        else:
            reward += 1.5 - 0.5 * ori_dist
    elif dist < bonus_dist_threshold:
        if ori_dist > 0.1:
            reward += 4.0 - 2.0 * ori_dist
        elif ori_dist < 0.05:
            reward += 6.0 - 2.0 * ori_dist
        else:
            reward += 4.5 - 2.0 * ori_dist

    return reward


class InvKinTaskVisualizer:

    def __init__(self):
        plt.ion()
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111,
                                         projection='3d')  # Use 3D projection
        self._ax.set_xlim(-1, 1)  # Set appropriate limits for your data
        self._ax.set_ylim(-1, 1)
        self._ax.set_zlim(0, 1)
        self._state_line_render, = self._ax.plot([], [], [],
                                                 'r-')  # 3D line plot
        self._target_pos_render, = self._ax.plot([], [], [],
                                                 'bo')  # 3D scatter plot
        self._target_x_axis, = self._ax.plot([], [], [], 'r-')
        self._target_y_axis, = self._ax.plot([], [], [], 'g-')
        self._target_z_axis, = self._ax.plot([], [], [], 'b-')

        # Initialize material director lines (empty at first)
        self._m1_lines = []
        self._m2_lines = []
        for _ in range(NUM_EDGES):
            m1_line, = self._ax.plot([], [], [], 'green', linewidth=2.5)
            m2_line, = self._ax.plot([], [], [], 'purple', linewidth=2.5)
            self._m1_lines.append(m1_line)
            self._m2_lines.append(m2_line)

    def render(self, arm_pos: np.ndarray, target_pos: np.ndarray,
               target_quat: np.ndarray, m1_getter: Callable,
               m2_getter: Callable):
        scale = 0.05

        self._state_line_render.set_data([0, *arm_pos[:, 0]],
                                         [0, *arm_pos[:, 1]])
        self._state_line_render.set_3d_properties([0, *arm_pos[:, 2]
                                                   ])  # Set z-coordinates

        tx = target_pos[0]
        ty = target_pos[1]
        tz = target_pos[2]
        self._target_pos_render.set_data([tx], [ty])
        self._target_pos_render.set_3d_properties([tz])  # Set z-coordinate

        t_rot = R.from_quat(target_quat).as_matrix()
        # Draw target reference frame axes
        self._target_x_axis.set_data([tx, tx + scale * t_rot[0, 0]],
                                     [ty, ty + scale * t_rot[1, 0]])
        self._target_x_axis.set_3d_properties([tz, tz + scale * t_rot[2, 0]])

        self._target_y_axis.set_data([tx, tx + scale * t_rot[0, 1]],
                                     [ty, ty + scale * t_rot[1, 1]])
        self._target_y_axis.set_3d_properties([tz, tz + scale * t_rot[2, 1]])

        self._target_z_axis.set_data([tx, tx + scale * t_rot[0, 2]],
                                     [ty, ty + scale * t_rot[1, 2]])
        self._target_z_axis.set_3d_properties([tz, tz + scale * t_rot[2, 2]])

        for i in range(len(arm_pos) - 1):
            # Compute midpoint of the edge
            x = 0.5 * (arm_pos[i][0] + arm_pos[i + 1][0])
            y = 0.5 * (arm_pos[i][1] + arm_pos[i + 1][1])
            z = 0.5 * (arm_pos[i][2] + arm_pos[i + 1][2])

            # Get material directors
            m1 = scale * m1_getter(i)
            m2 = scale * m2_getter(i)

            # Update m1 director (green)
            self._m1_lines[i].set_data([x, x + m1[0]], [y, y + m1[1]])
            self._m1_lines[i].set_3d_properties([z, z + m1[2]])

            # Update m2 director (purple)
            self._m2_lines[i].set_data([x, x + m2[0]], [y, y + m2[1]])
            self._m2_lines[i].set_3d_properties([z, z + m2[2]])

        plt.draw()
        plt.pause(0.05)
