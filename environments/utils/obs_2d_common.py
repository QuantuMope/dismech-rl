import numpy as np
from typing import List, NamedTuple

import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from environments.utils.common import array, NUM_NODES, RADIUS

OBSTACLE_RADIUS = 0.1

OBSTACLES_2D = np.array([
    [0.20, 0.12],
    [0.60, 0.12],
    [0.40, 0.00],
    [0.40, 0.37],
    [0.40, 0.57],
    [0.40, 0.77],
],
                        dtype=np.float32)

TARGET_POS = np.array([0.8, 0.15], dtype=np.float32)


class Obstacle2DObservation(NamedTuple):
    """
    Observation of the environment.

    Attributes:
        state_pos: The position of the soft manipulator.
        state_vel: The velocity of the soft manipulator.
        curr_kappa_bar: The current kappa_bar of the soft manipulator's control points.
        target_pos: The position of the target.

    Note that the values should only be an alf.TensorSpec
    when used to specify the observation spec.
    """
    state_pos: array
    state_vel: array
    curr_kappa_bar: array
    target_pos: array


def obstacles_2d_task_reward(obs: Obstacle2DObservation) -> float:
    tip_position = obs.state_pos[-2:]

    dist = np.linalg.norm(obs.target_pos - tip_position)
    reward = -dist**2

    bonus_dist_threshold = 0.05  # cm
    if bonus_dist_threshold < dist < 2 * bonus_dist_threshold:
        reward += 0.5
    elif dist < bonus_dist_threshold:
        reward += 2.0

    return reward


class Obstacle2DTaskVisualizer:

    def __init__(self, ctrl_indices: np.ndarray):
        plt.ion()
        self._fig, self._ax = plt.subplots()
        self._ax.set_xlim(-0.2, 1)  # Set appropriate limits for your data
        self._ax.set_ylim(-0.2, 1)

        self._state_circles = []
        for i in range(NUM_NODES):
            color = 'red'
            if i in ctrl_indices:
                color = 'blue'
            circle = plt.Circle((0, 0), radius=RADIUS, color=color, alpha=0.7)
            self._ax.add_artist(circle)
            self._state_circles.append(circle)

        circle = plt.Circle(tuple(TARGET_POS),
                            radius=0.075,
                            color="orange",
                            alpha=0.7)
        self._ax.add_artist(circle)

        self._obstacle_patches = []
        for obs in OBSTACLES_2D:
            circle = plt.Circle(obs, OBSTACLE_RADIUS, color='green')
            self._ax.add_artist(circle)
            self._obstacle_patches.append(circle)

    def render(self, arm_pos_history: List[np.ndarray]):
        for arm_pos in arm_pos_history:
            for i, pos in enumerate(arm_pos):
                self._state_circles[i].set_center(pos)

            plt.draw()
            plt.pause(0.05)
