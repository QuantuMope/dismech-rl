import numpy as np
from typing import Union
from scipy.spatial.transform import Rotation as R

import alf

NUM_NODES = 21
NUM_EDGES = NUM_NODES - 1
LENGTH = 1.0
RADIUS = 5e-2
YOUNG_MOD = 10e6
DENSITY = 1000
POISSON = 0.5
SHEAR_MOD = YOUNG_MOD / (2 * (1 + POISSON))
MU = 0.0

# We do this so that IDE doesn't complain when defining the observation spec
array = Union[np.ndarray, alf.TensorSpec]


@alf.configurable(blacklist=['boundary', 'dt', 'workspace_dim'])
class MovingTarget:
    """
    A moving target for the soft manipulator to follow with its end effector.
    """

    def __init__(self,
                 boundary: np.ndarray,
                 dt: float,
                 workspace_dim: int = 3,
                 change_direction_prob: float = 0.025,
                 target_velocity: float = 0.5):
        """
        Args:
            boundary: The boundaries of the workspace. Should be either a (2,2) or (3,2) array
                depending on the workspace dimension.
            dt: The timestep.
            workspace_dim: The dimension of the workspace. Should be either 2 or 3.
            change_direction_prob: The probability of changing the direction of the target.
            target_velocity: The target velocity.
        """
        assert boundary.shape == (workspace_dim, 2)
        self._boundary = boundary
        self._dt = dt
        self._ws_dim = workspace_dim
        self._change_direction_prob = change_direction_prob
        self._target_velocity = target_velocity

        self._pos = np.zeros((self._ws_dim, ), dtype=np.float32)
        self._vel = np.zeros((self._ws_dim, ), dtype=np.float32)

        self.reset()

    @property
    def pos(self) -> np.ndarray:
        return self._pos.copy()

    @property
    def vel(self) -> np.ndarray:
        return self._vel.copy()

    def _reset_velocity(self):
        """
        Helper function to sample a random velocity.
        """
        rand_dir1 = np.pi * np.random.uniform(0, 2)
        rand_dir2 = np.pi * np.random.uniform(0, 2)

        if self._ws_dim == 3:
            self._vel[0] = np.cos(rand_dir1) * np.sin(rand_dir2)
            self._vel[1] = np.sin(rand_dir1) * np.sin(rand_dir2)
            self._vel[2] = np.cos(rand_dir2)
        else:
            self._vel[0] = np.cos(rand_dir1) * np.sin(rand_dir2)
            self._vel[1] = np.cos(rand_dir2)

        self._vel *= self._target_velocity

    def reset(self):
        # Reset position
        for i in range(self._ws_dim):
            self._pos[i] = np.random.uniform(*self._boundary[i])

        # Reset velocity
        self._reset_velocity()

    def step(self):
        assert self._pos is not None and self._vel is not None

        # Switch the velocity direction if the position passes a boundary
        hit_boundary = False
        for i in range(self._ws_dim):
            if self._pos[i] < self._boundary[i][0] or  \
               self._pos[i] > self._boundary[i][1]:
                hit_boundary = True
                self._vel[i] *= -1

        if not hit_boundary and np.random.uniform(
        ) < self._change_direction_prob:
            self._reset_velocity()

        self._pos += (self._vel * self._dt)


class StationaryTarget(MovingTarget):
    """
    A simple target that is stationary.
    In other words, there is no reason to call step().
    """

    def __init__(self, boundary: np.ndarray):
        super().__init__(
            boundary=boundary,
            dt=1.0,  # unused, just set to something arbitrary
            workspace_dim=3,
            change_direction_prob=0.0,
            target_velocity=0.0)

        self._orientation = None

    @property
    def orientation(self) -> np.ndarray:
        return self._orientation.astype(np.float32)

    def step(self):
        raise RuntimeError("Shouldn't call step() on a stationary target.")

    def reset(self):
        super().reset()

        # Randomly sample a quaternion
        # self._orientation = np.random.normal(0, 1, 4)
        # self._orientation /= np.linalg.norm(self._orientation)
        rpy = np.array([0.0, 0.0, np.random.uniform(-np.pi / 2, np.pi / 2)])
        self._orientation = R.from_euler('xyz', rpy).as_quat()
