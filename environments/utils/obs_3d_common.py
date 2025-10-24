import numpy as np
from typing import List, NamedTuple

from typing import Tuple
import pyvista as pv
from scipy.spatial.transform import Rotation as R

from environments.utils.common import array, RADIUS

OBSTACLE_RADIUS = 0.03

TARGET_POS = np.array([0.8, 0.0, 0.35], dtype=np.float32)

OBSTACLES_3D = np.array(
    [[[0.4425154, 0.49850288, 0.58865217],
      [0.45413066, -0.49850288, 0.66510208]],
     [[0.47060477, 0.49020672, 0.40550133],
      [0.53797149, -0.49020672, 0.59057215]],
     [[0.52103567, 0.47770318, 0.45720746],
      [0.63848153, -0.47770318, 0.18627359]],
     [[0.42403193, 0.46883875, 0.29457996],
      [0.29228492, -0.46883875, 0.61614434]],
     [[0.54241729, 0.48989333, 0.36434108],
      [0.48081261, -0.48989333, 0.1740178]],
     [[0.48713428, 0.484539, 0.03351721], [0.40485587, -0.484539, 0.26615021]],
     [[0.46114762, 0.48291636, 0.47846214],
      [0.3813956, -0.48291636, 0.23187181]],
     [[0.40333985, 0.45045316, 0.06916875],
      [0.29705281, -0.45045316, -0.35162915]],
     [[0.51022397, 0.49968443, 0.22460561],
      [0.51712798, -0.49968443, 0.25945148]],
     [[0.51817018, 0.47493092, -0.04629318],
      [0.56796335, -0.47493092, -0.3549728]],
     [[0.51254713, 0.49128671, 0.52905185],
      [0.59559333, -0.49128671, 0.36276049]],
     [[0.42318606, 0.49787579, 0.2999037],
      [0.44455141, -0.49787579, 0.38947103]]],
    dtype=np.float32)


class Obstacle3DObservation(NamedTuple):
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
    obstacle_pos: array


def obstacle_3d_task_reward(obs: Obstacle3DObservation) -> float:
    tip_position = obs.state_pos[-3:]
    dist = np.linalg.norm(obs.target_pos - tip_position)
    reward = -dist**2

    bonus_dist_threshold = 0.05  # cm
    if bonus_dist_threshold < dist < 2 * bonus_dist_threshold:
        reward += 0.5
    elif dist < bonus_dist_threshold:
        reward += 2.0

    return reward


class Obstacle3DTaskVisualizer:

    def __init__(self, window_size: Tuple[int, int] = (1024, 768)):
        """Initialize the 3D visualization for static obstacles and dynamic manipulator."""
        self._plotter = pv.Plotter(window_size=window_size,
                                   notebook=False,
                                   off_screen=False)
        self._plotter.set_background('white')
        self._plotter.add_axes()
        self._plotter.show_bounds(grid=False, location='outer', all_edges=True)

        self._dynamic_actors = []
        self._static_obstacle_names = []

        # self._add_static_obstacles(OBSTACLES_3D)
        self._add_goal_sphere()
        """Start the interactive visualization window."""
        self._plotter.show(interactive_update=True, auto_close=False)

    @staticmethod
    def _create_capsule_parts(resolution: int = 12):
        """Create individual capsule components (unit cylinder and spheres)."""
        cyl = pv.Cylinder(center=(0, 0, 0),
                          direction=(0, 0, 1),
                          radius=1.0,
                          height=1.0,
                          resolution=resolution,
                          capping=True)
        sph1 = pv.Sphere(radius=1.0,
                         center=(0, 0, 0),
                         theta_resolution=resolution,
                         phi_resolution=resolution)
        sph2 = pv.Sphere(radius=1.0,
                         center=(0, 0, 0),
                         theta_resolution=resolution,
                         phi_resolution=resolution)
        return cyl, sph1, sph2

    def _transform_capsule(self,
                           start: np.ndarray,
                           end: np.ndarray,
                           radius: float,
                           resolution: int = 12):
        """Return a correctly scaled capsule mesh from endpoints."""
        direction = end - start
        length = np.linalg.norm(direction)

        if length < 1e-10:
            return pv.Sphere(radius=radius, center=start)

        # Compute rotation
        dir_norm = direction / length
        z_axis = np.array([0, 0, 1])
        dot = np.clip(np.dot(z_axis, dir_norm), -1.0, 1.0)

        if np.allclose(dir_norm, z_axis):
            rot_mat = np.eye(3)
        elif np.allclose(dir_norm, -z_axis):
            rot_mat = R.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()
        else:
            axis = np.cross(z_axis, dir_norm)
            angle = np.arccos(dot)
            rot_mat = R.from_rotvec(axis / np.linalg.norm(axis) *
                                    angle).as_matrix()

        # Create unit parts
        cyl, sph1, sph2 = self._create_capsule_parts(resolution=resolution)

        # Transform cylinder
        cyl_tf = np.eye(4)
        cyl_tf[:3, :3] = rot_mat @ np.diag([radius, radius, length])
        cyl_tf[:3, 3] = (start + end) / 2
        cyl = cyl.transform(cyl_tf, inplace=False)

        # Transform sphere 1 (at start)
        s1_tf = np.eye(4)
        s1_tf[:3, :3] = rot_mat @ np.eye(3) * radius
        s1_tf[:3, 3] = start
        sph1 = sph1.transform(s1_tf, inplace=False)

        # Transform sphere 2 (at end)
        s2_tf = np.eye(4)
        s2_tf[:3, :3] = rot_mat @ np.eye(3) * radius
        s2_tf[:3, 3] = end
        sph2 = sph2.transform(s2_tf, inplace=False)

        return cyl.merge([sph1, sph2])

    def _add_goal_sphere(self):
        """Add a green semi-transparent goal sphere."""
        self._plotter.add_mesh(pv.Sphere(center=TARGET_POS, radius=0.05),
                               color='green',
                               opacity=0.7,
                               name="goal")

    def clear_static_obstacles(self):
        """Remove all previously added static obstacles."""
        for name in self._static_obstacle_names:
            self._plotter.remove_actor(name)
        self._static_obstacle_names.clear()

    def set_static_obstacles(self,
                             obstacles: np.ndarray,
                             colors: List[str] = None):
        """Replace static obstacles with a new set."""
        self.clear_static_obstacles()

        if colors is None:
            colors = ['gray'] * len(obstacles)

        for i, obstacle in enumerate(obstacles):
            start, end = obstacle
            color = colors[i] if i < len(colors) else 'gray'
            capsule = self._transform_capsule(start, end, OBSTACLE_RADIUS)
            name = f"static_capsule_{i}"
            self._plotter.add_mesh(capsule,
                                   color=color,
                                   opacity=1.0,
                                   name=name)
            self._static_obstacle_names.append(name)

    # def _add_static_obstacles(self,
    #                          obstacles: np.ndarray,
    #                          colors: List[str] = None):
    #     """Add static capsule obstacles to the visualization."""
    #     if colors is None:
    #         colors = ['gray'] * len(obstacles)
    #
    #     for i, obstacle in enumerate(obstacles):
    #         start, end = obstacle
    #         color = colors[i] if i < len(colors) else 'gray'
    #         capsule = self._transform_capsule(start, end, OBSTACLE_RADIUS)
    #         self._plotter.add_mesh(capsule, color=color, opacity=1.0, name=f"static_capsule_{i}")

    def render(self, arm_pos: np.ndarray):
        """Update the dynamic manipulator with new capsule segments."""

        # Lazy init the dynamic actors for the soft arm
        if not self._dynamic_actors:
            for i in range(len(arm_pos) - 1):
                start, end = arm_pos[i], arm_pos[i + 1]
                t = i / max(1, len(arm_pos) - 2)
                color = (t, 0, 1 - t)  # blue to red
                capsule = self._transform_capsule(start, end, RADIUS)
                actor = self._plotter.add_mesh(capsule,
                                               color=color,
                                               name=f"dynamic_capsule_{i}")
                self._dynamic_actors.append(actor)
        else:
            # After soft arms are created, reuse and update the transforms
            for i in range(len(arm_pos) - 1):
                start, end = arm_pos[i], arm_pos[i + 1]
                capsule = self._transform_capsule(start, end, RADIUS)
                self._dynamic_actors[i].mapper.dataset.shallow_copy(capsule)

        self._plotter.render()
        self._plotter.update()

    def close(self):
        """Close the visualization window."""
        self._plotter.close()
