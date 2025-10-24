from typing import Tuple, Dict, Literal
import numpy as np

import alf


def generate_ctrl_and_state_indices(
        num_nodes: int, ctrl_spacing: int, state_spacing: int,
        offset: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate control and state indices with state indices being more dense.
    Args:
        num_nodes: Total number of nodes in the system
        ctrl_spacing: Spacing between control nodes
        state_spacing: Spacing between state nodes. This should be a multiple of ctrl_spacing.
        offset: Starting offset for indices.
    Returns:
        (ctrl_indices, state_indices)
        - ctrl_indices: indices for control points
        - state_indices: indices for state observation (twice as dense)
    """
    assert ctrl_spacing > state_spacing and ctrl_spacing % state_spacing == 0
    ctrl_indices = np.array(
        [i for i in range(offset, num_nodes - 1, ctrl_spacing)],
        dtype=np.int32)
    state_indices = np.array(
        [i for i in range(offset, num_nodes, state_spacing)], dtype=np.int32)

    assert 0 not in ctrl_indices and num_nodes-1 not in ctrl_indices, \
        "Control indices should only include interior nodes."

    return ctrl_indices, state_indices


def generate_segment_boundaries(num_nodes: int,
                                ctrl_indices: np.ndarray) -> np.ndarray:
    """
    Generate segment boundaries based on control indices.
    Args:
        num_nodes: Total number of nodes in the system
        ctrl_indices: Indices of control nodes
    Returns:
        Indices marking the boundaries of segments
    """
    # Calculate midpoints between control indices
    segment_boundaries = [1]  # Start with 0

    for i in range(len(ctrl_indices) - 1):
        midpoint = (ctrl_indices[i] + ctrl_indices[i + 1]) // 2
        segment_boundaries.append(midpoint)

    segment_boundaries.append(num_nodes - 1)  # End with num_nodes

    return np.array(segment_boundaries, dtype=np.int32)


def apply_uniform_curvature(ctrl_indices: np.ndarray,
                            segment_boundaries: np.ndarray,
                            policy_action: np.ndarray,
                            dm_curv_action: np.ndarray, ws_dim: int):
    """
    Apply curvature uniformly across segments.
    Args:
        ctrl_indices: Indices of control nodes
        segment_boundaries: Indices marking the boundaries of segments
        dm_curv_action: DisMech curvature matrix shaped as (num_ctrl, 4).
            The values of this matrix will be changed in-place.
    """
    assert ws_dim in [2, 3]
    if ws_dim == 3:
        policy_action = policy_action.reshape((-1, 2))
    else:
        policy_action = policy_action.reshape((-1, 1))
    # Apply uniform curvature in each segment
    for i in range(len(segment_boundaries) - 1):
        start_bound = segment_boundaries[i]
        end_bound = segment_boundaries[i + 1]

        # Get the number of nodes in this segment
        nodes_in_segment = end_bound - start_bound

        # Skip if segment is empty
        if nodes_in_segment <= 0:
            continue

        # Control index for this segment
        control_idx = min(i, len(ctrl_indices) - 1)

        # Calculate uniform curvature values (divided by number of nodes)
        uniform_kappa1 = policy_action[control_idx, 0] / nodes_in_segment

        # Assign uniform values to all nodes in this segment
        dm_curv_action[start_bound:end_bound, 2] = uniform_kappa1

        if ws_dim == 3:
            uniform_kappa2 = policy_action[control_idx, 1] / nodes_in_segment
            dm_curv_action[start_bound:end_bound, 3] = uniform_kappa2


def apply_uniform_theta(ctrl_indices: np.ndarray,
                        segment_boundaries: np.ndarray,
                        policy_action: np.ndarray,
                        dm_theta_action: np.ndarray):
    """
    Apply curvature uniformly across segments.
    Args:
        ctrl_indices: Indices of control nodes
        segment_boundaries: Indices marking the boundaries of segments
        dm_theta_action: A DisMech theta matrix shaped as (num_ctrl, 3).
            The values of this matrix will be changed in-place.
    """
    # Apply uniform theta in each segment
    for i in range(len(segment_boundaries) - 1):
        start_bound = segment_boundaries[i]
        end_bound = segment_boundaries[i + 1]

        # Get the number of nodes in this segment
        nodes_in_segment = end_bound - start_bound

        # Skip if segment is empty
        if nodes_in_segment <= 0:
            continue

        # Control index for this segment
        control_idx = min(i, len(ctrl_indices) - 1)

        # Calculate uniform theta values (divided by number of nodes)
        uniform_theta = policy_action[control_idx] / nodes_in_segment

        # Assign uniform values to all nodes in this segment
        dm_theta_action[start_bound:end_bound, 2] = uniform_theta


@alf.configurable(blacklist=['ctrl_indices', 'num_nodes'])
class ActionConverter:
    """
    This class converts numpy actions outputted from a policy into
    DisMech compatible actions.
    Note that this class assumes a single limb currently.
    Will need to modify to support multi-limb actions.
    """

    def __init__(
        self,
        ctrl_indices: np.ndarray,
        num_nodes: int,
        ws_dim: int = 3,
        delta_kappa_scale: float = 0.05,
        kappa_bar_range: Tuple[float, float] = (-np.pi / 2, np.pi / 2),
        include_twist: bool = False,
        delta_theta_scale: float = 0.05,
        twist_bar_range: Tuple[float, float] = (-np.pi / 2, np.pi / 2),
        smooth_action: bool = True,
    ):
        assert delta_kappa_scale > 0.0, "delta_kappa_scale must be greater than 0.0"
        self._delta_kappa_scale = delta_kappa_scale
        self._kappa_bar_range = kappa_bar_range
        self._ctrl_indices = ctrl_indices
        self._n_ctrl_points = len(self._ctrl_indices)
        self._num_nodes = num_nodes
        self._ws_dim = ws_dim
        self._include_twist = include_twist
        self._delta_theta_scale = delta_theta_scale
        self._twist_bar_range = twist_bar_range
        self._smooth_action = smooth_action

        self._curr_kappa_bar = np.zeros(
            (self._n_ctrl_points * (self._ws_dim - 1), ), dtype=np.float32)
        self._prev_kappa_bar = self._curr_kappa_bar.copy()

        if self._include_twist:
            assert self._ws_dim == 3, "Twist action is only valid for 3D workspaces."
            self._curr_theta_bar = np.zeros((self._n_ctrl_points, ),
                                            dtype=np.float32)
            self._prev_theta_bar = self._curr_theta_bar.copy()

        if not self._smooth_action:
            self._dm_curvature_action = np.zeros((self._n_ctrl_points, 4))
            self._dm_curvature_action[:, 1] = self._ctrl_indices
            if self._include_twist:
                self._dm_theta_action = np.zeros((self._n_ctrl_points, 3))
                self._dm_theta_action[:, 1] = self._ctrl_indices
        else:
            self._dm_curvature_action = np.zeros((self._num_nodes - 2, 4))
            self._dm_curvature_action[:, 1] = np.arange(1, self._num_nodes - 1)
            if self._include_twist:
                self._dm_theta_action = np.zeros((self._num_nodes - 2, 3))
                self._dm_theta_action[:, 1] = np.arange(1, self._num_nodes - 1)
            self._segment_boundaries = generate_segment_boundaries(
                self._num_nodes, self._ctrl_indices)

    def action_spec(self) -> alf.BoundedTensorSpec:
        num_dofs = self._n_ctrl_points * (self._ws_dim - 1)
        if self._include_twist:
            num_dofs += self._n_ctrl_points
        return alf.BoundedTensorSpec((num_dofs, ), minimum=-1.0, maximum=1.0)

    @property
    def curr_kappa_bar(self) -> np.ndarray:
        return self._curr_kappa_bar.astype(np.float32)

    @property
    def curr_kappa_bar_smooth(self) -> np.ndarray:
        return self._dm_curvature_action.astype(np.float32)

    @property
    def curr_theta_bar(self) -> np.ndarray:
        return self._curr_theta_bar.astype(np.float32)

    def transform_action(
        self,
        action: np.ndarray,
        output_type: Literal["dismech", "elastica"],
        interpolate_steps: int = 0,
        voronoi_lengths: np.ndarray = None,
    ) -> Dict[str, np.ndarray]:

        if output_type == "elastica":
            assert voronoi_lengths is not None, \
                "voronoi_lengths must be provided for elastica output type"

        curvature_twist_boundary = self._n_ctrl_points * (self._ws_dim - 1)

        # Scale delta kappa bar and add it to the current kappa bar
        curv_action_raw = action[:curvature_twist_boundary]
        self._curr_kappa_bar += curv_action_raw * self._delta_kappa_scale

        # Clip the kappa bar to the valid range
        self._curr_kappa_bar = np.clip(self._curr_kappa_bar,
                                       self._kappa_bar_range[0],
                                       self._kappa_bar_range[1],
                                       dtype=np.float32)

        dismech_action = {}

        if interpolate_steps > 0:
            kappa_bar_target = (self._curr_kappa_bar -
                                self._prev_kappa_bar) / interpolate_steps
            action_type = "delta_curvature"
        else:
            kappa_bar_target = self._curr_kappa_bar
            action_type = "curvature"

        if not self._smooth_action:
            self._dm_curvature_action[:, 2] = kappa_bar_target[:self.
                                                               _n_ctrl_points]
            if self._ws_dim == 3:
                self._dm_curvature_action[:, 3] = kappa_bar_target[
                    self._n_ctrl_points:]
        else:
            apply_uniform_curvature(self._ctrl_indices,
                                    self._segment_boundaries, kappa_bar_target,
                                    self._dm_curvature_action, self._ws_dim)

        dismech_action[action_type] = self._dm_curvature_action

        # Do the same for theta if we are actuating twist.
        if self._include_twist:
            theta_action_raw = action[curvature_twist_boundary:]
            self._curr_theta_bar += theta_action_raw * self._delta_theta_scale
            self._curr_theta_bar = np.clip(self._curr_theta_bar,
                                           self._twist_bar_range[0],
                                           self._twist_bar_range[1],
                                           dtype=np.float32)

            if interpolate_steps > 0:
                twist_bar_target = (self._curr_theta_bar -
                                    self._prev_theta_bar) / interpolate_steps
                action_type = "delta_theta"
            else:
                twist_bar_target = self._curr_theta_bar
                action_type = "theta"

            if not self._smooth_action:
                self._dm_theta_action[:, 2] = self._curr_theta_bar
            else:
                apply_uniform_theta(self._ctrl_indices,
                                    self._segment_boundaries, twist_bar_target,
                                    self._dm_theta_action)

            dismech_action[action_type] = self._dm_theta_action

        self._prev_kappa_bar = self._curr_kappa_bar.copy()
        if self._include_twist:
            self._prev_theta_bar = self._curr_theta_bar.copy()

        if output_type == "elastica":
            delta_curvature = dismech_action["delta_curvature"][:, 2:].T
            # Need to change curvature polarity for second index to match DisMech's frame.
            delta_curvature[1] *= -1.0

            # DisMech uses integrated curvature while Elastica does not
            # Here we'll convert to unintegrated curvature.
            delta_curvature /= voronoi_lengths

            elastica_action = {"delta_curvature": delta_curvature}

            if self._include_twist:
                delta_twist = dismech_action["delta_theta"][:, 2].T
                delta_twist /= voronoi_lengths

                # Need to change twist polarity to match DisMech's frame.
                elastica_action["delta_theta"] = -delta_twist

            return elastica_action

        return dismech_action

    def reset(self):
        self._curr_kappa_bar[:] = 0.0
        self._prev_kappa_bar[:] = 0.0
        if self._include_twist:
            self._curr_theta_bar[:] = 0.0
            self._prev_theta_bar[:] = 0.0
