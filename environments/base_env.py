from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, NamedTuple

from utils.action_converter import ActionConverter, generate_ctrl_and_state_indices
from environments.utils.common import NUM_NODES

import alf
from alf.utils import common
from alf.environments.alf_environment import AlfEnvironment
from alf.data_structures import TimeStep, StepType


class BaseEnv(AlfEnvironment, ABC):
    """
    An abstract base class for constructing environments for RL.
    """

    def __init__(self,
                 sim_timestep: float,
                 control_interval: int,
                 timeout_steps: int,
                 render: bool = False):
        """
        Args:
            sim_timestep: Simulation timestep.
            control_interval: The number of sim steps for every control.
            timeout_steps: Number of steps before timing out an episode.
            render: Whether to render the environment.
        """
        super().__init__()
        # Derived classes should define these specs
        self._observation_spec = None
        self._action_spec = None

        self._sim_timestep = sim_timestep
        self._control_interval = control_interval
        self._timeout_counter = 0
        self._timeout_steps = timeout_steps
        self._render = render
        self._done = True
        self._simulator = None
        self._arm = None

        # We use a common scheme across all tasks.
        # We will control every 4th node. The observation will be twice as dense holding every 2nd node.
        self._ctrl_indices, self._state_indices = (
            generate_ctrl_and_state_indices(num_nodes=NUM_NODES,
                                            ctrl_spacing=4,
                                            state_spacing=2,
                                            offset=2))

        assert self._state_indices[-1] == NUM_NODES-1, \
            "Need to make sure that the tip position is included in the state."

        self._n_state_points = len(self._state_indices)
        self._n_ctrl_points = len(self._ctrl_indices)

        # This action converter's parameters are configured in control_conf.py to ensure
        # that all environments have the exact same control scheme.
        self._action_converter = ActionConverter(
            ctrl_indices=self._ctrl_indices, num_nodes=NUM_NODES)

    @abstractmethod
    def _create_new_sim(self):
        """
        Child classes should implement this function for creating new environments.
        """
        pass

    @abstractmethod
    def _generate_observation(self) -> NamedTuple:
        """
        Child classes should implement this function for generating observations.

        Returns:
            An observation where each field is an unbatched numpy array.
        """
        pass

    @abstractmethod
    def _compute_reward(self, obs: NamedTuple) -> float:
        """
        Child classes should implement this function for computing rewards.

        Args:
            obs: An observation from _generate_observation().

        Returns:
            A scalar reward.
        """
        pass

    @abstractmethod
    def render(self, mode='rgb_array'):
        """
        Child classes should implement this function for rendering.

        This function will be called when evaluating a checkpoint via alf.bin.play

        Args:
            mode: Unused (for alf API compliance)
        """
        pass

    def _custom_sim_params(self):
        """
        Child classes can implement this function for customizing the simulation parameters.

        Note that all structures should be created **before** defining forces.
        """
        pass

    @abstractmethod
    def _custom_step(self, action: np.ndarray):
        """
        Child classes should implement this function for defining the env step.

        This function is called in _step() below.

        Args:
            action: The action from the policy.
        """
        pass

    def _step(self, action: np.ndarray) -> TimeStep:
        if self._done:
            return self._reset()

        step_success = True
        try:
            self._custom_step(action)
        except RuntimeError:
            step_success = False

        obs = self._generate_observation()

        # If simulation fails for any reason, we terminate the
        # episode with a large negative reward.
        if not step_success:
            step_type = StepType.LAST
            discount = 0.0
            reward = -10
            self._done = True

        elif self._timeout_counter == self._timeout_steps:
            step_type = StepType.LAST
            discount = 1.0
            reward = self._compute_reward(obs)
            self._done = True

        else:
            step_type = StepType.MID
            discount = 1.0
            reward = self._compute_reward(obs)

        # This would indicate that things have exploded.
        # In this case treat it the same as if simulation failed.
        if reward < -10 or np.isnan(reward):
            step_type = StepType.LAST
            discount = 0.0
            reward = -10
            self._done = True

        self._timeout_counter += 1

        return TimeStep(step_type=step_type,
                        observation=obs,
                        reward=np.float32(reward),
                        discount=np.float32(discount),
                        prev_action=action,
                        env_info={},
                        env_id=np.int32(0))

    def _custom_reset(self):
        pass

    def _reset(self) -> TimeStep:
        self._create_new_sim()
        self._action_converter.reset()
        self._custom_reset()

        self._timeout_counter = 0
        self._done = False

        obs = self._generate_observation()

        return TimeStep(step_type=StepType.FIRST,
                        observation=obs,
                        reward=np.float32(0.0),
                        discount=np.float32(1.0),
                        prev_action=self.action_spec().numpy_zeros(),
                        env_info={},
                        env_id=np.int32(0))

    def seed(self, seed: Optional[int] = None):
        common.set_random_seed(seed)

    def observation_spec(self) -> alf.NestedTensorSpec:
        assert self._observation_spec is not None, \
            "Observation Spec must be specified."
        return self._observation_spec

    def action_spec(self) -> alf.NestedBoundedTensorSpec:
        assert self._action_spec is not None, \
            "Action Spec must be specified."
        return self._action_spec

    def env_info_spec(self) -> alf.NestedTensorSpec:
        return {}

    def reward_spec(self) -> alf.TensorSpec:
        return alf.TensorSpec(())
