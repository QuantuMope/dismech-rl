from functools import partial
from typing import Tuple, Literal

import alf
from alf.algorithms.agent import Agent
from alf.algorithms.ppo_algorithm import PPOAlgorithm
from alf.algorithms.ppo_loss import PPOLoss
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.nest.utils import NestConcat


def get_alg_constructor(optimizer: alf.optimizers.Optimizer,
                        hidden_layers: Tuple = (256, ) * 3,
                        policy_type: Literal["sac", "ppo"] = "sac"):
    """
    Helper function for constructing the policy network and algorithm.

    Args:
        optimizer: The optimizer to be used for training.
        hidden_layers: The hidden layers of the policy network.
        policy_type: "sac" or "ppo".
    """

    proj_net = partial(alf.networks.NormalProjectionNetwork,
                       state_dependent_std=True,
                       scale_distribution=True,
                       std_transform=alf.math.clipped_exp)

    actor_distribution_network_cls = partial(
        alf.networks.ActorDistributionNetwork,
        preprocessing_combiner=NestConcat(),
        fc_layer_params=hidden_layers,
        use_fc_ln=True,
        continuous_projection_net_ctor=proj_net)

    if policy_type == "sac":
        critic_network_cls = partial(
            alf.networks.CriticNetwork,
            observation_preprocessing_combiner=NestConcat(),
            action_preprocessing_combiner=NestConcat(),
            joint_fc_layer_params=hidden_layers,
            use_fc_ln=True)

        rl_alg_ctor = partial(SacAlgorithm,
                              actor_network_cls=actor_distribution_network_cls,
                              critic_network_cls=critic_network_cls)

    elif policy_type == "ppo":

        value_network_cls = partial(alf.networks.ValueNetwork,
                                    preprocessing_combiner=NestConcat(),
                                    fc_layer_params=hidden_layers,
                                    use_fc_ln=True)

        rl_alg_ctor = partial(
            PPOAlgorithm,
            actor_network_ctor=actor_distribution_network_cls,
            value_network_ctor=value_network_cls,
            loss_class=PPOLoss)

    alg_ctor = partial(Agent,
                       rl_algorithm_cls=rl_alg_ctor,
                       optimizer=optimizer)

    return alg_ctor
