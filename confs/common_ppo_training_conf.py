import alf
from confs.rl_alg_constructor import get_alg_constructor

utd_ratio = alf.define_config("utd_ratio", 4)

alf.config("create_environment",
           batch_size_per_env=1,
           num_parallel_environments=256)

alf.config("PPOLoss", entropy_regularization=5e-2, td_loss_weight=0.5)

optimizer = alf.optimizers.AdamTF(lr=1e-3)
learning_alg_ctor = get_alg_constructor(optimizer=optimizer, policy_type="ppo")

alf.config("TrainerConfig",
           algorithm_ctor=learning_alg_ctor,
           temporally_independent_train_step=True,
           mini_batch_length=1,
           mini_batch_size=1024,
           unroll_length=4,
           num_iterations=5_000_000,
           num_updates_per_train_iter=utd_ratio,
           evaluate=False,
           summarize_first_interval=False,
           summarize_grads_and_vars=False,
           summarize_action_distributions=False,
           debug_summaries=False,
           num_checkpoints=100,
           whole_replay_buffer_training=True,
           clear_replay_buffer=True)
