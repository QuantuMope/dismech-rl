import alf
from confs.rl_alg_constructor import get_alg_constructor

utd_ratio = alf.define_config("utd_ratio", 4)

alf.config("create_environment",
           batch_size_per_env=1,
           num_parallel_environments=500)

alf.config("SacAlgorithm",
           target_update_tau=0.005,
           target_update_period=8,
           use_entropy_reward=False)

optimizer = alf.optimizers.AdamTF(lr=1e-3)
learning_alg_ctor = get_alg_constructor(optimizer=optimizer, policy_type="sac")

alf.config(
    "TrainerConfig",
    algorithm_ctor=learning_alg_ctor,
    initial_collect_steps=1000,
    temporally_independent_train_step=True,
    mini_batch_length=2,
    mini_batch_size=1024,
    unroll_length=1,
    num_iterations=10_000,
    num_updates_per_train_iter=utd_ratio,
    evaluate=False,
    summarize_first_interval=False,
    summarize_grads_and_vars=False,
    summarize_action_distributions=False,
    debug_summaries=False,
    num_checkpoints=1,
    whole_replay_buffer_training=False,
    # With 500 parallel envs, total replay buffer size is 2_000_000
    replay_buffer_length=4_000)
