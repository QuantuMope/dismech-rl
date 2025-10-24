import alf

from environments.dm_ik_env import DisMechInvKinEnv
from environments.ea_ik_env import ElasticaInvKinEnv

render = alf.define_config("render", False)
sim_framework = alf.define_config("sim_framework", "dismech")
assert sim_framework in ["dismech", "elastica"]

alf.import_config("common_sac_training_conf.py")
alf.import_config("control_conf.py").define_control_type("bend_twist")

alf.config("DisMechInvKinEnv",
           sim_timestep=5e-2,
           control_interval=2,
           timeout_steps=100,
           render=render)

alf.config("ElasticaInvKinEnv",
           sim_timestep=2e-4,
           control_interval=500,
           timeout_steps=100,
           render=render)

load_func = lambda *args, **kwargs: (DisMechInvKinEnv() if sim_framework ==
                                     "dismech" else ElasticaInvKinEnv())

alf.config("create_environment",
           env_name=f"{sim_framework}_soft_manipulator_ik",
           env_load_fn=load_func)
