import alf

from environments.dm_obs_2d_env import DisMechObstacle2DEnv
from environments.ea_obs_2d_env import ElasticaObstacle2DEnv

render = alf.define_config("render", False)
sim_framework = alf.define_config("sim_framework", "dismech")
assert sim_framework in ["dismech", "elastica"]

alf.import_config("common_sac_training_conf.py")
alf.import_config("control_conf.py").define_control_type("contact_bend_2d")

alf.config("DisMechObstacle2DEnv",
           sim_timestep=5e-2,
           control_interval=10,
           timeout_steps=50,
           render=render)

alf.config("ElasticaObstacle2DEnv",
           sim_timestep=2e-4,
           control_interval=2500,
           timeout_steps=50,
           render=render)

load_func = lambda *args, **kwargs: (DisMechObstacle2DEnv() if sim_framework ==
                                     "dismech" else ElasticaObstacle2DEnv())

alf.config("create_environment",
           env_name=f"{sim_framework}_soft_manipulator_2d_obstacles",
           env_load_fn=load_func)
