import alf

from environments.dm_obs_3d_env import DisMechObstacle3DEnv
from environments.ea_obs_3d_env import ElasticaObstacle3DEnv

render = alf.define_config("render", False)
sim_framework = alf.define_config("sim_framework", "dismech")
assert sim_framework in ["dismech", "elastica"]

alf.import_config("common_sac_training_conf.py")
alf.import_config("control_conf.py").define_control_type("contact_bend")

alf.config("DisMechObstacle3DEnv",
           sim_timestep=5e-2,
           control_interval=10,
           timeout_steps=50,
           render=render)

alf.config("ElasticaObstacle3DEnv",
           sim_timestep=2e-4,
           control_interval=2500,
           timeout_steps=50,
           render=render)

load_func = lambda *args, **kwargs: (DisMechObstacle3DEnv() if sim_framework ==
                                     "dismech" else ElasticaObstacle3DEnv())

alf.config("create_environment",
           env_name=f"{sim_framework}_soft_manipulator_3d_obstacles",
           env_load_fn=load_func)
