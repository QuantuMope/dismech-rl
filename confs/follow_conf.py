import alf

from environments.dm_follow_env import DisMechFollowEnv
from environments.ea_follow_env import ElasticaFollowEnv

render = alf.define_config("render", False)
ws_dim = alf.define_config("ws_dim", 3)
sim_framework = alf.define_config("sim_framework", "dismech")
assert sim_framework in ["dismech", "elastica"]
if sim_framework == "elastica":
    assert ws_dim != 2, "2D follow target not supported for elastica"

control_type = "bend" if ws_dim == 3 else "bend_2d"

alf.import_config("common_sac_training_conf.py")
alf.import_config("control_conf.py").define_control_type(control_type)

alf.config("DisMechFollowEnv",
           sim_timestep=5e-2,
           control_interval=2,
           timeout_steps=500,
           workspace_dim=ws_dim,
           render=render)

alf.config("ElasticaFollowEnv",
           sim_timestep=2e-4,
           control_interval=500,
           timeout_steps=500,
           render=render)

alf.config("ActionConverter", ws_dim=ws_dim)

load_func = lambda *args, **kwargs: (DisMechFollowEnv() if sim_framework ==
                                     "dismech" else ElasticaFollowEnv())

alf.config("create_environment",
           env_name=f"{sim_framework}_soft_manipulator_follow_target",
           env_load_fn=load_func)
