from typing import Dict

from brax.envs import Env

from qdax import environments
from qdax.core.gp.functions import available_functions, constants


def update_config(config: Dict, env: Env = None) -> Dict:
    if config["solver"] not in ["cgp", "lgp"]:
        raise ValueError("Solver must be either cgp or lgp.")

    if env is None:
        env = environments.create(
            config["env_name"],
            episode_length=config.get("episode_length", 1000),
        )

    config["n_functions"] = len(available_functions)
    config["n_constants"] = len(constants) if config.get("use_input_constants", True) else 0
    config["n_in_env"] = env.observation_size
    config["n_in"] = config["n_in_env"] + config["n_constants"]
    config["n_out"] = env.action_size if not config.get("symmetry", False) else int(env.action_size / 2)

    if config["solver"] == "cgp":
        config["buffer_size"] = config["n_in"] + config["n_nodes"]
        config["program_state_size"] = config["buffer_size"]
        config["genome_size"] = 3 * config["n_nodes"] + config["n_out"]
        levels_back = config.get("levels_back")
        if levels_back is not None and levels_back < config["n_in"]:
            config["levels_back"] = config["n_in"]
    else:
        config["n_registers"] = config["n_in"] + config["n_extra_registers"] + config["n_out"]
        config["program_state_size"] = config["n_registers"]
        config["genome_size"] = 5 * config["n_rows"]

    if config.get("symmetry", False):
        config["program_state_size"] = config["program_state_size"] * 2

    return config
