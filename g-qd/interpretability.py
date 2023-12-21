import os
import sys
from typing import Dict

import jax.numpy as jnp

from qdax import environments
from qdax.core.gp.interpretability_utils import evaluate_interpretability
from qdax.core.gp.utils import update_config
from qdax.core.gp.visualization_utils import expression_from_genome
from qdax.environments.kheperax.task import KheperaxConfig, KheperaxTargetedTask
from qdax.types import Genotype, Fitness


def interpretability_of_repertoire(
        config: Dict,
        repertoire_base_path: str,
) -> None:
    genotypes = jnp.load(repertoire_base_path + "genotypes.npy")
    fitnesses = jnp.load(repertoire_base_path + "fitnesses.npy")
    interpretabilities = interpretability_of_genotypes(config, genotypes, fitnesses)
    jnp.save(f"{repertoire_base_path}interpretabilities.npy", interpretabilities)


def interpretability_of_genotypes(
        config: Dict,
        genotypes: Genotype,
        fitnesses: Fitness
) -> jnp.ndarray:
    interpretabilities = []
    for pos in range(len(genotypes)):
        if fitnesses.at[pos].get() > -jnp.inf:
            expressions = expression_from_genome(genotypes.at[pos].get(), config)
            pos_interpretabilites = [evaluate_interpretability(expr) for expr in expressions]
            interpretabilities.append(sum(pos_interpretabilites) / len(pos_interpretabilites))
        else:
            interpretabilities.append(-jnp.inf)

    return jnp.asarray(interpretabilities)


if __name__ == '__main__':

    base_path = "../results"
    if len(sys.argv) > 1:
        base_path = sys.argv[1]

    seeds = range(1)
    envs = ["pointmaze", "robotmaze", "hopper_uni", "walker2d_uni"]

    for seed in seeds:
        for env_name in envs:
            cfg_me = {
                "n_nodes": 50,
                "solver": "cgp",
            }

            # Init environment
            if env_name in ["kheperax", "robotmaze"]:
                config_kheperax = KheperaxConfig.get_default()
                env = KheperaxTargetedTask.create_environment(config_kheperax)
            else:
                env = environments.create(env_name, episode_length=10, legacy_spring=False)

            # Update config with env info
            cfg_me = update_config(cfg_me, env)

            for run_type in ["bimapelites_both_cgp_", "bimapelites_s1_cgp_", "bimapelites_s2_cgp_", "mapelites_cgp_"]:

                full_path = f"{base_path}/{run_type}{env_name}_{seed}/"
                print(full_path)
                if run_type.startswith("m"):
                    interpretability_of_repertoire(cfg_me, full_path)
                else:
                    for i in range(1, 3):
                        interpretability_of_repertoire(cfg_me, f"{full_path}r{i}_")
