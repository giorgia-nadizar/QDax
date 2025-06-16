import os
import os
import sys
import time
from datetime import datetime
from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
import yaml
from jax._src.flatten_util import ravel_pytree

from hierarchy.parameters_mapper import map_output_to_nn_params, distill_archive
from qdax import environments
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids, MapElitesRepertoire
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.gp.hierarchical_encoding import compute_genome_to_hierarchical_step_fn
from qdax.core.gp.hierarchical_evaluation import gp_hierarchical_scoring_function_navigation_envs
from qdax.core.gp.individual import compute_genome_mask, compute_mutation_mask, generate_population, \
    compute_mutation_fn, compute_variation_mutation_fn
from qdax.core.gp.utils import update_config
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.utils.metrics import CSVLogger, default_qd_metrics


def _log_occupation(loop: int, iteration: int, occupation_dict: Dict, occupation_logger: CSVLogger) -> None:
    occupation_metrics = {"loop": loop, "iteration": iteration}
    for centroid_id in range(len(occupation_dict)):
        occupation_metrics["centroid"] = centroid_id
        occupation_metrics["occupation"] = occupation_dict[centroid_id]
        occupation_metrics["fitness"] = occupation_dict[centroid_id]
        occupation_logger.log(occupation_metrics)


def run_qd(config: Dict, target_path: str = "../results") -> None:
    extra_info = "_short" if config["episode_length"] < 1000 else ""
    run_name = f"hme_{config['env_name']}{extra_info}_{config['seed']}"

    env = environments.create(config["env_name"], episode_length=config["episode_length"], use_contact_forces=False)

    # Update config with env infogit
    config = update_config(config, env)

    # Init a random key
    random_key = jax.random.PRNGKey(config["seed"])

    # Init population of controllers
    genome_mask = compute_genome_mask(config, config["n_in"], config["n_out"])
    mutation_mask = compute_mutation_mask(config, config["n_out"])
    random_key, pop_key = jax.random.split(random_key)
    if config.get("fixed_outputs", False):
        fixed_outputs = jnp.arange(start=config["buffer_size"] - config["n_out"], stop=config["buffer_size"], step=1)
        population = generate_population(
            pop_size=config["pop_size"],
            genome_mask=genome_mask,
            rnd_key=pop_key,
            fixed_genome_trailing=fixed_outputs
        )
        config["p_mut_outputs"] = 0
    else:
        population = generate_population(pop_size=config["pop_size"], genome_mask=genome_mask, rnd_key=pop_key)

    # Create the initial environment states
    random_key, env_key = jax.random.split(random_key)
    env_keys = jnp.repeat(jnp.expand_dims(env_key, axis=0), repeats=config["pop_size"], axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_env_states = reset_fn(env_keys)
    program_initial_states = jnp.zeros((config["pop_size"], config["program_state_size"]))

    # Define scoring function
    encoding_fn = compute_genome_to_hierarchical_step_fn(env, config)
    policy_layer_sizes = config["policy_hidden_layer_sizes"] + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    fake_batch_key, random_key = jax.random.split(random_key, 2)
    fake_batch = jnp.zeros(shape=(env.observation_size - 2,))
    fake_params = policy_network.init(fake_batch_key, fake_batch)

    _, reconstruction_fn = ravel_pytree(fake_params)
    controllers_repertoire = MapElitesRepertoire.load(reconstruction_fn, config["controllers_repertoire_path"])
    distilled_repertoire = distill_archive(
        controllers_repertoire,
        config["target_centroids"],
        config["fitness_threshold"],
    )

    output_params_mapping_fn = partial(map_output_to_nn_params, repertoire=distilled_repertoire)
    scoring_fn = partial(
        gp_hierarchical_scoring_function_navigation_envs,
        init_states=(init_env_states, program_initial_states),
        episode_length=config["episode_length"],
        encoding_fn=encoding_fn,
        policy_network=policy_network,
        output_params_mapping_fn=output_params_mapping_fn
    )

    # Define emitter
    mutation_fn = compute_mutation_fn(genome_mask, mutation_mask)
    variation_fn = None
    variation_perc = 0.0
    if config["solver"] == "lgp":
        variation_fn = compute_variation_mutation_fn(genome_mask, mutation_mask)
        variation_perc = 1.0

    mixing_emitter = MixingEmitter(
        mutation_fn=mutation_fn,
        variation_fn=variation_fn,
        variation_percentage=variation_perc,
        batch_size=config["pop_size"]
    )

    # Define a metrics function
    reward_offset = environments.reward_offset[config["env_name"]]  # min reward value to ensure qd_score is positive
    qd_offset = reward_offset * config["episode_length"] if config["env_name"] not in ["kheperax", "robotmaze"] else 1.5

    metrics_function = partial(
        default_qd_metrics,
        qd_offset=qd_offset,
    )

    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
    )

    min_bd, max_bd = env.behavior_descriptor_limits
    centroids = compute_euclidean_centroids((33, 33), min_bd, max_bd)

    # Compute initial repertoire and emitter state
    repertoire, emitter_state, random_key = map_elites.init(population, centroids, random_key)

    print(run_name)
    repertoire_path = f"{target_path}/{run_name}/"
    os.makedirs(repertoire_path, exist_ok=True)
    with open(f"{repertoire_path}/config.yaml", "w") as file:
        yaml.dump(config, file)

    # Launch MAP-Elites iterations
    log_period = 10
    num_loops = int(config["n_iterations"] / log_period)

    headers = ["loop", "iteration", "qd_score", "max_fitness", "coverage", "time", "current_time"]

    csv_logger = CSVLogger(
        f"{target_path}/{run_name}.csv",
        header=headers
    )
    all_metrics = {}
    occupation_logger = CSVLogger(
        f"{target_path}/{run_name}_occupation.csv",
        header=["loop", "iteration", "centroid", "occupation", "fitness"]
    )

    # main loop
    map_elites_scan_update = map_elites.scan_update

    for i in range(num_loops):
        start_time = time.time()
        # main iterations
        (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
            map_elites_scan_update,
            (repertoire, emitter_state, random_key),
            (),
            length=log_period,
        )
        timelapse = time.time() - start_time
        current_time = datetime.now()

        # log metrics
        logged_metrics = {"time": timelapse, "loop": 1 + i, "iteration": 1 + i * log_period,
                          "current_time": current_time}

        for key, value in metrics.items():
            # take last value
            logged_metrics[key] = value[-1]

            # take all values
            if key in all_metrics.keys():
                all_metrics[key] = jnp.concatenate([all_metrics[key], value])
            else:
                all_metrics[key] = value

        csv_logger.log(logged_metrics)

        # occupation metrics
        _log_occupation(loop=1 + i, iteration=1 + i * log_period, occupation_dict=repertoire.occupation,
                        occupation_logger=occupation_logger)

        print(f"Loop: {i + 1}, "
              f"max_fitness: {logged_metrics['max_fitness']}, "
              f"time: {logged_metrics['time']}, "
              f"coverage: {logged_metrics['coverage']}")

        if i % 10 == 0:
            repertoire.save(path=repertoire_path)

    repertoire.save(path=repertoire_path)


if __name__ == '__main__':
    path = "../results"
    if len(sys.argv) > 1:
        path = sys.argv[1]

    print(f"Target path: {path}\n")
    n_itx = 10000

    seeds = range(1)

    print(f"Total runs: {len(seeds)}")
    env_name = "anttrap"
    episode_length = 250
    for sd in seeds:
        env_cfg = {
            "n_nodes": 50,
            "p_mut_inputs": 0.1,
            "p_mut_functions": 0.1,
            "p_mut_outputs": 0.3,
            "solver": "cgp",
            "env_name": env_name,
            "episode_length": episode_length,
            "pop_size": 100,
            "n_iterations": n_itx,
            "seed": sd,
            "fixed_outputs": True,
            "policy_hidden_layer_sizes": (256, 256),
            "controllers_repertoire_path": "../results/me_ant_omni_direct_0_finegrid/",
            "target_centroids": jnp.asarray([[x, y] for x in [-7.5, -2.5, 2.5, 7.5] for y in [-7.5, -2.5, 2.5, 7.5]]),
            "fitness_threshold": 160,
            "hierarchy": True,
        }

        run_qd(env_cfg, target_path=path)
