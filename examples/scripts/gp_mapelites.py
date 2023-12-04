import copy
import os
import sys

import time
from datetime import datetime
from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
import wandb

from qdax.core.bi_map_elites import BiMAPElites, sampling_function
from qdax.core.gp.encoding import compute_genome_to_step_fn
from qdax.core.gp.evaluation import gp_scoring_function_brax_envs
from qdax.core.gp.graph_utils import get_graph_descriptor_extractor
from qdax.core.gp.individual import compute_genome_mask, compute_mutation_mask, generate_population, \
    compute_mutation_fn, compute_variation_mutation_fn
from qdax.core.gp.utils import update_config
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax import environments

from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.environments.kheperax.task import KheperaxConfig, KheperaxTargetedTask
from qdax.types import RNGKey, Genotype

from qdax.utils.metrics import CSVLogger, default_biqd_metrics, default_qd_metrics, qd_metrics_with_bi_tracking

from wandb.sdk.wandb_run import Run


def _log_occupation(loop: int, iteration: int, occupation_dict: Dict, occupation_logger: CSVLogger) -> None:
    occupation_metrics = {"loop": loop, "iteration": iteration}
    for centroid_id in range(len(occupation_dict)):
        occupation_metrics["centroid"] = centroid_id
        occupation_metrics["occupation"] = occupation_dict[centroid_id]
        occupation_metrics["fitness"] = occupation_dict[centroid_id]
        occupation_logger.log(occupation_metrics)


def run_common(config: Dict, target_path: str = "../../results") -> None:
    bi_map_elites = config.get("bi_map", False)

    run_name = f"bimapelites_{config['sampler']}" if bi_map_elites else "mapelites"
    run_name += f"_{config['solver']}_{config['env_name']}_{config['seed']}"

    api = wandb.Api(timeout=40)
    wb_run = wandb.init(
        config=config,
        project="cgpax",
        name=run_name
    )

    # Init environment
    if config["env_name"] == "kheperax":
        config_kheperax = KheperaxConfig.get_default()
        env = KheperaxTargetedTask.create_environment(config_kheperax)
    else:
        env = environments.create(config["env_name"], episode_length=config["episode_length"], legacy_spring=False)

    bd_extraction_fn = environments.behavior_descriptor_extractor[config["env_name"]]
    n_behavior_descriptors = 2 if config["env_name"] == "kheperax" else env.behavior_descriptor_length

    # Update config with env info
    config = update_config(config, env)

    # Init a random key
    random_key = jax.random.PRNGKey(config["seed"])

    # Init population of controllers
    genome_mask = compute_genome_mask(config, config["n_in"], config["n_out"])
    mutation_mask = compute_mutation_mask(config, config["n_out"])
    random_key, pop_key = jax.random.split(random_key)
    population = generate_population(pop_size=config["pop_size"], genome_mask=genome_mask, rnd_key=pop_key)

    # Create the initial environment states
    random_key, env_key = jax.random.split(random_key)
    env_keys = jnp.repeat(jnp.expand_dims(env_key, axis=0), repeats=config["pop_size"], axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_env_states = reset_fn(env_keys)
    program_initial_states = jnp.zeros((config["pop_size"], config["program_state_size"]))

    # Graph descriptor extractor
    descriptor_extraction_fn, n_graph_descriptors = get_graph_descriptor_extractor(config)

    # Define scoring function
    encoding_fn = compute_genome_to_step_fn(env, config)
    scoring_fn = partial(
        gp_scoring_function_brax_envs,
        init_states=(init_env_states, program_initial_states),
        episode_length=config["episode_length"],
        encoding_fn=encoding_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
        graph_descriptor_extractor=descriptor_extraction_fn
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
    qd_offset = reward_offset * config["episode_length"] if config["env_name"] != "kheperax" else 1.5

    behavior_descriptors_ids = jnp.arange(n_behavior_descriptors)
    graph_descriptors_ids = jnp.arange(
        start=n_behavior_descriptors,
        stop=n_behavior_descriptors + n_graph_descriptors,
        step=1
    )

    if bi_map_elites:
        metrics_function = partial(
            default_biqd_metrics,
            qd_offset=qd_offset,
        )

        sampling_id_fn = sampling_function(config["sampler"])
        map_elites = BiMAPElites(
            scoring_function=scoring_fn,
            emitter=mixing_emitter,
            metrics_function=metrics_function,
            descriptors_indexes1=behavior_descriptors_ids,
            descriptors_indexes2=graph_descriptors_ids,
            sampling_id_function=sampling_id_fn
        )

        config["n_behavior_descriptors"] = n_behavior_descriptors
        config["n_graph_descriptors"] = n_graph_descriptors
    else:

        if not config.get("track_separate", True):
            metrics_function = partial(
                default_qd_metrics,
                qd_offset=qd_offset,
            )

        else:

            print(f"{datetime.now()} - metrics centroids computation starting...")
            # Compute the graph descriptors uniformly
            graph_centroids, random_key = compute_cvt_centroids(
                num_descriptors=n_graph_descriptors,
                num_init_cvt_samples=config["qd"]["n_init_cvt_samples"],
                num_centroids=config["qd"]["n_centroids"],
                minval=config["qd"]["min_gd"],
                maxval=config["qd"]["max_gd"],
                random_key=random_key,
            )

            # Compute the behavior descriptors uniformly
            behavior_centroids, random_key = compute_cvt_centroids(
                num_descriptors=n_behavior_descriptors,
                num_init_cvt_samples=config["qd"]["n_init_cvt_samples"],
                num_centroids=config["qd"]["n_centroids"],
                minval=config["qd"]["min_bd"],
                maxval=config["qd"]["max_bd"],
                random_key=random_key,
            )
            print(f"{datetime.now()} - metrics centroids computation done")

            metrics_function = partial(
                qd_metrics_with_bi_tracking,
                qd_offset=qd_offset,
                centroids1=behavior_centroids,
                centroids2=graph_centroids,
                descriptors_indexes1=behavior_descriptors_ids,
                descriptors_indexes2=graph_descriptors_ids
            )

        map_elites = MAPElites(
            scoring_function=scoring_fn,
            emitter=mixing_emitter,
            metrics_function=metrics_function,
        )

        config["n_descriptors"] = n_behavior_descriptors + n_graph_descriptors

    runner = bi_mapelites_run if bi_map_elites else mapelites_run
    runner(
        config=config,
        random_key=random_key,
        init_population=population,
        map_elites=map_elites,
        target_path=target_path,
        wb_run=wb_run
    )


def bi_mapelites_run(
    config: Dict,
    random_key: RNGKey,
    init_population: Genotype,
    map_elites: MAPElites,
    target_path: str,
    wb_run: Run
) -> None:
    # Compute the centroids
    print(f"{datetime.now()} - start centroids computation")

    # Compute the graph descriptors uniformly
    graph_centroids, random_key = compute_cvt_centroids(
        num_descriptors=config["n_graph_descriptors"],
        num_init_cvt_samples=config["qd"]["n_init_cvt_samples"],
        num_centroids=config["qd"]["n_centroids"],
        minval=config["qd"]["min_gd"],
        maxval=config["qd"]["max_gd"],
        random_key=random_key,
    )

    # Compute the behavior descriptors uniformly
    behavior_centroids, random_key = compute_cvt_centroids(
        num_descriptors=config["n_behavior_descriptors"],
        num_init_cvt_samples=config["qd"]["n_init_cvt_samples"],
        num_centroids=config["qd"]["n_centroids"],
        minval=config["qd"]["min_bd"],
        maxval=config["qd"]["max_bd"],
        random_key=random_key,
    )
    print(f"{datetime.now()} - centroids computation done")

    # Compute initial repertoire and emitter state
    repertoire, emitter_state, random_key = map_elites.init(
        init_genotypes=init_population,
        centroids1=behavior_centroids,
        centroids2=graph_centroids,
        random_key=random_key
    )

    # Launch MAP-Elites iterations
    log_period = 10
    num_loops = int(config["n_iterations"] / log_period)

    headers = ["loop", "iteration", "qd_score1", "max_fitness", "coverage1", "time", "current_time", "qd_score2",
               "coverage2"]

    csv_logger = CSVLogger(
        f"{target_path}/{wb_run.name}.csv",
        header=headers
    )
    all_metrics = {}
    occupation_logger1 = CSVLogger(
        f"{target_path}/{wb_run.name}_occupation.csv",
        header=["loop", "iteration", "centroid", "occupation", "fitness"]
    )
    occupation_logger2 = CSVLogger(
        f"{target_path}/{wb_run.name}_extra_occupation.csv",
        header=["loop", "iteration", "centroid", "occupation", "fitness"]
    )

    # main loop
    scan_update_fn = map_elites.scan_update
    print(wb_run.name)
    for i in range(num_loops):
        start_time = time.time()
        # main iterations
        (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
            scan_update_fn,
            (repertoire, emitter_state, random_key),
            (),
            length=log_period,
        )
        timelapse = time.time() - start_time
        current_time = datetime.now()

        # log metrics
        logged_metrics = {
            "time": timelapse,
            "loop": 1 + i,
            "iteration": 1 + i * log_period,
            "current_time": current_time
        }

        for key, value in metrics.items():
            # take last value
            logged_metrics[key] = value[-1]

            # take all values
            if key in all_metrics.keys():
                all_metrics[key] = jnp.concatenate([all_metrics[key], value])
            else:
                all_metrics[key] = value

        csv_logger.log(logged_metrics)
        wb_run.log(logged_metrics)

        # occupation metrics
        _log_occupation(loop=1 + i, iteration=1 + i * log_period, occupation_dict=repertoire.repertoire1.occupation,
                        occupation_logger=occupation_logger1)
        _log_occupation(loop=1 + i, iteration=1 + i * log_period, occupation_dict=repertoire.repertoire2.occupation,
                        occupation_logger=occupation_logger2)

        print(f"Loop: {i + 1}, "
              f"max_fitness: {logged_metrics['max_fitness']}, "
              f"time: {logged_metrics['time']}, "
              f"coverage: {logged_metrics['coverage1']}")

    repertoire_path = f"{target_path}/{wb_run.name}/"
    os.makedirs(repertoire_path, exist_ok=True)
    repertoire.save(path=repertoire_path)

    wb_run.finish()


def mapelites_run(
    config: Dict,
    random_key: RNGKey,
    init_population: Genotype,
    map_elites: MAPElites,
    target_path: str,
    wb_run: Run
) -> None:
    # Compute the centroids
    print(f"{datetime.now()} - start {config['n_descriptors']} centroids computation")

    # Compute the descriptors uniformly
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=config["n_descriptors"],
        num_init_cvt_samples=config["qd"]["n_init_cvt_samples"],
        num_centroids=config["qd"]["n_centroids"],
        minval=config["qd"]["min_bd"],
        maxval=config["qd"]["max_bd"],
        random_key=random_key,
    )

    print(f"{datetime.now()} - centroids computation done")

    # Compute initial repertoire and emitter state
    repertoire, emitter_state, random_key = map_elites.init(init_population, centroids, random_key)

    # Launch MAP-Elites iterations
    log_period = 10
    num_loops = int(config["n_iterations"] / log_period)

    headers = ["loop", "iteration", "qd_score", "max_fitness", "coverage", "time", "current_time"]

    if config.get("track_separate", True):
        headers.extend(["coverage1", "coverage2"])

    csv_logger = CSVLogger(
        f"{target_path}/{wb_run.name}.csv",
        header=headers
    )
    all_metrics = {}
    occupation_logger = CSVLogger(
        f"{target_path}/{wb_run.name}_occupation.csv",
        header=["loop", "iteration", "centroid", "occupation", "fitness"]
    )

    # main loop
    map_elites_scan_update = map_elites.scan_update
    print({wb_run.name})
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
        wb_run.log(logged_metrics)

        # occupation metrics
        _log_occupation(loop=1 + i, iteration=1 + i * log_period, occupation_dict=repertoire.occupation,
                        occupation_logger=occupation_logger)

        print(f"Loop: {i + 1}, "
              f"max_fitness: {logged_metrics['max_fitness']}, "
              f"time: {logged_metrics['time']}, "
              f"coverage: {logged_metrics['coverage']}")

    repertoire_path = f"{target_path}/{wb_run.name}/"
    os.makedirs(repertoire_path, exist_ok=True)
    repertoire.save(path=repertoire_path)

    wb_run.finish()


if __name__ == '__main__':
    path = "../../results"
    if len(sys.argv) > 1:
        path = sys.argv[1]

    print(f"Target path: {path}\n")
    n_itx = 10000

    seeds = range(5, 10)
    sampling_methods = ["both", "s1", "s2", "me"]

    custom_seed = os.environ.get("SEED")
    if custom_seed is not None:
        seeds = [int(custom_seed)]
        print(f"Custom seed: {custom_seed}")

    custom_sampling = os.environ.get("SAMPLING")
    if custom_sampling is not None:
        sampling_methods = [custom_sampling]
        print(f"Custom sampling: {custom_sampling}")

    env_duration_tuples = [
        ("pointmaze", 100),
        ("kheperax", 250),
        ("hopper_uni", 1000),
        ("walker2d_uni", 1000),
    ]

    print(f"Total runs: {len(seeds) * len(env_duration_tuples) * len(sampling_methods)}")
    for env_name, episode_length in env_duration_tuples:
        for sd in seeds:
            # map elites with two repertoires
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
                "qd": {
                    "n_init_cvt_samples": 500000,
                    "n_centroids": 1024,
                    "min_bd": 0.0,
                    "max_bd": 1.0,
                    "min_gd": 0.0,
                    "max_gd": 1.0
                },
                "seed": sd,
                "graph_descriptors": ["complexity", "function_arities_fraction"],
            }

            for sampl in sampling_methods:
                current_cfg = copy.deepcopy(env_cfg)
                if not sampl.startswith("me"):
                    current_cfg["sampler"] = sampl
                    current_cfg["bi_map"] = True

                else:
                    # standard map elites
                    current_cfg["bi_map"] = False
                    current_cfg["track_separate"] = True

                run_common(current_cfg, target_path=path)
