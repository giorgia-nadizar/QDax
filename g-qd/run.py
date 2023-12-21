import copy
import functools
import os
import sys

import time
from datetime import datetime
from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import wandb

from qdax.baselines.genetic_algorithm import GeneticAlgorithm
from qdax.core.bi_map_elites import BiMAPElites, sampling_function
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.gp.encoding import compute_genome_to_step_fn
from qdax.core.gp.evaluation import gp_scoring_function_brax_envs
from qdax.core.gp.graph_utils import get_graph_descriptor_extractor
from qdax.core.gp.individual import compute_genome_mask, compute_mutation_mask, generate_population, \
    compute_mutation_fn, compute_variation_mutation_fn
from qdax.core.gp.utils import update_config
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax import environments

from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.environments.damages_wrappers import BrokenSensorsWrapper
from qdax.environments.kheperax.task import KheperaxConfig, KheperaxTargetedTask, KheperaxSpeedTargetedTask
from qdax.tasks.brax_envs import scoring_function_brax_envs
from qdax.types import RNGKey, Genotype, Fitness, ExtraScores

from qdax.utils.metrics import CSVLogger, default_biqd_metrics, default_qd_metrics, qd_metrics_with_bi_tracking, \
    default_ga_metrics

from wandb.sdk.wandb_run import Run


def validate_repertoire(
        config: Dict,
        repertoire: MapElitesRepertoire,
        random_key: RNGKey,
        sensors_breakage_id: int
) -> Tuple[Fitness, RNGKey]:
    validation_fitnesses, random_key = validate_genotypes(config, repertoire.genotypes, random_key, sensors_breakage_id)
    validation_fitnesses = jnp.where(repertoire.fitnesses > -jnp.inf, validation_fitnesses, -jnp.inf)
    return validation_fitnesses, random_key


def validate_genotypes(
        config: Dict,
        genotypes: Genotype,
        random_key: RNGKey,
        sensors_breakage_id: int
) -> Tuple[Fitness, RNGKey]:
    # Init environment
    if config["env_name"] == "kheperax":
        config_kheperax = KheperaxConfig.with_episode_length(config["episode_length"])
        env = KheperaxTargetedTask.create_environment(config_kheperax)
    elif config["env_name"] == "robotmaze":
        config_robotmaze = KheperaxConfig.with_episode_length(config["episode_length"])
        env = KheperaxSpeedTargetedTask.create_environment(config_robotmaze)
    else:
        env = environments.create(config["env_name"], episode_length=config["episode_length"], legacy_spring=False)

    sensors_breakage_mask = jnp.ones(env.observation_size)
    sensors_breakage_mask = sensors_breakage_mask.at[sensors_breakage_id].set(0)
    env = BrokenSensorsWrapper(env, sensors_breakage_mask)

    # Update config with env info
    config = update_config(config, env)

    # Create the initial environment states
    random_key, env_key = jax.random.split(random_key)
    env_keys = jnp.repeat(jnp.expand_dims(env_key, axis=0), repeats=len(genotypes), axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_env_states = reset_fn(env_keys)
    program_initial_states = jnp.zeros((len(genotypes), config["program_state_size"]))

    # Define scoring function
    encoding_fn = compute_genome_to_step_fn(env, config)
    scoring_fn = partial(
        gp_scoring_function_brax_envs,
        init_states=(init_env_states, program_initial_states),
        episode_length=config["episode_length"],
        encoding_fn=encoding_fn,
    )
    fitnesses, _, _, random_key = scoring_fn(
        genotypes, random_key
    )

    return fitnesses, random_key


def compute_validation_metrics(fitnesses: Fitness, validation_fitnesses: Fitness) -> Dict:
    fill_mask = fitnesses > -jnp.inf
    fitnesses = jnp.nan_to_num(fitnesses)
    # offset for the pointmaze
    if jnp.max(fitnesses) < 0:
        fitnesses = fitnesses + 100
        validation_fitnesses = validation_fitnesses + 100
    max_fitness = jnp.max(fitnesses)
    min_validation_fitness = jnp.nanmin(validation_fitnesses)
    validation_fitnesses = jnp.nan_to_num(validation_fitnesses, nan=min_validation_fitness)
    fitness_difference = fitnesses[fill_mask] - validation_fitnesses[fill_mask]
    relative_fitness_difference = fitness_difference / fitnesses[fill_mask]

    max_validation_fitness = jnp.max(validation_fitnesses)
    difference_of_maxs = max_fitness - max_validation_fitness
    difference_baseline = max_fitness - validation_fitnesses[jnp.argmax(fitnesses)]
    return {
        "average_difference": jnp.mean(fitness_difference),
        "min_difference": jnp.min(fitness_difference),
        "max_difference": jnp.max(fitness_difference),
        "average_relative_difference": jnp.mean(relative_fitness_difference),
        "min_relative_difference": jnp.min(relative_fitness_difference),
        "max_relative_difference": jnp.max(relative_fitness_difference),
        "max_validation_fitness": max_validation_fitness,
        "difference_of_maxs": difference_of_maxs,
        "relative_difference_of_maxs": difference_of_maxs / max_fitness,
        "relative_difference_baseline": difference_baseline / max_fitness
    }


def _log_occupation(loop: int, iteration: int, occupation_dict: Dict, occupation_logger: CSVLogger) -> None:
    occupation_metrics = {"loop": loop, "iteration": iteration}
    for centroid_id in range(len(occupation_dict)):
        occupation_metrics["centroid"] = centroid_id
        occupation_metrics["occupation"] = occupation_dict[centroid_id]
        occupation_metrics["fitness"] = occupation_dict[centroid_id]
        occupation_logger.log(occupation_metrics)


def run_ga(config: Dict, target_path: str = "../results") -> None:
    run_name = f"ga_{config['solver']}_{config['env_name']}_{config['seed']}"

    if os.path.exists(f"{target_path}/{run_name}"):
        print(f"{target_path}/{run_name}/ -> skipped")
        return

    api = wandb.Api(timeout=40)
    wb_run = wandb.init(
        config=config,
        project="cgpax",
        name=run_name
    )

    # Init environment
    if config["env_name"] == "kheperax":
        config_kheperax = KheperaxConfig.with_episode_length(config["episode_length"])
        env = KheperaxTargetedTask.create_environment(config_kheperax)
    elif config["env_name"] == "robotmaze":
        config_robotmaze = KheperaxConfig.with_episode_length(config["episode_length"])
        env = KheperaxSpeedTargetedTask.create_environment(config_robotmaze)
    else:
        env = environments.create(config["env_name"], episode_length=config["episode_length"], legacy_spring=False)

    # Update config with env info
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
            pop_size=config["parents_size"],
            genome_mask=genome_mask,
            rnd_key=pop_key,
            fixed_genome_trailing=fixed_outputs
        )
        config["p_mut_outputs"] = 0
    else:
        population = generate_population(pop_size=config["parents_size"], genome_mask=genome_mask, rnd_key=pop_key)

    # Create the initial environment states
    random_key, env_key = jax.random.split(random_key)
    env_keys = jnp.repeat(jnp.expand_dims(env_key, axis=0), repeats=config["parents_size"], axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_env_states = reset_fn(env_keys)
    program_initial_states = jnp.zeros((config["parents_size"], config["program_state_size"]))

    # Define scoring function
    encoding_fn = compute_genome_to_step_fn(env, config)
    qd_scoring_fn = partial(
        gp_scoring_function_brax_envs,
        init_states=(init_env_states, program_initial_states),
        episode_length=config["episode_length"],
        encoding_fn=encoding_fn,
    )

    def scoring_fn(genotypes: Genotype, rnd_key: RNGKey) -> Tuple[Fitness, ExtraScores, RNGKey]:
        fitnesses, _, extra_scores, rnd_key = qd_scoring_fn(genotypes, rnd_key)
        fitnesses = jnp.nan_to_num(fitnesses, nan=-jnp.inf)
        return fitnesses, extra_scores, rnd_key

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
        batch_size=config["parents_size"]
    )

    genetic_algorithm = GeneticAlgorithm(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=default_ga_metrics,
    )

    # Compute initial repertoire and emitter state
    repertoire, emitter_state, random_key = genetic_algorithm.init(population, config["pop_size"], random_key)

    # Launch GA iterations
    log_period = 10
    num_loops = int(config["n_iterations"] / log_period)

    headers = ["loop", "iteration", "max_fitness", "time", "current_time"]
    validation_headers = [
        "average_difference",
        "max_difference",
        "min_difference",
        "average_relative_difference",
        "min_relative_difference",
        "max_relative_difference",
        "max_validation_fitness",
        "difference_of_maxs",
        "relative_difference_of_maxs",
        "relative_difference_baseline",
        "repertoire_id",
        "sensor_id"
    ]

    csv_logger = CSVLogger(
        f"{target_path}/{wb_run.name}.csv",
        header=headers
    )
    all_metrics = {}
    validation_logger = CSVLogger(
        f"{target_path}/{wb_run.name}_validation.csv",
        header=validation_headers
    )

    # main loop
    ga_scan_update = genetic_algorithm.scan_update
    print(wb_run.name)
    for i in range(num_loops):
        start_time = time.time()
        # main iterations
        (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
            ga_scan_update,
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

        print(f"Loop: {i + 1}, "
              f"max_fitness: {logged_metrics['max_fitness']}, "
              f"time: {logged_metrics['time']}")

    repertoire_path = f"{target_path}/{wb_run.name}/"
    os.makedirs(repertoire_path, exist_ok=True)
    repertoire.save(path=repertoire_path)

    for sensor_id in range(config["n_in_env"]):
        validation_fitnesses, random_key = validate_repertoire(config, repertoire, random_key, sensor_id)
        jnp.save(f"{repertoire_path}/validation_fitnesses_{sensor_id}.npy", validation_fitnesses)

        metrics = compute_validation_metrics(repertoire.fitnesses, validation_fitnesses)
        metrics["repertoire_id"] = "main"
        metrics["sensor_id"] = sensor_id
        validation_logger.log(metrics)

    wb_run.finish()


def run_qd(config: Dict, target_path: str = "../results") -> None:
    bi_map_elites = config.get("bi_map", False)

    run_name = f"bimapelites_{config['sampler']}" if bi_map_elites else "mapelites"
    run_name += f"_{config['solver']}_{config['env_name']}_{config['seed']}"

    if os.path.exists(f"{target_path}/{run_name}"):
        print(f"{target_path}/{run_name}/ -> skipped")
        return

    api = wandb.Api(timeout=40)
    wb_run = wandb.init(
        config=config,
        project="cgpax",
        name=run_name
    )

    # Init environment
    if config["env_name"] == "kheperax":
        config_kheperax = KheperaxConfig.with_episode_length(config["episode_length"])
        env = KheperaxTargetedTask.create_environment(config_kheperax)
    elif config["env_name"] == "robotmaze":
        config_robotmaze = KheperaxConfig.with_episode_length(config["episode_length"])
        env = KheperaxSpeedTargetedTask.create_environment(config_robotmaze)
    else:
        env = environments.create(config["env_name"], episode_length=config["episode_length"], legacy_spring=False)

    bd_extraction_fn = environments.behavior_descriptor_extractor[config["env_name"]]
    n_behavior_descriptors = 2 if config["env_name"] in ["kheperax", "robotmaze"] else env.behavior_descriptor_length

    # Update config with env info
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
    qd_offset = reward_offset * config["episode_length"] if config["env_name"] not in ["kheperax", "robotmaze"] else 1.5

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

            graph_centroids = jnp.load(f"{target_path}/graph_centroids.npy")
            behavior_centroids = jnp.load(f"{target_path}/behavior_centroids_pointmaze.npy") \
                if config["env_name"] == "pointmaze" \
                else jnp.load(f"{target_path}/behavior_centroids_{n_behavior_descriptors}d.npy")

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
        map_elites: BiMAPElites,
        target_path: str,
        wb_run: Run
) -> None:
    graph_centroids = jnp.load(f"{target_path}/graph_centroids.npy")
    behavior_centroids = jnp.load(f"{target_path}/behavior_centroids_pointmaze.npy") \
        if config["env_name"] == "pointmaze" \
        else jnp.load(f"{target_path}/behavior_centroids_{config['n_behavior_descriptors']}d.npy")

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

    validation_headers = [
        "average_difference",
        "max_difference",
        "min_difference",
        "average_relative_difference",
        "min_relative_difference",
        "max_relative_difference",
        "max_validation_fitness",
        "difference_of_maxs",
        "relative_difference_of_maxs",
        "relative_difference_baseline",
        "repertoire_id",
        "sensor_id"
    ]

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
    validation_logger = CSVLogger(
        f"{target_path}/{wb_run.name}_validation.csv",
        header=validation_headers
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

    for sensor_id in range(config["n_in_env"]):
        r1_validation_fitnesses, random_key = validate_repertoire(config, repertoire.repertoire1, random_key, sensor_id)
        r2_validation_fitnesses, random_key = validate_repertoire(config, repertoire.repertoire2, random_key, sensor_id)
        jnp.save(f"{repertoire_path}/r1_validation_fitnesses_{sensor_id}.npy", r1_validation_fitnesses)
        jnp.save(f"{repertoire_path}/r2_validation_fitnesses_{sensor_id}.npy", r2_validation_fitnesses)

        metrics1 = compute_validation_metrics(repertoire.repertoire1.fitnesses, r1_validation_fitnesses)
        metrics1["repertoire_id"] = 1
        metrics1["sensor_id"] = sensor_id
        validation_logger.log(metrics1)

        metrics2 = compute_validation_metrics(repertoire.repertoire1.fitnesses, r1_validation_fitnesses)
        metrics2["repertoire_id"] = 2
        metrics2["sensor_id"] = sensor_id
        validation_logger.log(metrics2)

    wb_run.finish()


def mapelites_run(
        config: Dict,
        random_key: RNGKey,
        init_population: Genotype,
        map_elites: MAPElites,
        target_path: str,
        wb_run: Run
) -> None:
    centroids = jnp.load(f"{target_path}/me_centroids_pointmaze.npy") \
        if config["env_name"] == "pointmaze" \
        else jnp.load(f"{target_path}/me_centroids_{config['n_descriptors']}d.npy")

    # Compute initial repertoire and emitter state
    repertoire, emitter_state, random_key = map_elites.init(init_population, centroids, random_key)

    # Launch MAP-Elites iterations
    log_period = 10
    num_loops = int(config["n_iterations"] / log_period)

    headers = ["loop", "iteration", "qd_score", "max_fitness", "coverage", "time", "current_time"]

    if config.get("track_separate", True):
        headers.extend(["coverage1", "coverage2"])

    validation_headers = [
        "average_difference",
        "max_difference",
        "min_difference",
        "average_relative_difference",
        "min_relative_difference",
        "max_relative_difference",
        "max_validation_fitness",
        "difference_of_maxs",
        "relative_difference_of_maxs",
        "relative_difference_baseline",
        "repertoire_id",
        "sensor_id"
    ]

    csv_logger = CSVLogger(
        f"{target_path}/{wb_run.name}.csv",
        header=headers
    )
    all_metrics = {}
    occupation_logger = CSVLogger(
        f"{target_path}/{wb_run.name}_occupation.csv",
        header=["loop", "iteration", "centroid", "occupation", "fitness"]
    )
    validation_logger = CSVLogger(
        f"{target_path}/{wb_run.name}_validation.csv",
        header=validation_headers
    )

    # main loop
    map_elites_scan_update = map_elites.scan_update
    print(wb_run.name)
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

    for sensor_id in range(config["n_in_env"]):
        validation_fitnesses, random_key = validate_repertoire(config, repertoire, random_key, sensor_id)
        jnp.save(f"{repertoire_path}/validation_fitnesses_{sensor_id}.npy", validation_fitnesses)

        metrics = compute_validation_metrics(repertoire.fitnesses, validation_fitnesses)
        metrics["repertoire_id"] = "main"
        metrics["sensor_id"] = sensor_id
        validation_logger.log(metrics)

    wb_run.finish()


def validate_nn_repertoire(
        config: Dict,
        repertoire: MapElitesRepertoire,
        random_key: RNGKey,
        sensors_breakage_id: int
) -> Tuple[Fitness, RNGKey]:
    validation_fitnesses, random_key = validate_nn_genotypes(config, repertoire.genotypes, random_key,
                                                             sensors_breakage_id)
    validation_fitnesses = jnp.where(repertoire.fitnesses > -jnp.inf, validation_fitnesses, -jnp.inf)
    return validation_fitnesses, random_key


def validate_nn_genotypes(
        config: Dict,
        genotypes: Genotype,
        random_key: RNGKey,
        sensors_breakage_id: int
) -> Tuple[Fitness, RNGKey]:
    # Init environment
    if config["env_name"] == "kheperax":
        config_kheperax = KheperaxConfig.with_episode_length(config["episode_length"])
        env = KheperaxTargetedTask.create_environment(config_kheperax)
    elif config["env_name"] == "robotmaze":
        config_robotmaze = KheperaxConfig.with_episode_length(config["episode_length"])
        env = KheperaxSpeedTargetedTask.create_environment(config_robotmaze)
    else:
        env = environments.create(config["env_name"], episode_length=config["episode_length"], legacy_spring=False)

    sensors_breakage_mask = jnp.ones(env.observation_size)
    sensors_breakage_mask = sensors_breakage_mask.at[sensors_breakage_id].set(0)
    env = BrokenSensorsWrapper(env, sensors_breakage_mask)

    # Init policy network
    policy_layer_sizes = config["policy_hidden_layer_sizes"] + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # TODO fix this
    n_centroids = 1024

    # Create the initial environment states
    random_key, subkey = jax.random.split(random_key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=n_centroids, axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_states = reset_fn(keys)

    # Define scoring function
    def play_step_fn(
            env_state,
            policy_params,
            random_key,
    ):
        actions = policy_network.apply(policy_params, env_state.obs)

        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )

        return next_state, policy_params, random_key, transition

    # Prepare the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[config["env_name"]]
    scoring_fn = functools.partial(
        scoring_function_brax_envs,
        init_states=init_states,
        episode_length=config["episode_length"],
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    fitnesses, _, _, random_key = scoring_fn(
        genotypes, random_key
    )

    return fitnesses, random_key


def run_ne(config: Dict, target_path: str = "../results") -> None:
    run_name = f"ne_{config['env_name']}_{config['seed']}"

    if os.path.exists(f"{target_path}/{run_name}"):
        print(f"{target_path}/{run_name}/ -> skipped")
        return

    api = wandb.Api(timeout=40)
    wb_run = wandb.init(
        config=config,
        project="cgpax",
        name=run_name
    )

    # Init environment
    if config["env_name"] == "kheperax":
        config_kheperax = KheperaxConfig.with_episode_length(config["episode_length"])
        env = KheperaxTargetedTask.create_environment(config_kheperax)
    elif config["env_name"] == "robotmaze":
        config_robotmaze = KheperaxConfig.with_episode_length(config["episode_length"])
        env = KheperaxSpeedTargetedTask.create_environment(config_robotmaze)
    else:
        env = environments.create(config["env_name"], episode_length=config["episode_length"], legacy_spring=False)

    # Update config with env info
    # config = update_config(config, env)

    # Init a random key
    random_key = jax.random.PRNGKey(config["seed"])

    # Init policy network
    policy_layer_sizes = config["policy_hidden_layer_sizes"] + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=config["pop_size"])
    fake_batch = jnp.zeros(shape=(config["pop_size"], env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    # Create the initial environment states
    random_key, subkey = jax.random.split(random_key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=config["pop_size"], axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_states = reset_fn(keys)

    # Define scoring function
    def play_step_fn(
            env_state,
            policy_params,
            random_key,
    ):
        actions = policy_network.apply(policy_params, env_state.obs)

        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )

        return next_state, policy_params, random_key, transition

    # Prepare the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[config["env_name"]]
    n_behavior_descriptors = 2 if config["env_name"] == "kheperax" or config["env_name"] == "robotmaze" \
        else env.behavior_descriptor_length
    scoring_fn = functools.partial(
        scoring_function_brax_envs,
        init_states=init_states,
        episode_length=config["episode_length"],
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[config["env_name"]]

    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * config["episode_length"],
    )

    # Define emitter
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=config["iso_sigma"], line_sigma=config["line_sigma"]
    )
    mixing_emitter = MixingEmitter(
        mutation_fn=None,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=config["pop_size"]
    )

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
    )

    # Compute the centroids
    behavior_centroids = jnp.load(f"{target_path}/behavior_centroids_pointmaze.npy") \
        if config["env_name"] == "pointmaze" \
        else jnp.load(f"{target_path}/behavior_centroids_{n_behavior_descriptors}d.npy")

    # Compute initial repertoire and emitter state
    repertoire, emitter_state, random_key = map_elites.init(init_variables, behavior_centroids, random_key)

    # Launch MAP-Elites iterations
    log_period = 10
    num_loops = int(config["n_iterations"] / log_period)

    headers = ["loop", "iteration", "qd_score", "max_fitness", "coverage", "time", "current_time"]

    validation_headers = [
        "average_difference",
        "max_difference",
        "min_difference",
        "average_relative_difference",
        "min_relative_difference",
        "max_relative_difference",
        "max_validation_fitness",
        "difference_of_maxs",
        "relative_difference_of_maxs",
        "relative_difference_baseline",
        "repertoire_id",
        "sensor_id"
    ]

    csv_logger = CSVLogger(
        f"{target_path}/{wb_run.name}.csv",
        header=headers
    )
    all_metrics = {}
    occupation_logger = CSVLogger(
        f"{target_path}/{wb_run.name}_occupation.csv",
        header=["loop", "iteration", "centroid", "occupation", "fitness"]
    )
    validation_logger = CSVLogger(
        f"{target_path}/{wb_run.name}_validation.csv",
        header=validation_headers
    )

    # main loop
    map_elites_scan_update = map_elites.scan_update
    print(wb_run.name)
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

    for sensor_id in range(env.observation_size):
        validation_fitnesses, random_key = validate_nn_repertoire(config, repertoire, random_key, sensor_id)
        jnp.save(f"{repertoire_path}/validation_fitnesses_{sensor_id}.npy", validation_fitnesses)

        metrics = compute_validation_metrics(repertoire.fitnesses, validation_fitnesses)
        metrics["repertoire_id"] = "main"
        metrics["sensor_id"] = sensor_id
        validation_logger.log(metrics)

    wb_run.finish()


if __name__ == '__main__':
    path = "../results"
    if len(sys.argv) > 1:
        path = sys.argv[1]

    print(f"Target path: {path}\n")
    n_itx = 10000

    seeds = range(0, 30)
    run_types = ["both", "s1", "s2", "me", "ga", "ne"]

    custom_seed = os.environ.get("SEED")
    if custom_seed is not None:
        seeds = [int(custom_seed)]
        print(f"Custom seed: {custom_seed}")

    custom_sampling = os.environ.get("SAMPLING")
    if custom_sampling is not None:
        run_types = [custom_sampling]
        print(f"Custom sampling: {custom_sampling}")

    env_duration_tuples = [
        ("pointmaze", 100),
        ("robotmaze", 1000),
        ("hopper_uni", 1000),
        ("walker2d_uni", 1000),
    ]

    print(f"Total runs: {len(seeds) * len(env_duration_tuples) * len(run_types)}")
    for env_name, episode_length in env_duration_tuples:
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
                "graph_descriptors": ["function_arities"],
                "custom_archive_shape": True,
                "fixed_outputs": True
            }

            for run_type in run_types:
                current_cfg = copy.deepcopy(env_cfg)
                if run_type == "ne":
                    current_cfg["policy_hidden_layer_sizes"] = (64, 64)
                    current_cfg["iso_sigma"] = 0.005
                    current_cfg["line_sigma"] = 0.05
                    run_ne(env_cfg, target_path=path)
                if run_type == "ga":
                    current_cfg["parents_size"] = 90
                    run_ga(current_cfg, target_path=path)
                else:
                    if run_type == "me":
                        # standard map elites
                        current_cfg["bi_map"] = False
                        current_cfg["track_separate"] = True

                    else:
                        # map elites with double archive
                        current_cfg["sampler"] = run_type
                        current_cfg["bi_map"] = True
                    run_qd(current_cfg, target_path=path)
