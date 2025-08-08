import functools

import jax
import pytest

import qdax.tasks.brax.v1 as environments
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.graphs.cartesian_genetic_programming import CGP, cgp_mutation
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
import jax.numpy as jnp
from qdax.tasks.brax.v1.env_creators import scoring_function_brax_envs as scoring_function
from qdax.utils.metrics import default_qd_metrics


def test_cgp_with_me() -> None:
    """Test that CGP can be used with ME and is jit safe.
        """


    batch_size = 10
    env_name = 'walker2d_uni'
    episode_length = 100
    num_iterations = 20
    seed = 42
    num_init_cvt_samples = 5_000
    num_centroids = 1024
    min_descriptor = 0.
    max_descriptor = 1.0

    # Init environment
    env = environments.create(env_name, episode_length=episode_length)
    reset_fn = jax.jit(env.reset)

    # Init a random key
    key = jax.random.key(seed)

    # Init the CGP policy graph with default values
    policy_graph = CGP(
        n_inputs=env.observation_size,
        n_outputs=env.action_size,
    )

    # Init the population of CGP genomes
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=batch_size)
    init_cgp_genomes = jax.vmap(policy_graph.init)(keys)


    # Define the play step fn for CGP to interact with the env
    def cgp_play_step_fn(
            env_state,
            policy_params,
            key,
    ):
        """
        Play an environment step and return the updated state and the transition.
        """

        actions = policy_graph.apply(policy_params, env_state.obs)

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

        return next_state, policy_params, key, transition

    # Prepare the scoring function
    descriptor_extraction_fn = environments.descriptor_extractor[env_name]
    scoring_fn_cgp = functools.partial(
        scoring_function,
        episode_length=episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=cgp_play_step_fn,
        descriptor_extractor=descriptor_extraction_fn,
    )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[env_name]

    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * episode_length,
    )

    # Define emitter
    cgp_variation_fn = functools.partial(
        cgp_mutation, cgp=policy_graph  # , mutation_probabilities={"inputs" : .2}
    )
    mixing_emitter = MixingEmitter(
        mutation_fn=cgp_variation_fn,
        variation_fn=None,
        variation_percentage=0.0,   # note: CGP works with mutation only
        batch_size=batch_size
    )

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn_cgp,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
    )

    # Compute the centroids
    key, subkey = jax.random.split(key)
    centroids = compute_cvt_centroids(
        num_descriptors=env.descriptor_length,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=min_descriptor,
        maxval=max_descriptor,
        key=subkey,
    )

    # Compute initial repertoire and emitter state
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = map_elites.init(init_cgp_genomes, centroids, subkey)

    # Check repertoire is not empty
    pytest.assume(jnp.any(repertoire.fitnesses > -jnp.inf))

    # Initial elements in repertoire
    n_initial_individuals = jnp.sum(repertoire.fitnesses > -jnp.inf)

    log_period = 3
    num_loops = num_iterations // log_period

    # Main loop
    map_elites_scan_update = map_elites.scan_update
    for i in range(num_loops):
        (
            repertoire,
            emitter_state,
            key,
        ), current_metrics = jax.lax.scan(
            map_elites_scan_update,
            (repertoire, emitter_state, key),
            (),
            length=log_period,
        )

    # Initial elements in repertoire
    n_final_individuals = jnp.sum(repertoire.fitnesses > -jnp.inf)

    # Check coverage did not decrease
    pytest.assume(n_final_individuals >= n_initial_individuals)


def test_cgp_with_me_ask_tell() -> None:
    """Test that CGP can be used with ME in its ask-tell way and is jit safe.
        """


    batch_size = 10
    env_name = 'walker2d_uni'
    episode_length = 100
    num_iterations = 20
    seed = 42
    num_init_cvt_samples = 5_000
    num_centroids = 1024
    min_descriptor = 0.
    max_descriptor = 1.0

    # Init environment
    env = environments.create(env_name, episode_length=episode_length)
    reset_fn = jax.jit(env.reset)

    # Init a random key
    key = jax.random.key(seed)

    # Init the CGP policy graph with default values
    policy_graph = CGP(
        n_inputs=env.observation_size,
        n_outputs=env.action_size,
    )

    # Init the population of CGP genomes
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=batch_size)
    init_cgp_genomes = jax.vmap(policy_graph.init)(keys)


    # Define the play step fn for CGP to interact with the env
    def cgp_play_step_fn(
            env_state,
            policy_params,
            key,
    ):
        """
        Play an environment step and return the updated state and the transition.
        """

        actions = policy_graph.apply(policy_params, env_state.obs)

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

        return next_state, policy_params, key, transition

    # Prepare the scoring function
    descriptor_extraction_fn = environments.descriptor_extractor[env_name]
    scoring_fn_cgp = functools.partial(
        scoring_function,
        episode_length=episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=cgp_play_step_fn,
        descriptor_extractor=descriptor_extraction_fn,
    )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[env_name]

    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * episode_length,
    )

    # Define emitter
    cgp_variation_fn = functools.partial(
        cgp_mutation, cgp=policy_graph  # , mutation_probabilities={"inputs" : .2}
    )
    mixing_emitter = MixingEmitter(
        mutation_fn=cgp_variation_fn,
        variation_fn=None,
        variation_percentage=0.0,   # note: CGP works with mutation only
        batch_size=batch_size
    )

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn_cgp,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
    )

    # Compute the centroids
    key, subkey = jax.random.split(key)
    centroids = compute_cvt_centroids(
        num_descriptors=env.descriptor_length,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=min_descriptor,
        maxval=max_descriptor,
        key=subkey,
    )

    # Evaluate the initial population
    key, subkey = jax.random.split(key)
    fitnesses, descriptors, extra_scores = scoring_fn_cgp(init_cgp_genomes, subkey)

    # Compute initial repertoire and emitter state
    repertoire, emitter_state, metrics = map_elites.init_ask_tell(
        genotypes=init_cgp_genomes,
        fitnesses=fitnesses,
        descriptors=descriptors,
        centroids=centroids,
        key=key,
        extra_scores=extra_scores,
    )

    # Check repertoire is not empty
    pytest.assume(jnp.any(repertoire.fitnesses > -jnp.inf))

    # Initial elements in repertoire
    n_initial_individuals = jnp.sum(repertoire.fitnesses > -jnp.inf)

    ask_fn = jax.jit(map_elites.ask)
    tell_fn = jax.jit(map_elites.tell)

    for i in range(num_iterations):
        key, subkey = jax.random.split(key)
        # Generate solutions
        genotypes, extra_info = ask_fn(repertoire, emitter_state, subkey)

        # Evaluate solutions: get fitness, descriptor and extra scores.
        # This is where custom evaluations on CPU or GPU can be added.
        key, subkey = jax.random.split(key)
        fitnesses, descriptors, extra_scores = scoring_fn_cgp(genotypes, subkey)

        # Update MAP-Elites
        repertoire, emitter_state, current_metrics = tell_fn(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            repertoire=repertoire,
            emitter_state=emitter_state,
            extra_scores=extra_scores,
            extra_info=extra_info,
        )

    # Initial elements in repertoire
    n_final_individuals = jnp.sum(repertoire.fitnesses > -jnp.inf)

    # Check coverage did not decrease
    pytest.assume(n_final_individuals >= n_initial_individuals)
