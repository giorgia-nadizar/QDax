from functools import partial
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from brax.envs import Env, State as EnvState
from jax import jit, lax, vmap

from qdax.core.neuroevolution.buffers.buffer import Transition, QDTransition
from qdax.types import (
    Descriptor,
    EnvState,
    ExtraScores,
    Fitness,
    Genotype,
    RNGKey,
    Program,
    ProgramState
)


@partial(
    jax.jit,
    static_argnames=(
        "episode_length",
        "encoding_fn",
        "behavior_descriptor_extractor",
        "graph_descriptor_extractor"
    ),
)
def gp_scoring_function_brax_envs(
    genotypes: Genotype,
    random_key: RNGKey,
    init_states: Tuple[EnvState, ProgramState],
    episode_length: int,
    encoding_fn: Callable[
        [jnp.ndarray],
        Callable[[EnvState, ProgramState, RNGKey], Tuple[EnvState, ProgramState, RNGKey, Transition]]
    ],
    behavior_descriptor_extractor: Callable[
        [QDTransition, jnp.ndarray],
        Descriptor
    ] = vmap(lambda x, y: jnp.empty((0,))),
    graph_descriptor_extractor: Callable[
        [Genotype],
        Descriptor
    ] = vmap(lambda x: jnp.empty((0,)))
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    def _generate_unroll(
        init_state: Tuple[EnvState, ProgramState],
        genotype: Genotype,
        random_key: RNGKey
    ) -> Tuple[EnvState, QDTransition]:
        program_step_fn = encoding_fn(genotype)
        init_env_state, init_program_state = init_state

        def _scan_play_step_fn(
            carry: Tuple[EnvState, ProgramState, RNGKey], unused_arg: Any
        ) -> Tuple[Tuple[EnvState, ProgramState, RNGKey], Transition]:
            env_s, program_s, rnd_key, transition = program_step_fn(*carry)
            return (env_s, program_s, rnd_key), transition

        (env_state, program_state, _), transitions = jax.lax.scan(
            f=_scan_play_step_fn,
            init=(init_env_state, init_program_state, random_key),
            xs=(),
            length=episode_length
        )

        return env_state, transitions

    random_key, subkey = jax.random.split(random_key)
    unroll_fn = partial(_generate_unroll, random_key=subkey)
    _, data = jax.vmap(unroll_fn)(init_states, genotypes)

    # create a mask to extract data properly
    is_done = jnp.clip(jnp.cumsum(data.dones, axis=1), 0, 1)
    mask = jnp.roll(is_done, 1, axis=1)
    mask = mask.at[:, 0].set(0)

    # scores
    fitnesses = jnp.sum(data.rewards * (1.0 - mask), axis=1)
    env_descriptors = behavior_descriptor_extractor(data, mask)
    graph_descriptors = graph_descriptor_extractor(genotypes)
    descriptors = jnp.concatenate([env_descriptors, graph_descriptors], axis=1)

    return (
        fitnesses,
        descriptors,
        {
            "transitions": data,
        },
        random_key,
    )


@partial(
    jax.jit,
    static_argnames=(
        "episode_length",
        "encoding_fn",
        "behavior_descriptor_extractor"
    ),
)
def gp_scoring_function_brax_envs_rewards_descriptors(
    genotypes: Genotype,
    random_key: RNGKey,
    init_states: Tuple[EnvState, ProgramState],
    episode_length: int,
    encoding_fn: Callable[
        [jnp.ndarray],
        Callable[[EnvState, ProgramState, RNGKey], Tuple[EnvState, ProgramState, RNGKey, Transition]]
    ],
    behavior_descriptor_extractor: Callable[
        [QDTransition, jnp.ndarray],
        Descriptor
    ] = vmap(lambda x, y: jnp.empty((0,)))
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    def _generate_unroll(
        init_state: Tuple[EnvState, ProgramState],
        genotype: Genotype,
        random_key: RNGKey
    ) -> Tuple[EnvState, QDTransition]:
        program_step_fn = encoding_fn(genotype)
        init_env_state, init_program_state = init_state

        def _scan_play_step_fn(
            carry: Tuple[EnvState, ProgramState, RNGKey], unused_arg: Any
        ) -> Tuple[Tuple[EnvState, ProgramState, RNGKey], Transition]:
            env_s, program_s, rnd_key, transition = program_step_fn(*carry)
            return (env_s, program_s, rnd_key), transition

        (env_state, program_state, _), transitions = jax.lax.scan(
            f=_scan_play_step_fn,
            init=(init_env_state, init_program_state, random_key),
            xs=(),
            length=episode_length
        )

        return env_state, transitions

    random_key, subkey = jax.random.split(random_key)
    unroll_fn = partial(_generate_unroll, random_key=subkey)
    _, data = jax.vmap(unroll_fn)(init_states, genotypes)

    # create a mask to extract data properly
    is_done = jnp.clip(jnp.cumsum(data.dones, axis=1), 0, 1)
    mask = jnp.roll(is_done, 1, axis=1)
    mask = mask.at[:, 0].set(0)

    # scores
    fitnesses = jnp.sum(data.rewards * (1.0 - mask), axis=1)
    env_descriptors = behavior_descriptor_extractor(data, mask)
    health_rewards = jnp.expand_dims(jnp.sum(data.health_rewards * (1.0 - mask), axis=1), axis=1)
    run_rewards = jnp.expand_dims(jnp.sum(data.run_rewards * (1.0 - mask), axis=1), axis=1)

    descriptors = jnp.concatenate([env_descriptors, health_rewards, run_rewards], axis=1)

    return (
        fitnesses,
        descriptors,
        {
            "transitions": data,
        },
        random_key,
    )


def evaluate_program(
    program: Program,
    initial_program_state: ProgramState,
    rnd_key: RNGKey,
    env: Env,
    episode_length: int = 1000
) -> Tuple[float, Tuple[float, float, float]]:
    initial_env_state = jit(env.reset)(rnd_key)
    initial_rewards_state = (0.0, 0.0, 0.0)

    def _rollout_loop(
        carry: Tuple[EnvState, ProgramState, Tuple[float, float, float], float, int],
        unused_arg: Any
    ) -> Tuple[Tuple[EnvState, ProgramState, Tuple[float, float, float], float, int], Any]:
        env_state, program_state, rewards_state, cum_rew, active_episode = carry
        inputs = env_state.obs
        new_program_state, actions = program(inputs, program_state)
        new_state = jit(env.step)(env_state, actions)
        rew_forward, rew_ctrl, rew_healthy = rewards_state
        rew_forward += new_state.metrics.get("reward_forward", new_state.metrics.get("reward_run", 0)) * active_episode
        rew_ctrl += new_state.metrics.get("reward_ctrl", 0) * active_episode
        rew_healthy += new_state.metrics.get("reward_healthy",
                                             new_state.metrics.get("reward_survive", 0)) * active_episode
        new_rewards_state = rew_forward, rew_ctrl, rew_healthy
        corrected_reward = new_state.reward * active_episode
        new_active_episode = (active_episode * (1 - new_state.done)).astype(int)
        new_carry = new_state, new_program_state, new_rewards_state, cum_rew + corrected_reward, new_active_episode
        return new_carry, corrected_reward

    (final_env_state, _, final_rewards, cum_reward, _), _ = lax.scan(
        f=_rollout_loop,
        init=(initial_env_state, initial_program_state, initial_rewards_state, initial_env_state.reward, 1),
        xs=(),
        length=episode_length,
    )
    return cum_reward, final_rewards


def evaluate_genome(
    genome: jnp.ndarray,
    rnd_key: RNGKey,
    initial_program_state: ProgramState,
    env: Env,
    encoding_function: Callable[[jnp.ndarray], Program],
    episode_length: int = 1000,
    inner_evaluator: Callable[
        [Program, ProgramState, RNGKey, Env, int],
        Tuple[float, Tuple[float, float, float]]
    ] = evaluate_program
) -> Tuple[float, Tuple[float, float, float]]:
    return inner_evaluator(
        encoding_function(genome),
        initial_program_state,
        rnd_key,
        env,
        episode_length
    )


def evaluate_genome_n_times(
    genome: jnp.ndarray,
    rnd_key: RNGKey,
    initial_program_state: ProgramState,
    env: Env, n_times: int,
    encoding_function: Callable[[jnp.ndarray], Program],
    episode_length: int = 1000,
    inner_evaluator: Callable = evaluate_program
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    rnd_key, *sub_keys = jax.random.split(rnd_key, n_times + 1)
    partial_evaluate_genome = partial(evaluate_genome,
                                      initial_program_state=initial_program_state,
                                      env=env,
                                      episode_length=episode_length,
                                      encoding_function=encoding_function,
                                      inner_evaluator=inner_evaluator
                                      )
    vmap_evaluate_genome = vmap(partial_evaluate_genome, in_axes=(None, 0))
    return vmap_evaluate_genome(genome, jnp.array(sub_keys))
