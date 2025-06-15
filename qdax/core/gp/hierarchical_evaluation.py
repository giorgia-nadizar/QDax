from functools import partial
from typing import Any, Callable, Tuple, Mapping

import jax
import jax.numpy as jnp
from brax.envs import State as EnvState

from qdax.core.neuroevolution.buffers.buffer import Transition, QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.environments import get_final_xy_position
from qdax.types import (
    Descriptor,
    EnvState,
    ExtraScores,
    Fitness,
    Genotype,
    RNGKey,
    ProgramState
)


@partial(
    jax.jit,
    static_argnames=(
            "episode_length",
            "encoding_fn",
            "policy_network",
            "output_params_mapping_fn"
    ),
)
def gp_hierarchical_scoring_function_navigation_envs(
        genotypes: Genotype,
        random_key: RNGKey,
        init_states: Tuple[EnvState, ProgramState],
        episode_length: int,
        encoding_fn: Callable[
            [jnp.ndarray],
            Callable[[EnvState, ProgramState, MLP, Callable[[jnp.ndarray], Mapping], RNGKey], Tuple[
                EnvState, ProgramState, RNGKey, Transition]]],
        policy_network: MLP,
        output_params_mapping_fn: Callable[[jnp.ndarray], Mapping],
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    def _generate_unroll(
            init_state: Tuple[EnvState, ProgramState],
            genotype: Genotype,
            random_key: RNGKey
    ) -> Tuple[EnvState, QDTransition]:
        hierarchical_program_step_fn = encoding_fn(genotype)
        init_env_state, init_program_state = init_state

        def _scan_play_step_fn(
                carry: Tuple[EnvState, ProgramState, RNGKey], unused_arg: Any
        ) -> Tuple[Tuple[EnvState, ProgramState, RNGKey], Transition]:
            e_state, p_state, r_key = carry
            env_s, program_s, rnd_key, transition = hierarchical_program_step_fn(
                e_state, p_state, policy_network, output_params_mapping_fn, r_key
            )
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
    env_descriptors = get_final_xy_position(data, mask)

    return (
        fitnesses,
        env_descriptors,
        {
            "transitions": data,
        },
        random_key,
    )
