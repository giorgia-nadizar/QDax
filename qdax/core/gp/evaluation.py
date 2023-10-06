from functools import partial
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from brax.envs import Env, State as EnvState
from jax import jit, lax, vmap

from qdax.types import EnvState, RNGKey, ProgramState, Program


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
