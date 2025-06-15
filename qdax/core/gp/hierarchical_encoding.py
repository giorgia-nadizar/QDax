from typing import Callable, Tuple, Dict, Mapping

import jax.numpy as jnp
from brax.envs import Env

from qdax.core.gp.encoding import compute_encoding_function
from qdax.core.neuroevolution.buffers.buffer import Transition, QDTransitionDetailed
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.types import ProgramState, EnvState, RNGKey


def compute_genome_to_hierarchical_step_fn(
        environment: Env,
        config: Dict,
        outputs_wrapper: Callable[[jnp.ndarray], jnp.ndarray] = jnp.tanh
) -> Callable[
    [jnp.ndarray],
    Callable[[EnvState, ProgramState, MLP, Callable[[jnp.ndarray], Mapping], RNGKey],
    Tuple[EnvState, ProgramState, RNGKey, Transition]]
]:
    encoding_fn = compute_encoding_function(config, outputs_wrapper)

    def _genome_to_program_step_fn(
            genome: jnp.ndarray
    ) -> Callable[[EnvState, ProgramState, MLP, Callable[[jnp.ndarray], Mapping], RNGKey], Tuple[
        EnvState, ProgramState, RNGKey, Transition]]:
        program = encoding_fn(genome)

        def _program_hierarchical_step_fn(
                env_state: EnvState,
                program_state: ProgramState,
                policy_network: MLP,
                output_params_mapping_fn: Callable[[jnp.ndarray], Mapping],
                random_key: RNGKey
        ) -> Tuple[EnvState, ProgramState, RNGKey, Transition]:
            observation = env_state.obs
            xy_pos, robot_state = jnp.split(observation, [2])
            next_program_state, program_output = program(xy_pos, program_state)
            policy_params = output_params_mapping_fn(program_output)
            actions = policy_network.apply(policy_params, robot_state)
            next_state = environment.step(env_state, actions)

            health_reward = next_state.metrics.get("reward_healthy", next_state.metrics.get("reward_survive", 0))
            run_reward = next_state.metrics.get("reward_forward", next_state.metrics.get("reward_run", 0))

            transition = QDTransitionDetailed(
                obs=env_state.obs,
                next_obs=next_state.obs,
                rewards=next_state.reward,
                health_rewards=health_reward,
                run_rewards=run_reward,
                dones=next_state.done,
                actions=actions,
                truncations=next_state.info["truncation"],
                state_desc=env_state.info["state_descriptor"],
                next_state_desc=next_state.info["state_descriptor"],
            )
            return next_state, next_program_state, random_key, transition

        return _program_hierarchical_step_fn

    return _genome_to_program_step_fn
