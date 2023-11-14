from functools import partial
from typing import Callable, Tuple, Dict

import jax.numpy as jnp
from brax.envs import Env
from jax import jit
from jax.lax import fori_loop

from qdax.core.gp.functions import function_switch, constants
from qdax.core.neuroevolution.buffers.buffer import Transition, QDTransitionDetailed
from qdax.environments.symmetric_envs import get_symmetric_input_indexes, get_action_joiner_func
from qdax.types import ProgramState, EnvState, RNGKey


def compute_encoding_function(
    config: Dict,
    outputs_wrapper: Callable[[jnp.ndarray], jnp.ndarray] = jnp.tanh
) -> Callable[
    [jnp.ndarray],
    Callable[[jnp.ndarray, ProgramState], Tuple[ProgramState, jnp.ndarray]]
]:
    if config["solver"] == "cgp":
        return partial(_genome_to_cgp_program, config=config, outputs_wrapper=outputs_wrapper)
    if config["solver"] == "lgp":
        return partial(_genome_to_lgp_program, config=config, outputs_wrapper=outputs_wrapper)
    raise ValueError("Solver must be either cgp or lgp.")


def compute_genome_to_symmetric_step_fn(
    environment: Env,
    config: Dict,
    outputs_wrapper: Callable[[jnp.ndarray], jnp.ndarray] = jnp.tanh
) -> Callable[
    [jnp.ndarray],
    Callable[[EnvState, ProgramState, RNGKey], Tuple[EnvState, ProgramState, RNGKey, Transition]]
]:
    encoding_fn = compute_encoding_function(config, outputs_wrapper)
    symmetric_input_indexes = get_symmetric_input_indexes(config["env_name"])
    action_joiner_function = get_action_joiner_func(config["env_name"])

    def _genome_to_program_step_fn(
        genome: jnp.ndarray
    ) -> Callable[[EnvState, ProgramState, RNGKey], Tuple[EnvState, ProgramState, RNGKey, Transition]]:
        program = encoding_fn(genome)

        def _program_step_fn(
            env_state: EnvState,
            program_state: ProgramState,
            random_key: RNGKey
        ) -> Tuple[EnvState, ProgramState, RNGKey, Transition]:
            observation = env_state.obs
            symmetric_observation = observation.take(symmetric_input_indexes)

            program_state_1, program_state_2 = jnp.split(program_state, 2)

            next_program_state_1, actions1 = program(observation, program_state_1)
            next_program_state_2, actions2 = program(symmetric_observation, program_state_2)
            next_program_state = jnp.concatenate([next_program_state_1, next_program_state_2])

            actions = action_joiner_function(actions1, actions2)

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

        return _program_step_fn

    return _genome_to_program_step_fn


def compute_genome_to_step_fn(
    environment: Env,
    config: Dict,
    outputs_wrapper: Callable[[jnp.ndarray], jnp.ndarray] = jnp.tanh
) -> Callable[
    [jnp.ndarray],
    Callable[[EnvState, ProgramState, RNGKey], Tuple[EnvState, ProgramState, RNGKey, Transition]]
]:
    encoding_fn = compute_encoding_function(config, outputs_wrapper)

    def _genome_to_program_step_fn(
        genome: jnp.ndarray
    ) -> Callable[[EnvState, ProgramState, RNGKey], Tuple[EnvState, ProgramState, RNGKey, Transition]]:
        program = encoding_fn(genome)

        def _program_step_fn(
            env_state: EnvState,
            program_state: ProgramState,
            random_key: RNGKey
        ) -> Tuple[EnvState, ProgramState, RNGKey, Transition]:
            program_inputs = env_state.obs
            next_program_state, actions = program(program_inputs, program_state)
            next_state = environment.step(env_state, actions)
            # TODO merge in a single field of the transition
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

        return _program_step_fn

    return _genome_to_program_step_fn


def _genome_to_cgp_program(
    genome: jnp.ndarray,
    config: Dict,
    outputs_wrapper: Callable[[jnp.ndarray], jnp.ndarray] = jnp.tanh
) -> Callable[[jnp.ndarray, ProgramState], Tuple[ProgramState, jnp.ndarray]]:
    x_genes, y_genes, f_genes, out_genes = jnp.split(genome, [config["n_nodes"] * i for i in range(1, 4)])

    def _program(inputs: jnp.ndarray, buffer: ProgramState) -> (ProgramState, jnp.ndarray):
        buffer = jnp.concatenate([inputs, constants[:config["n_constants"]], buffer[config["n_in"]:len(buffer)]])

        _, _, _, buffer = fori_loop(
            lower=config["n_in"],
            upper=len(buffer),
            body_fun=_update_buffer,
            init_val=(x_genes, y_genes, f_genes, buffer)
        )
        outputs = jnp.take(buffer, out_genes)
        bounded_outputs = outputs_wrapper(outputs)

        return buffer, bounded_outputs

    return _program


def _genome_to_lgp_program(
    genome: jnp.ndarray,
    config: Dict,
    outputs_wrapper: Callable[[jnp.ndarray], jnp.ndarray] = jnp.tanh
) -> Callable[[jnp.ndarray, ProgramState], Tuple[ProgramState, jnp.ndarray]]:
    lhs_genes, x_genes, y_genes, f_genes = jnp.split(genome, 4)
    output_positions = jnp.arange(start=config["n_registers"] - config["n_out"], stop=config["n_registers"])

    def _program(inputs: jnp.ndarray, register: ProgramState) -> (jnp.ndarray, ProgramState):
        register = jnp.zeros(config["n_registers"])
        register = jnp.concatenate(
            [inputs, constants[:config["n_constants"]], register[config["n_in"]:len(register)]])

        _, _, _, _, _, register = fori_loop(
            lower=0,
            upper=config["n_rows"],
            body_fun=_update_register,
            init_val=(lhs_genes, x_genes, y_genes, f_genes, config["n_in"], register)
        )
        outputs = jnp.take(register, output_positions)
        bounded_outputs = outputs_wrapper(outputs)

        return register, bounded_outputs

    return _program


@jit
def _update_buffer(
    buffer_idx: int,
    carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, ProgramState]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, ProgramState]:
    x_genes, y_genes, f_genes, buffer = carry
    n_in = len(buffer) - len(x_genes)
    idx = buffer_idx - n_in
    f_idx = f_genes.at[idx].get()
    x_arg = buffer.at[x_genes.at[idx].get()].get()
    y_arg = buffer.at[y_genes.at[idx].get()].get()

    buffer = buffer.at[buffer_idx].set(function_switch(f_idx, x_arg, y_arg))
    return x_genes, y_genes, f_genes, buffer


@jit
def _update_register(
    row_idx: int,
    carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, ProgramState]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, ProgramState]:
    lhs_genes, x_genes, y_genes, f_genes, n_in, register = carry
    lhs_idx = lhs_genes.at[row_idx].get() + n_in
    f_idx = f_genes.at[row_idx].get()
    x_idx = x_genes.at[row_idx].get()
    x_arg = register.at[x_idx].get()
    y_idx = y_genes.at[row_idx].get()
    y_arg = register.at[y_idx].get()
    register = register.at[lhs_idx].set(function_switch(f_idx, x_arg, y_arg))
    return lhs_genes, x_genes, y_genes, f_genes, n_in, register
