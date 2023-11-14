from typing import Callable

import jax.numpy as jnp

from qdax.types import Action


def get_symmetric_input_indexes(env_name: str) -> jnp.ndarray:
    if "walker" in env_name:
        return jnp.asarray([0, 1, 5, 6, 7, 2, 3, 4, 8, 9, 10, 14, 15, 16, 11, 12, 13])
    if "ant" in env_name:
        return jnp.asarray(
            [0, 1, 2, 3, 4, 7, 8, 5, 6, 11, 12, 9, 10, 13, 14, 15, 16, 17, 18, 21, 22, 19, 20, 25, 26, 23, 24])
    else:
        raise ValueError("Symmetry not available for selected environment.")


def get_action_joiner_func(env_name: str) -> Callable[[Action, Action], Action]:
    if "walker" in env_name:
        def _join_walker_actions(actions1: Action, actions2: Action) -> Action:
            return jnp.concatenate([actions1, actions2])

        return _join_walker_actions
    if "ant" in env_name:
        def _join_ant_actions(actions1: Action, actions2: Action) -> Action:
            actions1a, actions1b = jnp.split(actions1, 2)
            actions2a, actions2b = jnp.split(actions2, 2)
            return jnp.concatenate([actions1a, actions2a, actions1b, actions2b])

        return _join_ant_actions
    else:
        raise ValueError("Symmetry not available for selected environment.")
