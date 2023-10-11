from __future__ import annotations

from typing import Callable

from jax import jit
from jax.tree_util import register_pytree_node_class
from functools import partial

import jax.numpy as jnp
from jax.lax import cond, switch


@register_pytree_node_class
class JaxFunction:

    def __init__(
        self,
        op: Callable[[float, float], float],
        arity: int,
        symbol: str = None
    ) -> None:
        self.operator = jit(op)
        self.arity = arity
        self.symbol = symbol if symbol is not None else op.__name__
        pass

    @partial(jit, static_argnums=0)
    def apply(
        self, x: float, y: float
    ) -> float:
        return self.operator(x, y)

    def __call__(
        self, x: float, y: float
    ) -> float:
        return self.apply(x, y)

    def tree_flatten(self):
        children = ()
        aux_data = {"operator": self.operator, "arity": self.arity, "symbol": self.symbol}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data["operator"], aux_data["arity"], aux_data["symbol"])


available_functions = {
    "plus": JaxFunction(lambda x, y: jnp.add(x, y), 2, "+"),
    "minus": JaxFunction(lambda x, y: jnp.add(x, -y), 2, "-"),
    "times": JaxFunction(lambda x, y: jnp.multiply(x, y), 2, "*"),
    "prot_div": JaxFunction(lambda x, y: cond(y == 0, jit(lambda a, b: 1.0), jnp.divide, x, y), 2, "/"),
    "abs": JaxFunction(lambda x, y: jnp.abs(x), 1, "abs"),
    "exp": JaxFunction(lambda x, y: jnp.exp(x), 1, "exp"),
    "sin": JaxFunction(lambda x, y: jnp.sin(x), 1, "sin"),
    "cos": JaxFunction(lambda x, y: jnp.cos(x), 1, "cos"),
    "prot_log": JaxFunction(lambda x, y: jnp.log(jnp.abs(x)), 1, "log"),
    "sqrt": JaxFunction(lambda x, y: jnp.sqrt(jnp.abs(x)), 1, "sqrt"),
    "lower": JaxFunction(lambda x, y: jnp.add(0.0, x < y), 2, "<"),
    "greater": JaxFunction(lambda x, y: jnp.add(0.0, x > y), 2, ">"),
    "lower0": JaxFunction(lambda x, y: jnp.add(0.0, x < 0.0), 1, "<0"),
    "greater0": JaxFunction(lambda x, y: jnp.add(0.0, x > 0.0), 1, ">0"),
}

function_arities = jnp.asarray([x.arity for x in available_functions.values()])
constants = jnp.asarray([0.1, 1])

arithmetic_functions = jnp.arange(4)
logical_functions = jnp.arange(start=len(available_functions) - 4 - 1, stop=len(available_functions), step=1)
trigonometric_functions = jnp.arange(start=len(arithmetic_functions), stop=logical_functions[0], step=1)


@jit
def function_switch(idx, *operands):
    return switch(idx, list(available_functions.values()), *operands)
