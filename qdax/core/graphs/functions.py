from __future__ import annotations

from typing import Callable, Dict, Optional

from jax import jit
from jax.tree_util import register_pytree_node_class
from functools import partial

import jax.numpy as jnp
from jax.lax import cond, switch


@register_pytree_node_class
class JaxFunction:
    """JAX-compatible wrapper for a binary or unary function.

        Wraps a numerical or logical operation so it can be used in JAX-traced
        computations, registered as a PyTree node, and efficiently applied via
        `jit`.

        Args:
            op: callable implementing the function. Must accept two arguments
                (`x`, `y`), even for unary operations (second argument ignored).
            arity: number of arguments the function actually uses (1 or 2).
            symbol: optional human-readable symbol for the function. Defaults to
                the name of the provided `op`.

        Attributes:
            operator: JIT-compiled version of the provided operation.
            arity: number of inputs the function uses.
            symbol: display string for the function.

        Methods:
            apply(x, y): Applies the function to inputs `x` and `y`.
            __call__(x, y): Alias for `.apply()`.
            tree_flatten(): PyTree flattening, stores metadata.
            tree_unflatten(): Reconstructs the function from metadata.
        """

    def __init__(self, op: Callable[[float, float], float], arity: int, symbol: str = None) -> None:
        self.operator = jit(op)
        self.arity = arity
        self.symbol = symbol if symbol is not None else op.__name__
        pass

    @partial(jit, static_argnums=0)
    def apply(self, x, y):
        return self.operator(x, y)

    def __call__(self, x, y):
        return self.apply(x, y)

    def tree_flatten(self):
        children = ()
        aux_data = {"operator": self.operator, "arity": self.arity, "symbol": self.symbol}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data["operator"], aux_data["arity"], aux_data["symbol"])


# Predefined dictionary of numeric JaxFunction instances covering common mathematical operations
function_set_numeric = {
    "minus": JaxFunction(lambda x, y: jnp.add(x, -y), 2, "-"),
    "times": JaxFunction(lambda x, y: jnp.multiply(x, y), 2, "*"),
    "prot_div": JaxFunction(lambda x, y: cond(y == 0, jit(lambda a, b: 1.0), jnp.divide, x, y), 2, "/"),
    "abs": JaxFunction(lambda x, y: jnp.abs(x), 1, "abs"),
    "exp": JaxFunction(lambda x, y: jnp.exp(x), 1, "exp"),
    "sin": JaxFunction(lambda x, y: jnp.sin(x), 1, "sin"),
    "cos": JaxFunction(lambda x, y: jnp.cos(x), 1, "cos"),
    "prot_log": JaxFunction(lambda x, y: jnp.log(jnp.abs(x)), 1, "log"),
    "lower": JaxFunction(lambda x, y: jnp.add(0.0, x < y), 2, "<"),
    "greater": JaxFunction(lambda x, y: jnp.add(0.0, x > y), 2, ">"),
    "sqrt": JaxFunction(lambda x, y: jnp.sqrt(jnp.abs(x)), 1, "sqrt"),
}

# Predefined dictionary of boolean JaxFunction instances implementing logical operations
function_set_boolean = {
    "and": JaxFunction(lambda x, y: jnp.logical_and(x, y), 2, "and"),
    "or": JaxFunction(lambda x, y: jnp.logical_or(x, y), 2, "or"),
    "xor": JaxFunction(lambda x, y: jnp.logical_xor(x, y), 2, "xor"),
    "and_not": JaxFunction(lambda x, y: jnp.logical_and(x, jnp.logical_not(y)), 2, "and_not"),
}


@register_pytree_node_class
class FunctionSet:
    """Container for a set of JAX-compatible functions.

        Stores a mapping of function names to `JaxFunction` objects and provides
        a JIT-compiled `apply` method that selects and executes a function by
        index. Designed for use in algorithms like Cartesian Genetic Programming
        where functions are chosen dynamically at runtime.

        Args:
            functions_dict: dictionary mapping function names to `JaxFunction`
                objects. Defaults to `function_set_numeric`, a commonly used set
                of functions processing numerical values.

        Attributes:
            function_set: the stored dictionary of named functions.
            apply: JIT-compiled function that takes an index and arguments, and
                executes the corresponding `JaxFunction`.

        Methods:
            __len__(): returns the number of functions in the set.
            tree_flatten(): PyTree flattening, stores metadata.
            tree_unflatten(): reconstructs the function set from metadata.
        """

    def __init__(self,
                 functions_dict: Optional[Dict[str, JaxFunction]] = None
                 ) -> None:
        self.function_set = functions_dict or function_set_numeric

        @jit
        def function_switch(idx, *operands):
            return switch(idx, list(function_set_numeric.values()), *operands)

        self.apply = function_switch

    def __len__(self):
        return len(self.function_set)

    def tree_flatten(self):
        children = ()
        aux_data = {"function_set": self.function_set}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data["function_set"])
