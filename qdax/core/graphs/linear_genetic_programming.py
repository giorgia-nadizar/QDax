"""Core components of Linear Genetic Programming (LGP) for graph evolution."""

from typing import Callable

import jax.numpy as jnp
from flax import struct
from jax import random

from qdax.core.graphs.functions import FunctionSet
from qdax.custom_types import RNGKey, Genotype


@struct.dataclass
class LGP:
    """Linear Genetic Programming (LGP) representation.
    """
    n_inputs: int
    n_outputs: int
    n_computation_registers: int = 5
    n_program_lines: int = 15
    function_set: FunctionSet = FunctionSet()
    input_constants: jnp.ndarray = jnp.asarray([0.1, 1.0])
    outputs_wrapper: Callable = jnp.tanh

    @property
    def n_registers(self) -> int:
        """Total number of registers used by LGP."""
        return self.n_inputs + len(self.input_constants) + self.n_computation_registers + self.n_outputs

    def init(
            self,
            rngs: RNGKey,
            *args,
    ) -> Genotype:
        """Initializes a random LGP genome.

            Args:
                rngs: JAX PRNG key used to generate random genome values.
                *args: Unused additional arguments for API compatibility.

            Returns:
                A dictionary containing the `"params"` key with genome sections as
                integer JAX arrays:
                    - `"target_registers_genes"`
                    - `"x_arguments_genes"`
                    - `"y_arguments_genes"`
                    - `"functions_genes"`
                The encoding is inspired by that of MLPs.
            """
        # determine bounds for genes for each section of the genome
        lhs_mask = (self.n_computation_registers + self.n_outputs) * jnp.ones(self.n_program_lines)
        lhs_offset = (self.n_inputs + len(self.input_constants)) * jnp.ones(self.n_program_lines)
        f_mask = len(self.function_set) * jnp.ones(self.n_program_lines)
        rhs_mask = self.n_registers * jnp.ones(self.n_program_lines)

        # generate the random float values for each section of the genome
        targets_key, x_key, y_key, f_key = random.split(rngs, 4)
        random_targets = random.uniform(key=targets_key, shape=lhs_mask.shape)
        random_x = random.uniform(key=x_key, shape=rhs_mask.shape)
        random_y = random.uniform(key=y_key, shape=rhs_mask.shape)
        random_f = random.uniform(key=f_key, shape=f_mask.shape)

        # rescale, cast to integer and store the random genome parts
        return {
            "params": {
                "target_registers_genes": (jnp.floor(random_targets * lhs_mask) + lhs_offset).astype(int),
                "x_connections_genes": jnp.floor(random_x * rhs_mask).astype(int),
                "y_connections_genes": jnp.floor(random_y * rhs_mask).astype(int),
                "functions_genes": jnp.floor(random_f * f_mask).astype(int)
            }
        }