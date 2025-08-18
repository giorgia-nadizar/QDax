"""Core components of Linear Genetic Programming (LGP) for graph evolution."""

from typing import Callable, Tuple

import jax.numpy as jnp
from flax import struct
from jax import random, jit
from jax.lax import fori_loop

from qdax.core.graphs.functions import FunctionSet
from qdax.custom_types import RNGKey, Genotype


@struct.dataclass
class LGP:
    """Linear Genetic Programming (LGP) representation.

    The LGP encoding uses a sequence of instructions (program lines) that
    operate on a set of registers to compute outputs. Each instruction
    selects one or more source operands (from input registers, constant
    registers, computation registers, or output registers), applies a function
    from the provided function set, and stores the result in a target register.

    The program executes sequentially, line by line, with later instructions
    potentially overwriting the results of earlier ones. The outputs of the
    program are taken from the last registers (i.e., output registers after execution.
    An optional output wrapper function (e.g., `tanh`) can be applied to
    constrain the final outputs to a desired range.

    Args:
        n_inputs: number of input values provided to the program (excluding constants).
            Typically set to the environment’s observation size, e.g., `env.observation_size`.
        n_outputs: number of outputs produced by the LGP individual.
            Typically set to the environment’s action size, e.g., `env.action_size`.
        n_computation_registers: number of internal registers available for intermediate
            computations. These registers are overwritten during program execution. Additional
            n_outputs_registers are also available for computation.
        n_program_lines: number of instructions in the program.
        function_set: set of allowed functions that instructions in the program can use.
        input_constants: array of constant values that can be used as additional inputs.
        outputs_wrapper: function applied to the outputs of the LGP program
            before returning them to bound them in a certain range.
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
        return self.n_inputs + len(self.input_constants) + self.n_assignable_registers

    @property
    def n_assignable_registers(self) -> int:
        """Number of registers that can be assigned by LGP."""
        return self.n_computation_registers + self.n_outputs

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
        lhs_mask = self.n_assignable_registers * jnp.ones(self.n_program_lines)
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

    def apply(self,
              lgp_genome_params: Genotype,
              obs: jnp.ndarray,
              ) -> jnp.ndarray:
        """Evaluates a LGP genome on a given input observation.

            This method interprets the integer-encoded genome to construct and
            execute the corresponding program. Program lines results are computed
            sequentially and stored in the registers, starting from the provided inputs and constants.

            Args:
                lgp_genome_params: dictionary of LGP genome parameters.
                obs: problem inputs/observation.

            Returns:
                Array of processed outputs after evaluating the genome and applying
                the output wrapper.
            """

        # define function to update the registers following the instructions of a 
        # given program line: get inputs from the x and y connections, then apply the function
        # and store the result in the target register
        @jit
        def _update_registers(line_idx: int,
                              carry: Tuple[Genotype, jnp.ndarray]
                              ) -> Tuple[Genotype, jnp.ndarray]:
            lgp_genes, regs = carry
            target_register_idx = lgp_genes["params"]["target_registers_genes"].at[line_idx].get()
            f_idx = lgp_genes["params"]["functions_genes"].at[line_idx].get()
            x_arg = regs.at[lgp_genes["params"]["x_connections_genes"].at[line_idx].get()].get()
            y_arg = regs.at[lgp_genes["params"]["y_connections_genes"].at[line_idx].get()].get()
            f_computed = self.function_set.apply(f_idx, x_arg, y_arg)
            regs = regs.at[target_register_idx].set(f_computed)
            return lgp_genes, regs

        # initialize the registers with inputs and constants and zeros for remaining registers
        registers = jnp.concatenate(
            [obs, self.input_constants, jnp.zeros(self.n_assignable_registers)])
        # apply the registers update function for all program lines
        _, registers = fori_loop(
            lower=0,
            upper=self.n_program_lines,
            body_fun=_update_registers,
            init_val=(lgp_genome_params, registers))
        outputs = registers[-self.n_outputs:]

        # apply wrapper to constraint the outputs in the correct domain
        return self.outputs_wrapper(outputs)
