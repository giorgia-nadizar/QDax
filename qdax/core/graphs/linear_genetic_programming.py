"""Core components of Linear Genetic Programming (LGP) for graph evolution."""

from typing import Callable, Tuple, Optional, Dict

import jax.numpy as jnp
from flax import struct
from jax import random, jit
from jax.lax import fori_loop

from qdax.core.graphs.utils import _mutate_subgenome
from qdax.core.graphs.functions import FunctionSet
from qdax.custom_types import RNGKey, Genotype, Mask


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

    def compute_active_lines(
            self,
            lgp_genome_params: Genotype,
    ) -> Mask:
        """
        Compute the mask of active (expressed) program lines in a LGP genome.
        This method identifies which lines are active by starting from the output
        registers and recursively marking all lines that contribute to them.

        Args:
            lgp_genome_params: the CGP genome parameters.

        Returns:
            Mask: a binary mask (1 = active, 0 = inactive) of length `n_program_lines`,
            indicating which lines are used in producing the final outputs.
        """

        active_lines = jnp.zeros(self.n_program_lines)
        registers_mask = jnp.where(jnp.arange(self.n_registers) >= (self.n_registers - self.n_outputs), 1, 0)

        # define function to mark if a line is active
        def _compute_active_lines(
                opposite_idx: int,
                carry: Tuple[Genotype, Mask, Mask]
        ) -> Tuple[Genotype, Mask, Mask]:
            lgp_genes, active, regs_mask = carry
            line_idx = len(active) - opposite_idx - 1
            line_use = regs_mask.at[lgp_genes["params"]["target_registers_genes"].at[line_idx].get()].get()
            active = active.at[line_idx].set(line_use)

            x_reg = lgp_genes["params"]["x_connections_genes"].at[line_idx].get()
            y_reg = lgp_genes["params"]["y_connections_genes"].at[line_idx].get()
            arity = self.function_set.arities.at[lgp_genes["params"]["functions_genes"][line_idx]].get()
            regs_mask = regs_mask.at[line_idx].set(0)
            regs_mask = regs_mask.at[x_reg].set(jnp.logical_or(line_use, regs_mask.at[x_reg].get()))
            regs_mask = regs_mask.at[y_reg].set(jnp.logical_or(
                regs_mask.at[y_reg].get(), jnp.logical_and(line_use, arity == 2)
            ))

            return lgp_genes, active, regs_mask

        _, active_lines, _ = fori_loop(
            lower=0,
            upper=self.n_program_lines,
            body_fun=_compute_active_lines,
            init_val=(lgp_genome_params, active_lines, registers_mask)
        )
        return active_lines.astype(int)

    def get_readable_program(
            self,
            lgp_genome_params: Genotype) -> str:
        """Generate a human-readable Python-like representation of an LGP program.

            The LGP genome is unrolled into a sequence of instructions that
            operate on registers. Inputs are first copied into registers, then
            program lines are expanded into assignment statements. Only active
            lines (contributing to the outputs) are included.

            Unary functions are printed in the form:
                r[target] = f(r[x])
            Binary functions are printed in the form:
                r[target] = r[x] op r[y]
            where `op` is the function symbol (e.g., `+`, `*`, `sin`).

            The outputs are read from the last `n_outputs` registers.

            Args:
                lgp_genome_params: LGP genotype.

            Returns:
                str: A string representing the program as a Python function,
                showing register initialization, executed instructions, and
                the final outputs.

            Example:
                def program(inputs):
                    r[[0, 1]] = inputs
                    r[[2]] = [0.1]
                    r[3] = r[0] + r[2]
                    r[4] = tanh(r[3])
                    outputs = r[[3, 4]]
                    return outputs
            """
        # header and inputs copy into registers
        program_lines = [f"def program(inputs):",
                         f"r[{list(range(self.n_inputs))}] = inputs",
                         f"r[{list(range(self.n_inputs, self.n_inputs + len(self.input_constants)))}] = {self.input_constants}"]

        functions = list(self.function_set.function_set.values())
        active_lines = self.compute_active_lines(lgp_genome_params)

        # execution
        for line_idx in range(self.n_program_lines):
            if active_lines[line_idx]:
                function = functions[lgp_genome_params["params"]["functions_genes"][line_idx]]
                target_reg = lgp_genome_params['params']['target_registers_genes'][line_idx]
                x_reg = lgp_genome_params['params']['x_connections_genes'][line_idx]
                y_reg = lgp_genome_params['params']['y_connections_genes'][line_idx]
                if function.arity > 1:
                    program_lines.append(f"r[{target_reg}] = r[{x_reg}] {function.symbol} r[{y_reg}]")
                else:
                    program_lines.append(f"r[{target_reg}] = {function.symbol}(r[{x_reg}])")

        # output selection
        program_lines.append(f"outputs = r[{list(range(self.n_registers - self.n_outputs, self.n_registers))}]")
        program_lines.append("return outputs")
        return "\n\t".join(program_lines)


def lgp_mutation(
        genotype: Genotype,
        rnd_key: RNGKey,
        lgp: LGP,
        p_mut_targets: float = 0.3,
        p_mut_inputs: float = 0.1,
        p_mut_functions: float = 0.1,
        mutation_probabilities: Optional[Dict[str, float]] = None
) -> Genotype:
    """Mutates a LGP genome using int-flip mutation.

        This mutation is implemented as a form of crossover with a newly
        generated "donor" genome: for each gene, the value is taken from the
        donor with a low probability, otherwise kept from the original genome.
        This ensures that all mutated genes remain valid (i.e., within the
        correct index ranges for their respective genome section).

        The function is compatible with standard emitters when wrapped using
        `functools.partial` to pre-bind the `lgp` instance and mutation
        probabilities.

        Mutation probabilities can be specified either via individual arguments
        (`p_mut_assignment_targets`, `p_mut_inputs`, `p_mut_functions`) or by passing a
        dictionary to `mutation_probabilities` with keys `"inputs"`, `"functions"`,
        and `"targets"`. When both are provided, the dictionary values override
        the individual arguments.

        Args:
            genotype: the CGP genome parameters to mutate.
            rnd_key: JAX PRNG key for randomness.
            lgp: LGP instance used to initialize the donor genome.
            p_mut_targets: probability of mutating each target assignment gene
                (ignored if overridden via `mutation_probabilities`).
            p_mut_inputs: probability of mutating each input connection gene
                (ignored if overridden via `mutation_probabilities`).
            p_mut_functions: probability of mutating each function gene
                (ignored if overridden via `mutation_probabilities`).
            mutation_probabilities: optional dictionary mapping `"inputs"`,
                `"functions"`, and `"targets"` to their mutation probabilities.

        Returns:
            The mutated genome.
        """

    # extract mutation probabilities if passed through a dictionary
    mutation_probabilities = mutation_probabilities or {}
    p_mut_targets = mutation_probabilities.get("target", p_mut_targets)
    p_mut_inputs = mutation_probabilities.get("inputs", p_mut_inputs)
    p_mut_functions = mutation_probabilities.get("functions", p_mut_functions)

    new_key, targets_key, x_key, y_key, f_key = random.split(rnd_key, 5)
    # generate the donor genotype -> only few genes from this will be used
    donor_genotype = lgp.init(new_key)

    # mutate each sub-part of the genome
    return {
        "params": {
            "target_registers_genes": _mutate_subgenome(genotype["params"]["target_registers_genes"],
                                                        donor_genotype["params"]["target_registers_genes"],
                                                        targets_key,
                                                        p_mut_targets),
            "x_connections_genes": _mutate_subgenome(genotype["params"]["x_connections_genes"],
                                                     donor_genotype["params"]["x_connections_genes"],
                                                     x_key,
                                                     p_mut_inputs),
            "y_connections_genes": _mutate_subgenome(genotype["params"]["y_connections_genes"],
                                                     donor_genotype["params"]["y_connections_genes"],
                                                     y_key,
                                                     p_mut_inputs),
            "functions_genes": _mutate_subgenome(genotype["params"]["functions_genes"],
                                                 donor_genotype["params"]["functions_genes"],
                                                 f_key,
                                                 p_mut_functions),
        }
    }
