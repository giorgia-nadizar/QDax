"""Core components of Cartesian Genetic Programming (CGP) for graph evolution."""

from typing import Callable, Union, Dict, Any, Tuple, Optional

import jax.numpy as jnp
from flax import struct
from flax.typing import FrozenVariableDict
from jax import random, jit
from jax.lax import fori_loop

from qdax.core.graphs.functions import FunctionSet
from qdax.custom_types import RNGKey


@struct.dataclass
class CGP:
    """Cartesian Genetic Programming (CGP) representation.

    The CGP encoding uses a fixed-length integer genome to describe a
    directed acyclic graph of computational nodes arranged in a single row
    (1D grid). Each node has up to two inputs, chosen from problem inputs,
    constant values, or outputs of earlier nodes, and applies a function
    from the provided function set.

    Problem outputs are taken from specific nodes in the graph. These
    connections can be evolved or fixed to the last nodes.
    An optional output wrapper function (e.g., `tanh`) can be applied
    to constrain the final outputs to a desired range.

    Args:
        n_inputs: number of input values provided to the graph (excluding constants).
        n_nodes: number of computational nodes in the graph.
        n_outputs: number of outputs produced by the CGP individual.
        function_set: set of allowed functions that nodes in the graph can use.
        input_constants: array of constant values that can be used as inputs
            alongside the external inputs.
        outputs_wrapper: function applied to the outputs of the CGP graph+
            before returning them to bound them in a certain range.
        fixed_outputs: whether the output nodes are fixed in their connections
            (last nodes in the sequence) or can be evolved.
    """
    n_inputs: int
    n_nodes: int
    n_outputs: int
    function_set: FunctionSet
    input_constants: jnp.ndarray = jnp.asarray([0.1, 1.0])
    outputs_wrapper: Callable = jnp.tanh
    fixed_outputs: bool = False

    def init(
            self,
            rngs: RNGKey,
            *args,
    ) -> Union[FrozenVariableDict, Dict[str, Any]]:
        """Initializes a random CGP genome.

            Args:
                rngs: JAX PRNG key used to generate random genome values.
                *args: Unused additional arguments for API compatibility.

            Returns:
                A dictionary containing the `"params"` key with genome sections as
                integer JAX arrays:
                    - `"x_connections_genes"`
                    - `"y_connections_genes"`
                    - `"functions_genes"`
                    - `"output_connections_genes"`
                The encoding is inspired by that of MLPs.
            """
        # determine bounds for genes for each section of the genome
        in_mask = jnp.arange(self.n_inputs + len(self.input_constants),
                             self.n_inputs + len(self.input_constants) + self.n_nodes)
        f_mask = len(self.function_set) * jnp.ones(self.n_nodes)
        out_mask = (self.n_inputs + len(self.input_constants) + self.n_nodes) * jnp.ones(self.n_outputs)

        # generate the random float values for each section of the genome
        x_key, y_key, f_key, out_key = random.split(rngs, 4)
        random_x = random.uniform(key=x_key, shape=in_mask.shape)
        random_y = random.uniform(key=y_key, shape=in_mask.shape)
        random_f = random.uniform(key=f_key, shape=f_mask.shape)
        random_out = random.uniform(key=out_key, shape=out_mask.shape)

        # rescale, cast to integer and store the random genome parts
        return {
            "params": {
                "x_connections_genes": jnp.floor(random_x * in_mask).astype(int),
                "y_connections_genes": jnp.floor(random_y * in_mask).astype(int),
                "functions_genes": jnp.floor(random_f * f_mask).astype(int),
                "output_connections_genes": out_mask if self.fixed_outputs else
                jnp.floor(random_out * out_mask).astype(int)
            }
        }

    def apply(self,
              cgp_genome_params: Union[FrozenVariableDict, Dict[str, Any]],
              obs: jnp.ndarray,
              ) -> jnp.ndarray:
        """Evaluates a CGP genome on a given input observation.

            This method interprets the integer-encoded genome to construct and
            execute the corresponding computational graph. Node values are computed
            sequentially and stored in a buffer, starting from the provided inputs and constants.

            Args:
                cgp_genome_params: dictionary of CGP genome parameters.
                obs: problem inputs/observation.

            Returns:
                Array of processed outputs after evaluating the genome and applying
                the output wrapper.
            """

        # define function to update buffer in a certain position: get inputs from the x and y connections
        # then apply the function
        @jit
        def _update_buffer(buffer_idx: int,
                           carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) \
                -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            x_genes, y_genes, f_genes, buff = carry
            n_in = len(buff) - len(x_genes)
            idx = buffer_idx - n_in
            f_idx = f_genes.at[idx].get()
            x_arg = buff.at[x_genes.at[idx].get()].get()
            y_arg = buff.at[y_genes.at[idx].get()].get()
            f_computed = self.function_set.apply(f_idx, x_arg, y_arg)
            buff = buff.at[buffer_idx].set(f_computed)
            return x_genes, y_genes, f_genes, buff

        # initialize the buffer with inputs and constants and use zeros as placeholders for computation
        buffer = jnp.concatenate([obs, self.input_constants, jnp.zeros(self.n_nodes)])
        # apply the buffer update function for all positions of the buffer to update it
        _, _, _, buffer = fori_loop(self.n_inputs, len(buffer), _update_buffer,
                                    (cgp_genome_params["params"]["x_connections_genes"],
                                     cgp_genome_params["params"]["y_connections_genes"],
                                     cgp_genome_params["params"]["functions_genes"], buffer))
        outputs = jnp.take(buffer, cgp_genome_params["params"]["output_connections_genes"])

        # apply wrapper to constraint the outputs in the correct domain
        return self.outputs_wrapper(outputs)


def _mutate_subgenome(
        x1: jnp.ndarray,
        x2: jnp.ndarray,
        key: RNGKey,
        p_mut: float
) -> jnp.ndarray:
    """Performs elementwise mutation of a genome section.

        For each gene, a random number in [0, 1) is drawn. If the number is
        greater than `p_mut`, the gene is kept from the original subgenome (`x1`);
        otherwise, it is replaced with the corresponding gene from the donor
        subgenome (`x2`).

        Args:
            x1: Original subgenome array.
            x2: Donor subgenome array (must be the same shape as `x1`).
            key: JAX PRNG key used to generate mutation probabilities.
            p_mut: Probability of replacing each gene with the donor's value.

        Returns:
            The mutated subgenome array.
        """
    mutation_probs = random.uniform(key=key, shape=x1.shape)
    return jnp.where(mutation_probs > p_mut, x1, x2).astype(int)


def cgp_mutation(
        genotype: FrozenVariableDict,
        rnd_key: RNGKey,
        cgp: CGP,
        p_mut_inputs: float = 0.1,
        p_mut_functions: float = 0.1,
        p_mut_outputs: float = 0.3,
        mutation_probabilities: Optional[Dict[str, float]] = None
) -> Union[FrozenVariableDict, Dict[str, Any]]:
    """Mutates a CGP genome using int-flip mutation.

        This mutation is implemented as a form of crossover with a newly
        generated "donor" genome: for each gene, the value is taken from the
        donor with a low probability, otherwise kept from the original genome.
        This ensures that all mutated genes remain valid (i.e., within the
        correct index ranges for their respective genome section).

        The function is compatible with standard emitters when wrapped using
        `functools.partial` to pre-bind the `cgp` instance and mutation
        probabilities.

        Mutation probabilities can be specified either via individual arguments
        (`p_mut_inputs`, `p_mut_functions`, `p_mut_outputs`) or by passing a
        dictionary to `mutation_probabilities` with keys `"inputs"`, `"functions"`,
        and `"outputs"`. When both are provided, the dictionary values override
        the individual arguments.

        Args:
            genotype: the CGP genome parameters to mutate.
            rnd_key: JAX PRNG key for randomness.
            cgp: CGP instance used to initialize the donor genome.
            p_mut_inputs: probability of mutating each input connection gene
                (ignored if overridden via `mutation_probabilities`).
            p_mut_functions: probability of mutating each function gene
                (ignored if overridden via `mutation_probabilities`).
            p_mut_outputs: probability of mutating each output connection gene
                (ignored if overridden via `mutation_probabilities`).
            mutation_probabilities: optional dictionary mapping `"inputs"`,
                `"functions"`, and `"outputs"` to their mutation probabilities.

        Returns:
            The mutated genome.
        """

    # extract mutation probabilities if passed through a dictionary
    mutation_probabilities = mutation_probabilities or {}
    p_mut_inputs = mutation_probabilities.get("inputs", p_mut_inputs)
    p_mut_functions = mutation_probabilities.get("functions", p_mut_functions)
    p_mut_outputs = mutation_probabilities.get("outputs", p_mut_outputs)

    new_key, x_key, y_key, f_key, out_key = random.split(rnd_key, 5)
    # generate the donor genotype -> only few genes from this will be used
    donor_genotype = cgp.init(new_key)

    # mutate each sub-part of the genome
    return {
        "params": {
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
            "output_connections_genes": _mutate_subgenome(genotype["params"]["output_connections_genes"],
                                                          donor_genotype["params"]["output_connections_genes"],
                                                          out_key,
                                                          p_mut_outputs),
        }
    }
