"""Core components of Cartesian Genetic Programming (CGP) for graph evolution."""

from typing import Callable, Dict, Tuple, Optional, Union

import jax.numpy as jnp
from flax import struct
from jax import random, jit
from jax.lax import fori_loop

from qdax.core.graphs.functions import FunctionSet
from qdax.core.graphs.utils import _mutate_subgenome
from qdax.custom_types import RNGKey, Genotype, Mask


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
            Typically set to the environment’s observation size, e.g., `env.observation_size`.
        n_outputs: number of outputs produced by the CGP individual.
            Typically set to the environment’s action size, e.g., `env.action_size`.
        n_nodes: number of computational nodes in the graph.
        function_set: set of allowed functions that nodes in the graph can use.
        input_constants: array of constant values that can be used as inputs
            alongside the external inputs.
        outputs_wrapper: function applied to the outputs of the CGP graph+
            before returning them to bound them in a certain range.
        fixed_outputs: whether the output nodes are fixed in their connections
            (last nodes in the sequence) or can be evolved.
    """
    n_inputs: int
    n_outputs: int
    n_nodes: int = 50
    function_set: FunctionSet = FunctionSet()
    input_constants: jnp.ndarray = jnp.asarray([0.1, 1.0])
    outputs_wrapper: Callable = jnp.tanh
    fixed_outputs: bool = False

    @property
    def buffer_size(self) -> int:
        """Size of the computation buffer used by CGP."""
        return self.n_inputs + len(self.input_constants) + self.n_nodes

    def init(
            self,
            rngs: RNGKey,
            *args,
    ) -> Genotype:
        """Initializes a random CGP genome.

            Args:
                rngs: JAX PRNG key used to generate random genome values.
                *args: Unused additional arguments for API compatibility.

            Returns:
                A dictionary containing the `"params"` key with genome sections as
                integer JAX arrays:
                    - `"inputs1"`
                    - `"inputs2"`
                    - `"functions"`
                    - `"outputs"`
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
                "inputs1": jnp.floor(random_x * in_mask).astype(int),
                "inputs2": jnp.floor(random_y * in_mask).astype(int),
                "functions": jnp.floor(random_f * f_mask).astype(int),
                "outputs": out_mask if self.fixed_outputs else
                jnp.floor(random_out * out_mask).astype(int)
            }
        }

    def apply(self,
              cgp_genome_params: Genotype,
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
                           carry: Tuple[Genotype, jnp.ndarray]
                           ) -> Tuple[Genotype, jnp.ndarray]:
            cgp_genes, buff = carry
            n_in = len(buff) - len(cgp_genes["params"]["inputs1"])
            idx = buffer_idx - n_in
            f_idx = cgp_genes["params"]["functions"].at[idx].get()
            x_arg = buff.at[cgp_genes["params"]["inputs1"].at[idx].get()].get()
            y_arg = buff.at[cgp_genes["params"]["inputs2"].at[idx].get()].get()
            f_computed = self.function_set.apply(f_idx, x_arg, y_arg)
            buff = buff.at[buffer_idx].set(f_computed)
            return cgp_genes, buff

        # initialize the buffer with inputs and constants and use zeros as placeholders for computation
        buffer = jnp.concatenate([obs, self.input_constants, jnp.zeros(self.n_nodes)])
        # apply the buffer update function for all positions of the buffer to update it
        _, buffer = fori_loop(
            lower=self.n_inputs + len(self.input_constants),
            upper=len(buffer),
            body_fun=_update_buffer,
            init_val=(cgp_genome_params, buffer))
        outputs = jnp.take(buffer, cgp_genome_params["params"]["outputs"])

        # apply wrapper to constraint the outputs in the correct domain
        return self.outputs_wrapper(outputs)

    def compute_active_nodes(
            self,
            cgp_genome_params: Genotype,
    ) -> Mask:
        """
        Compute the mask of active (expressed) nodes in a CGP genome.
        This method identifies which nodes are active by starting from the output
        connections and recursively marking all nodes that contribute to them.

        Args:
            cgp_genome_params: the CGP genome parameters.

        Returns:
            Mask: a binary mask (1 = active, 0 = inactive) of length `n_nodes`,
            indicating which nodes are used in producing the final outputs.
        """

        active_buffer = jnp.zeros(self.buffer_size)
        active_buffer = active_buffer.at[cgp_genome_params["params"]["outputs"]].set(1)

        # define function to mark if a buffer is active in a certain position
        def _compute_active_nodes(
                opposite_idx: int,
                carry: Tuple[Genotype, Mask],
        ) -> Tuple[Genotype, Mask]:
            cgp_genes, active = carry
            n_in = len(active) - len(cgp_genes["params"]["inputs1"])
            idx = len(active) - opposite_idx - 1
            x_idx = cgp_genes["params"]["inputs1"].at[idx - n_in].get().astype(int)
            y_idx = cgp_genes["params"]["inputs2"].at[idx - n_in].get().astype(int)
            arity = self.function_set.arities.at[cgp_genes["params"]["functions"][idx - n_in]].get()
            active = active.at[x_idx].set(jnp.logical_or(active.at[x_idx].get(), active.at[idx].get()))
            active = active.at[y_idx].set(jnp.logical_or(
                active.at[y_idx].get(), jnp.logical_and(active.at[idx].get(), arity == 2)
            ))
            return cgp_genes, active

        _, active_buffer = fori_loop(
            lower=0,
            upper=self.n_nodes,
            body_fun=_compute_active_nodes,
            init_val=(cgp_genome_params, active_buffer),
        )
        return active_buffer[-self.n_nodes:].astype(int)

    def get_readable_expression(
            self,
            cgp_genome_params: Genotype,
            inputs_mapping: Union[Dict[int, str], Callable[[int], str]] = None,
            outputs_mapping: Union[Dict[int, str], Callable[[int], str]] = None) -> str:
        """Generate a human-readable symbolic representation of a CGP genome.

            Unary functions are printed in the form:
                f(x)
            Binary functions are printed in the form:
                (x op y)
            where `op` is the function symbol (e.g., `+`, `*`, `sin`).

            Args:
                cgp_genome_params: CGP genotype.
                inputs_mapping (dict[int,str] | callable[[int], str]], optional):
                    Mapping from input indices to custom names.
                    - If a dict, keys are input indices
                    - If a callable, it is called with the input index and must
                      return the desired string
                    Defaults to "i0", "i1", ...
                outputs_mapping (dict[int,str] | callable[[int], str]], optional):
                    Mapping from output indices to custom names.
                    - If a dict, keys are output indices
                    - If a callable, it is called with the output index and must
                      return the desired string
                    Defaults to "o0", "o1", ...

            Returns:
                str: A multi-line string, with one line per output, showing the
                symbolic expression computed for each CGP output node.

            Example:
                o0 = (i0+i1)
                o1 = sin(i2)
            """
        inputs_mapping = inputs_mapping or {}
        if isinstance(inputs_mapping, dict):
            inputs_mapping_fn = lambda idx: inputs_mapping.get(idx, f"i{idx}")
        else:
            inputs_mapping_fn = inputs_mapping

        outputs_mapping = outputs_mapping or {}
        if isinstance(outputs_mapping, dict):
            outputs_mapping_fn = lambda idx: outputs_mapping.get(idx, f"o{idx}")
        else:
            outputs_mapping_fn = outputs_mapping

        n_in = self.n_inputs + len(self.input_constants)
        targets = []

        def _replace_cgp_expression(
                cgp_genes: Genotype,
                idx: int) -> str:
            if idx < self.n_inputs:
                return inputs_mapping_fn(int(idx))
            elif idx < n_in:
                return str(self.input_constants[idx - self.n_inputs])
            functions = list(self.function_set.function_set.values())
            gene_idx = idx - n_in
            function = functions[cgp_genes["params"]["functions"][gene_idx]]
            if function.arity == 1:
                return (f"{function.symbol}("
                        f"{_replace_cgp_expression(cgp_genes, int(cgp_genes['params']['inputs1'][gene_idx]))})")
            else:
                return f"({_replace_cgp_expression(cgp_genes, int(cgp_genes['params']['inputs1'][gene_idx]))}" \
                       f"{function.symbol}{_replace_cgp_expression(cgp_genes, int(cgp_genes['params']['inputs2'][gene_idx]))})"

        for i, out in enumerate(cgp_genome_params["params"]["outputs"]):
            targets.append(
                f"{outputs_mapping_fn(int(i))} = {self.outputs_wrapper.__name__}({_replace_cgp_expression(cgp_genome_params, out)})")

        return "\n".join(targets)


def cgp_mutation(
        genotype: Genotype,
        rnd_key: RNGKey,
        cgp: CGP,
        p_mut_inputs: float = 0.1,
        p_mut_functions: float = 0.1,
        p_mut_outputs: float = 0.3,
        mutation_probabilities: Optional[Dict[str, float]] = None
) -> Genotype:
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
            "inputs1": _mutate_subgenome(genotype["params"]["inputs1"],
                                         donor_genotype["params"]["inputs1"],
                                         x_key,
                                         p_mut_inputs),
            "inputs2": _mutate_subgenome(genotype["params"]["inputs2"],
                                         donor_genotype["params"]["inputs2"],
                                         y_key,
                                         p_mut_inputs),
            "functions": _mutate_subgenome(genotype["params"]["functions"],
                                           donor_genotype["params"]["functions"],
                                           f_key,
                                           p_mut_functions),
            "outputs": _mutate_subgenome(genotype["params"]["outputs"],
                                         donor_genotype["params"]["outputs"],
                                         out_key,
                                         p_mut_outputs),
        }
    }
