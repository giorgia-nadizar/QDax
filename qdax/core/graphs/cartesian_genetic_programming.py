"""Core components of Cartesian Genetic Programming (CGP) for graph evolution."""

from typing import Callable, Dict, Tuple, Optional, List

import jax.numpy as jnp
from flax import struct
from jax import random, jit
from jax.lax import fori_loop

from qdax.core.graphs.graph_genetic_programming import GGP, _mutate_subgenome
from qdax.custom_types import RNGKey, Genotype, Mask


@struct.dataclass
class CGP(GGP):
    """Cartesian Genetic Programming (CGP).

    Uses a fixed-length integer genome describing a directed acyclic graph
    of computational nodes arranged in a row (1D grid).

    Extra Args:
        n_nodes: number of computational nodes in the graph.
        fixed_outputs: whether output nodes are fixed (last nodes) or can be evolved.
    """

    n_nodes: int = 50
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
                A dictionary containing the `"genes"` key with genome sections as
                integer JAX arrays:
                    - `"inputs1"`
                    - `"inputs2"`
                    - `"functions"`
                    - `"outputs"`
                and the `"weights`" key with weights sections as floating point JAX arrays:
                    - `"inputs1"`
                    - `"inputs2"`
                    - `"functions"`
                The encoding is inspired by that of MLPs.
            """
        # determine bounds for genes for each section of the genome
        in_mask = jnp.arange(self.n_inputs + len(self.input_constants),
                             self.n_inputs + len(self.input_constants) + self.n_nodes)
        f_mask = len(self.function_set) * jnp.ones(self.n_nodes)
        out_mask = (self.n_inputs + len(self.input_constants) + self.n_nodes) * jnp.ones(self.n_outputs)

        # generate the random float values for each section of the genome
        n_key, out_key, weights_key = random.split(rngs, 3)
        random_n = random.uniform(key=n_key, shape=(self.n_nodes * 3,))
        random_x, random_y, random_f = jnp.split(random_n, 3)
        random_out = random.uniform(key=out_key, shape=out_mask.shape)
        random_weights = random.uniform(key=weights_key, shape=(self.n_nodes * 3,)) * 2 - 1

        # rescale, cast to integer and store the random genome parts
        return {
            "genes": {
                "inputs1": jnp.floor(random_x * in_mask).astype(int),
                "inputs2": jnp.floor(random_y * in_mask).astype(int),
                "functions": jnp.floor(random_f * f_mask).astype(int),
                "outputs": out_mask if self.fixed_outputs else jnp.floor(random_out * out_mask).astype(int),
            },
            "weights": self._init_weights(random_weights),
        }

    def apply(self,
              cgp_genome_params: Genotype,
              obs: jnp.ndarray,
              weights: Dict[str, jnp.ndarray] = None,
              ) -> jnp.ndarray:
        """Evaluates a CGP genome on a given input observation.

            This method interprets the integer-encoded genome to construct and
            execute the corresponding computational graph. Node values are computed
            sequentially and stored in a buffer, starting from the provided inputs and constants.

            Args:
                cgp_genome_params: dictionary of CGP genome parameters.
                weights: dictionary of weights for nodes and/or connections,
                    defaults to the CGP weights (or 1 if not weighted).
                obs: problem inputs/observation.

            Returns:
                Array of processed outputs after evaluating the genome and applying
                the output wrapper.
            """

        weights = weights or {}
        weights = {**cgp_genome_params["weights"], **weights}

        # define function to update buffer in a certain position: get inputs from the x and y connections
        # then apply the function
        @jit
        def _update_buffer(buffer_idx: int,
                           carry: Tuple[Genotype, jnp.ndarray]
                           ) -> Tuple[Genotype, jnp.ndarray]:
            cgp_genes, buff = carry
            n_in = len(buff) - len(cgp_genes["genes"]["inputs1"])
            idx = buffer_idx - n_in
            return self._update_memory(cgp_genes, weights, buff, idx, buffer_idx)

        # initialize the buffer with inputs and constants and use zeros as placeholders for computation
        buffer = jnp.concatenate([obs, self.input_constants, jnp.zeros(self.n_nodes)])
        # apply the buffer update function for all positions of the buffer to update it
        _, buffer = fori_loop(
            lower=self.n_inputs + len(self.input_constants),
            upper=len(buffer),
            body_fun=_update_buffer,
            init_val=(cgp_genome_params, buffer))
        outputs = jnp.take(buffer, cgp_genome_params["genes"]["outputs"])

        # apply wrapper to constraint the outputs in the correct domain
        return self.outputs_wrapper(outputs)

    def compute_active_mask(
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
        active_buffer = active_buffer.at[cgp_genome_params["genes"]["outputs"]].set(1)

        # define function to mark if a buffer is active in a certain position
        def _compute_active_nodes(
                opposite_idx: int,
                carry: Tuple[Genotype, Mask],
        ) -> Tuple[Genotype, Mask]:
            cgp_genes, active = carry
            n_in = len(active) - len(cgp_genes["genes"]["inputs1"])
            idx = len(active) - opposite_idx - 1
            x_idx = cgp_genes["genes"]["inputs1"].at[idx - n_in].get().astype(int)
            y_idx = cgp_genes["genes"]["inputs2"].at[idx - n_in].get().astype(int)
            arity = self.function_set.arities.at[cgp_genes["genes"]["functions"][idx - n_in]].get()
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

    def _get_readable_expression(
            self,
            cgp_genome_params: Genotype,
            inputs_mapping_fn: Callable[[int], str],
            outputs_mapping_fn: Callable[[int], str], ) -> List[str]:
        """Generate a human-readable symbolic representation of a CGP genome.

            Unary functions are printed in the form:
                f(x)
            Binary functions are printed in the form:
                (x op y)
            where `op` is the function symbol (e.g., `+`, `*`, `sin`).

            Args:
                cgp_genome_params: CGP genotype.
                inputs_mapping_fn: Mapping from input indices to custom names.
                outputs_mapping_fn Mapping from output indices to custom names.
            """
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
            function = functions[cgp_genes["genes"]["functions"][gene_idx]]
            node_weight, x_weight, y_weight = self._weights_representations(cgp_genes, gene_idx)
            if function.arity == 1:
                return (f"{node_weight}{function.symbol}({x_weight}"
                        f"{_replace_cgp_expression(cgp_genes, int(cgp_genes['genes']['inputs1'][gene_idx]))})")
            else:
                return f"{node_weight}({x_weight}{_replace_cgp_expression(cgp_genes, int(cgp_genes['genes']['inputs1'][gene_idx]))}" \
                       f"{function.symbol}{y_weight}{_replace_cgp_expression(cgp_genes, int(cgp_genes['genes']['inputs2'][gene_idx]))})"

        for i, out in enumerate(cgp_genome_params["genes"]["outputs"]):
            targets.append(
                f"{outputs_mapping_fn(int(i))} = {self.outputs_wrapper.__name__}({_replace_cgp_expression(cgp_genome_params, out)})")

        return targets


def cgp_mutation(
        genotype: Genotype,
        rnd_key: RNGKey,
        cgp: CGP,
        p_mut_inputs: float = 0.1,
        p_mut_functions: float = 0.1,
        p_mut_outputs: float = 0.3,
        weights_mut_sigma: float = 0.1,
        mutation_probabilities: Optional[Dict[str, float]] = None
) -> Genotype:
    """Mutates a CGP genome using int-flip mutation. If the genome is weighted, the weights
        are mutated with Gaussian mutation.

        This mutation is implemented as a form of crossover with a newly
        generated "donor" genome: for each gene, the value is taken from the
        donor with a low probability, otherwise kept from the original genome.
        This ensures that all mutated genes remain valid (i.e., within the
        correct index ranges for their respective genome section).

        The function is compatible with standard emitters when wrapped using
        `functools.partial` to pre-bind the `cgp` instance and mutation
        probabilities.

        Mutation probabilities and sigma can be specified either via individual arguments
        (`p_mut_inputs`, `p_mut_functions`, `p_mut_outputs`, `weights_mut_sigma`) or by passing a
        dictionary to `mutation_probabilities` with keys `"inputs"`, `"functions"`,
        `"outputs"`, and `"weights_sigma"`. When both are provided, the dictionary values override
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
            weights_mut_sigma: mutation step for weights Gaussian mutation
                (ignored if overridden via `mutation_probabilities`).
            mutation_probabilities: optional dictionary mapping `"inputs"`,
                `"functions"`, `"outputs"`, and `"weights_sigma"` to their mutation probabilities.

        Returns:
            The mutated genome.
        """

    # extract mutation probabilities if passed through a dictionary
    mutation_probabilities = mutation_probabilities or {}
    p_mut_inputs = mutation_probabilities.get("inputs", p_mut_inputs)
    p_mut_functions = mutation_probabilities.get("functions", p_mut_functions)
    p_mut_outputs = mutation_probabilities.get("outputs", p_mut_outputs)
    weights_mut_sigma = mutation_probabilities.get("weights_sigma", weights_mut_sigma)

    new_key, x_key, y_key, f_key, out_key, weights_key = random.split(rnd_key, 6)
    # generate the donor genotype -> only few genes from this will be used
    donor_genotype = cgp.init(new_key)

    weights_noise = weights_mut_sigma * random.normal(weights_key, shape=(cgp.n_nodes * 3,))
    node_w_noise, i1_w_noise, i2_w_noise = jnp.split(weights_noise, 3)

    # mutate each sub-part of the genome
    return {
        "genes": {
            "inputs1": _mutate_subgenome(genotype["genes"]["inputs1"],
                                         donor_genotype["genes"]["inputs1"],
                                         x_key,
                                         p_mut_inputs),
            "inputs2": _mutate_subgenome(genotype["genes"]["inputs2"],
                                         donor_genotype["genes"]["inputs2"],
                                         y_key,
                                         p_mut_inputs),
            "functions": _mutate_subgenome(genotype["genes"]["functions"],
                                           donor_genotype["genes"]["functions"],
                                           f_key,
                                           p_mut_functions),
            "outputs": _mutate_subgenome(genotype["genes"]["outputs"],
                                         donor_genotype["genes"]["outputs"],
                                         out_key,
                                         p_mut_outputs),
        },
        "weights": {
            "inputs1": genotype["weights"]["inputs1"] + cgp.weighted_inputs * i1_w_noise,
            "inputs2": genotype["weights"]["inputs2"] + cgp.weighted_inputs * i2_w_noise,
            "functions": genotype["weights"]["functions"] + cgp.weighted_functions * node_w_noise,
        }
    }
