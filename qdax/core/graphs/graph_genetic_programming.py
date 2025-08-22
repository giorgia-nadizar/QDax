from flax import struct
import jax.numpy as jnp
from typing import Callable, Dict, Union, List, Tuple

from jax import random

from qdax.core.graphs.functions import FunctionSet
from qdax.custom_types import RNGKey, Genotype, Mask


@struct.dataclass
class GGP:
    """Base class for Graph-based Genetic Programming (GGP) representations.

    Common parameters for GP encodings:

    Args:
        n_inputs: number of input values provided to the program/graph (excluding constants).
            Typically set to the environment’s observation size, e.g., `env.observation_size`.
        n_outputs: number of outputs produced by the GP individual.
            Typically set to the environment’s action size, e.g., `env.action_size`.
        function_set: set of allowed functions that nodes/instructions can use.
        input_constants: array of constant values that can be used as additional inputs.
        outputs_wrapper: function applied to the outputs before returning them
            (e.g., `tanh` to bound outputs).
        weighted_functions: whether the genome will contain weighting factors for each node/program line.
        weighted_inputs: whether the genome will contain weighting factors for each connection.
    """

    n_inputs: int
    n_outputs: int
    function_set: FunctionSet = FunctionSet()
    input_constants: jnp.ndarray = jnp.asarray([0.1, 1.0])
    outputs_wrapper: Callable = jnp.tanh
    weighted_functions: bool = False
    weighted_inputs: bool = False

    def init(self, rngs: RNGKey, *args):
        """Initialize a random genome (to be implemented by subclasses)."""
        raise NotImplementedError

    def apply(self, genome_params: Genotype, obs: jnp.ndarray, weights: Dict[str, jnp.ndarray] = None, ) -> jnp.ndarray:
        """Evaluate a genome on an input observation (subclass-specific)."""
        raise NotImplementedError

    def compute_active_mask(self, genome_params: Genotype, ) -> Mask:
        """Compute the mask of active (expressed) elements in a genome (subclass-specific)."""
        raise NotImplementedError

    def get_readable_expression(
            self,
            genome_params: Genotype,
            inputs_mapping: Union[Dict[int, str], Callable[[int], str]] = None,
            outputs_mapping: Union[Dict[int, str], Callable[[int], str]] = None
    ) -> str:
        """Generate a human-readable symbolic representation of a GGP genome.

            Unary functions are printed in the form:
                f(x)
            Binary functions are printed in the form:
                (x op y)
            where `op` is the function symbol (e.g., `+`, `*`, `sin`).

            Args:
                genome_params: GGP genotype.
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

        targets = self._get_readable_expression(genome_params, inputs_mapping_fn, outputs_mapping_fn)
        return "\n".join(targets)

    def _get_readable_expression(
            self,
            genome_params: Genotype,
            inputs_mapping_fn: Callable[[int], str],
            outputs_mapping_fn: Callable[[int], str]
    ) -> List[str]:
        """Worker class for """
        raise NotImplementedError

    def _weights_representations(self, genome: Genotype, gene_idx: int) -> Tuple[str, str, str]:
        input_weight = f"{genome['weights']['functions'][gene_idx]:.2f}*" if self.weighted_functions else ""
        x_weight = f"{genome['weights']['inputs1'][gene_idx]:.2f}*" if self.weighted_inputs else ""
        y_weight = f"{genome['weights']['inputs2'][gene_idx]:.2f}*" if self.weighted_inputs else ""
        return input_weight, x_weight, y_weight

    def _init_weights(self, random_weights: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Initialize the weights' dictionary."""
        random_node_weights, random_input_weights1, random_input_weights2 = jnp.split(random_weights, 3)
        return {
            "functions": random_node_weights if self.weighted_functions else jnp.ones_like(random_node_weights),
            "inputs1": random_input_weights1 if self.weighted_inputs else jnp.ones_like(random_input_weights1),
            "inputs2": random_input_weights2 if self.weighted_inputs else jnp.ones_like(random_input_weights2),
        }

    def _update_memory(self,
                       genome: Genotype,
                       weights: Dict[str, jnp.ndarray],
                       memory: jnp.ndarray,
                       gene_idx: int,
                       memory_idx: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        f_idx = genome["genes"]["functions"].at[gene_idx].get()
        x_arg = memory.at[genome["genes"]["inputs1"].at[gene_idx].get()].get() * weights["inputs1"].at[gene_idx].get()
        y_arg = memory.at[genome["genes"]["inputs2"].at[gene_idx].get()].get() * weights["inputs2"].at[gene_idx].get()
        f_computed = self.function_set.apply(f_idx, x_arg, y_arg) * weights["functions"].at[gene_idx].get()
        memory = memory.at[memory_idx].set(f_computed)
        return genome, memory


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
