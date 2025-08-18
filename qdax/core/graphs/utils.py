import jax.numpy as jnp
from jax import random

from qdax.custom_types import RNGKey


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
