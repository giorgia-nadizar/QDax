from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap

from qdax.types import RNGKey


def truncation_selection(
    genomes: jnp.ndarray,
    fitness_values: jnp.ndarray,
    rnd_key: RNGKey,
    n_samples: int
) -> jnp.ndarray:
    elites, _ = jnp.split(jnp.argsort(-fitness_values), [n_samples])
    return jnp.take(genomes, elites, axis=0)


def fp_selection(
    genomes: jnp.ndarray,
    fitness_values: jnp.ndarray,
    rnd_key: RNGKey,
    n_samples: int
) -> jnp.ndarray:
    p = 1 - ((jnp.max(fitness_values) - fitness_values) / (jnp.max(fitness_values) - jnp.min(fitness_values)))
    p /= jnp.sum(p)
    return jax.random.choice(rnd_key, genomes, shape=[n_samples], p=p, replace=False)


def tournament_selection(
    genomes: jnp.ndarray,
    fitness_values: jnp.ndarray,
    rnd_key: RNGKey,
    n_samples: int,
    tour_size: int
) -> jnp.ndarray:
    def _tournament(sample_key: RNGKey, genomes: jnp.ndarray, fitnesses: jnp.ndarray, tour_size: int) -> jnp.ndarray:
        indexes = jax.random.choice(sample_key, jnp.arange(start=0, stop=len(genomes)), shape=[tour_size], replace=True)
        mask = jnp.zeros_like(fitnesses)
        mask = mask.at[indexes].set(1)
        fitness_values_for_selection = (fitnesses + jnp.min(fitnesses) + 1) * mask
        best_genome = genomes.at[jnp.argmax(fitness_values_for_selection)].get()
        return best_genome

    sample_keys = jax.random.split(rnd_key, n_samples)
    partial_single_tournament = partial(_tournament, genomes=genomes, fitnesses=fitness_values, tour_size=tour_size)
    vmap_tournament = vmap(partial_single_tournament)
    return vmap_tournament(sample_key=sample_keys)
