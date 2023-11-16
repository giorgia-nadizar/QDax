from __future__ import annotations

from functools import partial
from typing import Callable, Tuple, Optional

import flax
import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.types import RNGKey, Genotype, Descriptor, Fitness, ExtraScores, Centroid


class DoubleMapElitesRepertoire(flax.struct.PyTreeNode):
    repertoire1: MapElitesRepertoire
    repertoire2: MapElitesRepertoire
    descriptors_indexes1: jnp.ndarray
    descriptors_indexes2: jnp.ndarray

    def save(self, path: str = "/.") -> None:
        self.repertoire1.save(f"{path}r1_")
        self.repertoire2.save(f"{path}r2_")
        jnp.save(path + "descriptors_indexes1.npy", self.descriptors_indexes1)
        jnp.save(path + "descriptors_indexes2.npy", self.descriptors_indexes2)

    @classmethod
    def load(cls, reconstruction_fn: Callable, path: str = "./") -> DoubleMapElitesRepertoire:
        repertoire1 = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path}r1_")
        repertoire2 = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path}r2_")
        descriptors_indexes1 = jnp.load(path + "descriptors_indexes1.npy")
        descriptors_indexes2 = jnp.load(path + "descriptors_indexes2.npy")
        return cls(repertoire1, repertoire2, descriptors_indexes1, descriptors_indexes2)

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample(self, random_key: RNGKey, num_samples: int) -> Tuple[Genotype, RNGKey]:
        genotypes = jnp.concatenate([self.repertoire1.genotypes, self.repertoire2.genotypes])
        fitnesses = jnp.concatenate([self.repertoire1.fitnesses, self.repertoire2.fitnesses])
        ids_to_keep = fitnesses != -jnp.inf
        genotypes_to_keep = genotypes.at[ids_to_keep].get()
        unique_genotypes = jnp.unique(genotypes_to_keep, axis=0)
        random_key, subkey = jax.random.split(random_key)
        samples = jax.random.choice(subkey, unique_genotypes, shape=(num_samples,))
        return samples, random_key

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample_first(self, random_key: RNGKey, num_samples: int) -> Tuple[Genotype, RNGKey]:
        return self.repertoire1.sample(random_key, num_samples)

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample_second(self, random_key: RNGKey, num_samples: int) -> Tuple[Genotype, RNGKey]:
        return self.repertoire2.sample(random_key, num_samples)

    @jax.jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: Optional[ExtraScores] = None,
    ) -> Tuple[DoubleMapElitesRepertoire, bool]:
        descriptors1 = batch_of_descriptors.take(self.descriptors_indexes1, axis=1)
        descriptors2 = batch_of_descriptors.take(self.descriptors_indexes2, axis=1)
        new_repertoire1, addition_condition1 = self.repertoire1.add(
            batch_of_genotypes,
            descriptors1,
            batch_of_fitnesses,
            batch_of_extra_scores
        )
        new_repertoire2, addition_condition2 = self.repertoire2.add(
            batch_of_genotypes,
            descriptors2,
            batch_of_fitnesses,
            batch_of_extra_scores
        )
        new_double_repertoire = DoubleMapElitesRepertoire(
            repertoire1=new_repertoire1,
            repertoire2=new_repertoire2,
            descriptors_indexes1=self.descriptors_indexes1,
            descriptors_indexes2=self.descriptors_indexes2
        )
        addition_condition = addition_condition1 + addition_condition2
        return new_double_repertoire, addition_condition

    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        descriptors_indexes1: jnp.ndarray,
        descriptors_indexes2: jnp.ndarray,
        centroids1: Centroid,
        centroids2: Centroid,
        extra_scores: Optional[ExtraScores] = None,
    ) -> DoubleMapElitesRepertoire:
        descriptors1 = descriptors.take(descriptors_indexes1, axis=1)
        descriptors2 = descriptors.take(descriptors_indexes2, axis=1)
        repertoire1 = MapElitesRepertoire.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors1,
            centroids=centroids1,
            extra_scores=extra_scores
        )
        repertoire2 = MapElitesRepertoire.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors2,
            centroids=centroids2,
            extra_scores=extra_scores
        )
        return DoubleMapElitesRepertoire(
            repertoire1=repertoire1,
            repertoire2=repertoire2,
            descriptors_indexes1=descriptors_indexes1,
            descriptors_indexes2=descriptors_indexes2
        )
