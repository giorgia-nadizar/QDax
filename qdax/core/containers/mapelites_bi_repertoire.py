from __future__ import annotations

from functools import partial
from typing import Callable, Tuple, Optional

import flax
import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.types import RNGKey, Genotype, Descriptor, Fitness, ExtraScores, Centroid, Mask


class MapElitesBiRepertoire(flax.struct.PyTreeNode):
    repertoire1: MapElitesRepertoire
    repertoire2: MapElitesRepertoire
    descriptors_indexes1: jnp.ndarray
    descriptors_indexes2: jnp.ndarray
    sampling_mask: Mask

    def save(self, path: str = "/.") -> None:
        self.repertoire1.save(f"{path}r1_")
        self.repertoire2.save(f"{path}r2_")
        jnp.save(path + "descriptors_indexes1.npy", self.descriptors_indexes1)
        jnp.save(path + "descriptors_indexes2.npy", self.descriptors_indexes2)

    @classmethod
    def load(cls, reconstruction_fn: Callable, path: str = "./") -> MapElitesBiRepertoire:
        repertoire1 = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path}r1_")
        repertoire2 = MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=f"{path}r2_")
        descriptors_indexes1 = jnp.load(path + "descriptors_indexes1.npy")
        descriptors_indexes2 = jnp.load(path + "descriptors_indexes2.npy")
        sampling_mask = jnp.ones(len(repertoire1.centroids) + len(repertoire2.centroids))
        return cls(repertoire1, repertoire2, descriptors_indexes1, descriptors_indexes2, sampling_mask)

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample(self, random_key: RNGKey, num_samples: int) -> Tuple[Genotype, RNGKey]:
        genotypes = jnp.concatenate([self.repertoire1.genotypes, self.repertoire2.genotypes])
        fitnesses = jnp.concatenate([self.repertoire1.fitnesses, self.repertoire2.fitnesses])
        _, unique_genotypes_indexes = jnp.unique(genotypes, axis=0, return_index=True, size=len(genotypes),
                                                 fill_value=jnp.zeros_like(genotypes[0]))
        filled_genotypes_mask = fitnesses != -jnp.inf
        unique_genotypes_mask = jnp.isin(jnp.arange(len(genotypes)), unique_genotypes_indexes)
        candidate_genotypes_mask = filled_genotypes_mask & unique_genotypes_mask & self.sampling_mask.astype(int)
        p = candidate_genotypes_mask.astype(int)
        p = p / jnp.sum(p)
        random_key, subkey = jax.random.split(random_key)
        samples = jax.tree_util.tree_map(
            lambda x: jax.random.choice(subkey, x, shape=(num_samples,), p=p),
            genotypes,
        )
        return samples, random_key

    # Note: sampling id semantics
    # 0 = sample from both
    # 1 = sample from first
    # 2 = sample from second
    @jax.jit
    def update_sampling_mask(self, sampling_id: int) -> MapElitesBiRepertoire:
        sampling_mask1 = jnp.ones(len(self.repertoire1.centroids)) * (sampling_id < 2)
        sampling_mask2 = jnp.ones(len(self.repertoire2.centroids)) * (sampling_id % 2 == 0)
        sampling_mask = jnp.concatenate([sampling_mask1, sampling_mask2])

        return MapElitesBiRepertoire(
            self.repertoire1,
            self.repertoire2,
            self.descriptors_indexes1,
            self.descriptors_indexes2,
            sampling_mask
        )

    @jax.jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: Optional[ExtraScores] = None,
    ) -> Tuple[MapElitesBiRepertoire, bool]:
        descriptors1 = batch_of_descriptors.take(self.descriptors_indexes1, axis=1)
        descriptors2 = batch_of_descriptors.take(self.descriptors_indexes2, axis=1)
        new_repertoire1, addition_condition1 = self.repertoire1.add_and_track(
            batch_of_genotypes,
            descriptors1,
            batch_of_fitnesses,
            batch_of_extra_scores
        )
        new_repertoire2, addition_condition2 = self.repertoire2.add_and_track(
            batch_of_genotypes,
            descriptors2,
            batch_of_fitnesses,
            batch_of_extra_scores
        )
        new_double_repertoire = MapElitesBiRepertoire(
            repertoire1=new_repertoire1,
            repertoire2=new_repertoire2,
            descriptors_indexes1=self.descriptors_indexes1,
            descriptors_indexes2=self.descriptors_indexes2,
            sampling_mask=self.sampling_mask
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
    ) -> MapElitesBiRepertoire:
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
        sampling_mask = jnp.ones(len(repertoire1.centroids) + len(repertoire2.centroids))
        return MapElitesBiRepertoire(
            repertoire1=repertoire1,
            repertoire2=repertoire2,
            descriptors_indexes1=descriptors_indexes1,
            descriptors_indexes2=descriptors_indexes2,
            sampling_mask=sampling_mask
        )
