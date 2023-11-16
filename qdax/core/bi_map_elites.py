from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_bi_repertoire import MapElitesBiRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.map_elites import MAPElites
from qdax.types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)


def sampling_function(function_name: str) -> Callable[[MapElitesBiRepertoire], int]:
    if "0" in function_name or "both" in function_name:
        return lambda x: 0
    elif "1" in function_name:
        return lambda x: 1
    elif "2" in function_name:
        return lambda x: 2
    elif "most_covered" in function_name:
        def _most_covered_sampler(bi_repertoire: MapElitesBiRepertoire) -> int:
            repertoire_empty1 = bi_repertoire.repertoire1.fitnesses == -jnp.inf
            coverage1 = 100 * jnp.mean(1.0 - repertoire_empty1)
            repertoire_empty2 = bi_repertoire.repertoire2.fitnesses == -jnp.inf
            coverage2 = 100 * jnp.mean(1.0 - repertoire_empty2)
            return 1 + (coverage1 < coverage2)

        return _most_covered_sampler
    elif "least_covered" in function_name:
        def _least_covered_sampler(bi_repertoire: MapElitesBiRepertoire) -> int:
            repertoire_empty1 = bi_repertoire.repertoire1.fitnesses == -jnp.inf
            coverage1 = 100 * jnp.mean(1.0 - repertoire_empty1)
            repertoire_empty2 = bi_repertoire.repertoire2.fitnesses == -jnp.inf
            coverage2 = 100 * jnp.mean(1.0 - repertoire_empty2)
            return 1 + (coverage1 > coverage2)

        return _least_covered_sampler
    else:
        raise ValueError("Solver must be either cgp or lgp.")


class BiMAPElites(MAPElites):
    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesBiRepertoire], Metrics],
        descriptors_indexes1: jnp.ndarray,
        descriptors_indexes2: jnp.ndarray = jnp.asarray([]),
        sampling_id_function: Callable[[MapElitesBiRepertoire], int] = lambda x: 0

        # Note: sampling id semantics
        # 0 = sample from both
        # 1 = sample from first
        # 2 = sample from second
    ) -> None:
        super(BiMAPElites, self).__init__(scoring_function, emitter, metrics_function)
        self._descriptors_indexes1 = descriptors_indexes1
        self._descriptors_indexes2 = descriptors_indexes2
        self._sampling_id_function = sampling_id_function

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        init_genotypes: Genotype,
        centroids1: Centroid,
        centroids2: Centroid,
        random_key: RNGKey,
    ) -> Tuple[MapElitesBiRepertoire, Optional[EmitterState], RNGKey]:
        # score initial genotypes
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            init_genotypes, random_key
        )

        # init the bi-repertoire
        bi_repertoire = MapElitesBiRepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            descriptors_indexes1=self._descriptors_indexes1,
            descriptors_indexes2=self._descriptors_indexes2,
            centroids1=centroids1,
            centroids2=centroids2,
            extra_scores=extra_scores
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=init_genotypes, random_key=random_key
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=bi_repertoire,
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        return bi_repertoire, emitter_state, random_key

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        bi_repertoire: MapElitesBiRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[MapElitesBiRepertoire, Optional[EmitterState], Metrics, RNGKey]:
        sampling_id = self._sampling_id_function(bi_repertoire)
        bi_repertoire = bi_repertoire.update_sampling_mask(sampling_id)

        # generate offsprings with the emitter
        offspring, random_key = self._emitter.emit(
            bi_repertoire, emitter_state, random_key
        )
        # scores the offspring
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            offspring, random_key
        )

        # add genotypes in the repertoire
        bi_repertoire, _ = bi_repertoire.add(offspring, descriptors, fitnesses, extra_scores)

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=bi_repertoire,
            genotypes=offspring,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # update the metrics
        metrics = self._metrics_function(bi_repertoire)

        return bi_repertoire, emitter_state, metrics, random_key
