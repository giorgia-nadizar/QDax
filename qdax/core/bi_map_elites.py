from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple, Any

import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
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


class DoubleRepME(MAPElites):
    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
        n_descriptors: int,
        extra_descriptors_indexes: jnp.ndarray = jnp.asarray([]),
        extra_metrics_functions: Callable[[MapElitesRepertoire], Metrics] = lambda x: {}
    ) -> None:
        super(DoubleRepME, self).__init__(scoring_function, emitter, metrics_function)
        self._main_descriptors_indexes = jnp.setdiff1d(jnp.arange(n_descriptors), extra_descriptors_indexes)
        self._extra_descriptors_indexes = extra_descriptors_indexes
        self._extra_metrics_functions = extra_metrics_functions

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        init_genotypes: Genotype,
        main_centroids: Centroid,
        extra_centroids: Centroid,
        random_key: RNGKey,
    ) -> Tuple[Tuple[MapElitesRepertoire, MapElitesRepertoire], Optional[EmitterState], RNGKey]:
        # score initial genotypes
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            init_genotypes, random_key
        )

        # split descriptors into main and extra
        main_descriptors = descriptors.take(self._main_descriptors_indexes, axis=1)
        extra_descriptors = descriptors.take(self._extra_descriptors_indexes, axis=1)

        # init the repertoire
        main_repertoire = MapElitesRepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=main_descriptors,
            centroids=main_centroids,
            extra_scores=extra_scores,
        )
        extra_repertoire = MapElitesRepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=extra_descriptors,
            centroids=extra_centroids,
            extra_scores=extra_scores,
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=init_genotypes, random_key=random_key
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=main_repertoire,
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=main_descriptors,
            extra_scores=extra_scores,
        )

        return (main_repertoire, extra_repertoire), emitter_state, random_key

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoires: Tuple[MapElitesRepertoire, MapElitesRepertoire],
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[Tuple[MapElitesRepertoire, MapElitesRepertoire], Optional[EmitterState], Metrics, RNGKey]:
        # generate offsprings with the emitter
        main_repertoire, extra_repertoire = repertoires
        genotypes, random_key = self._emitter.emit(
            main_repertoire, emitter_state, random_key
        )
        # scores the offspring
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )

        # split descriptors into main and extra
        main_descriptors = descriptors.take(self._main_descriptors_indexes, axis=1)
        extra_descriptors = descriptors.take(self._extra_descriptors_indexes, axis=1)

        # add genotypes in the repertoire
        main_repertoire = main_repertoire.add(genotypes, main_descriptors, fitnesses, extra_scores)
        extra_repertoire = extra_repertoire.add(genotypes, extra_descriptors, fitnesses, extra_scores)

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=main_repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=main_descriptors,
            extra_scores=extra_scores,
        )

        # update the metrics
        main_metrics = self._metrics_function(main_repertoire)
        extra_metrics = self._extra_metrics_functions(extra_repertoire)
        # extra_metrics = jax.tree_map(lambda k: "extra_" + k, extra_metrics)
        metrics = {**main_metrics, **extra_metrics}

        return (main_repertoire, extra_repertoire), emitter_state, metrics, random_key

    @partial(jax.jit, static_argnames=("self",))
    def scan_update(
        self,
        carry: Tuple[Tuple[MapElitesRepertoire, MapElitesRepertoire], Optional[EmitterState], RNGKey],
        unused: Any,
    ) -> Tuple[Tuple[Tuple[MapElitesRepertoire, MapElitesRepertoire], Optional[EmitterState], RNGKey], Metrics]:
        """Rewrites the update function in a way that makes it compatible with the
        jax.lax.scan primitive.

        Args:
            carry: a tuple containing the repertoire, the emitter state and a
                random key.
            unused: unused element, necessary to respect jax.lax.scan API.

        Returns:
            The updated repertoire and emitter state, with a new random key and metrics.
        """
        repertoires, emitter_state, random_key = carry
        (repertoires, emitter_state, metrics, random_key,) = self.update(
            repertoires,
            emitter_state,
            random_key,
        )

        return (repertoires, emitter_state, random_key), metrics
