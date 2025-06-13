from typing import Mapping

import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire, get_cells_indices


def distill_archive(
        repertoire: MapElitesRepertoire,
        target_centroids: jnp.ndarray,
        fitness_threshold: float = -jnp.inf,
) -> MapElitesRepertoire:
    ids_to_keep = jnp.where(repertoire.fitnesses > fitness_threshold)
    processed_repertoire = MapElitesRepertoire.init(
        genotypes=jax.tree_util.tree_map(
            lambda x: x[ids_to_keep],
            repertoire.genotypes
        ),
        fitnesses=repertoire.fitnesses[ids_to_keep],
        descriptors=repertoire.descriptors[ids_to_keep],
        centroids=repertoire.centroids[ids_to_keep],
    )
    distilled_ids = get_cells_indices(target_centroids, processed_repertoire.centroids)
    return MapElitesRepertoire.init(
        genotypes=jax.tree_util.tree_map(
            lambda x: x[distilled_ids],
            processed_repertoire.genotypes
        ),
        fitnesses=processed_repertoire.fitnesses[distilled_ids],
        descriptors=processed_repertoire.descriptors[distilled_ids],
        centroids=target_centroids
    )


def map_output_to_nn_params(
        raw_output: jnp.ndarray,
        repertoire: MapElitesRepertoire
) -> Mapping:
    # output will be in [-1, 1]
    # rescale it to fit the centroids shape
    # TODO: this is for beta testing, should be made much safe

    scaling_factor = jnp.max(repertoire.centroids)
    scaled_output = raw_output * scaling_factor
    cell_indices = get_cells_indices(jnp.expand_dims(scaled_output, 0), repertoire.centroids).astype(int)[0]
    params = jax.tree_util.tree_map(
        lambda x: x[cell_indices],
        repertoire.genotypes
    )
    return params
