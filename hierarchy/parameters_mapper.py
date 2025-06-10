from typing import Mapping

import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire, get_cells_indices


def map_output_to_nn_params(
        raw_output: jnp.ndarray,
        repertoire: MapElitesRepertoire
) -> Mapping:
    # output will be in [-1, 1]
    # rescale it to fit the centroids shape
    # TODO: this is for beta testing, should be made much safe

    scaling_factor = jnp.max(repertoire.centroids)
    scaled_output = raw_output * scaling_factor
    cell_indices = get_cells_indices(jnp.expand_dims(scaled_output, 0), repertoire.centroids).astype(int)
    params = jax.tree_util.tree_map(
        lambda x: x[cell_indices],
        repertoire.genotypes
    )
    return params
