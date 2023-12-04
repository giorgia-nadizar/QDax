"""Defines functions to retrieve metrics from training processes."""

from __future__ import annotations

import csv
from functools import partial
from typing import Dict, List

import jax
from jax import numpy as jnp

from qdax.core.containers.ga_repertoire import GARepertoire
from qdax.core.containers.mapelites_bi_repertoire import MapElitesBiRepertoire
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire, get_cells_indices
from qdax.core.containers.mome_repertoire import MOMERepertoire
from qdax.core.containers.nsga2_repertoire import NSGA2Repertoire
from qdax.types import Metrics
from qdax.utils.pareto_front import compute_hypervolume


class CSVLogger:
    """Logger to save metrics of an experiment in a csv file
    during the training process.
    """

    def __init__(self, filename: str, header: List) -> None:
        """Create the csv logger, create a file and write the
        header.

        Args:
            filename: path to which the file will be saved.
            header: header of the csv file.
        """
        self._filename = filename
        self._header = header
        with open(self._filename, "w") as file:
            writer = csv.DictWriter(file, fieldnames=self._header)
            # write the header
            writer.writeheader()

    def log(self, metrics: Dict[str, float]) -> None:
        """Log new metrics to the csv file.

        Args:
            metrics: A dictionary containing the metrics that
                need to be saved.
        """
        with open(self._filename, "a") as file:
            writer = csv.DictWriter(file, fieldnames=self._header)
            # write new metrics in a raw
            writer.writerow(metrics)


def default_ga_metrics(
    repertoire: GARepertoire,
) -> Metrics:
    """Compute the usual GA metrics that one can retrieve
    from a GA repertoire.

    Args:
        repertoire: a GA repertoire

    Returns:
        a dictionary containing the max fitness of the
            repertoire.
    """

    # get metrics
    max_fitness = jnp.max(repertoire.fitnesses, axis=0)

    return {
        "max_fitness": max_fitness,
    }


def default_nsga2_metrics(
    repertoire: NSGA2Repertoire,
) -> Metrics:
    """Compute the usual NSGA-II metrics that one can retrieve
    from a NSGA-II repertoire.

    Args:
        repertoire: a NSGA-II repertoire

    Returns:
        a dictionary containing the max fitness of the
            repertoire and the size of the pareto front.
    """

    # get metrics
    max_fitness = jnp.max(repertoire.fitnesses, axis=0)
    pareto_front_size = jnp.count_nonzero(repertoire.pareto_front_mask)
    pareto_fitnesses = repertoire.fitnesses * jnp.transpose(jnp.array([repertoire.pareto_front_mask, ] * 2))

    return {
        "max_fitness": max_fitness,
        "pareto_front_size": pareto_front_size,
        "pareto_fitnesses": pareto_fitnesses
    }


def default_qd_metrics(repertoire: MapElitesRepertoire, qd_offset: float) -> Metrics:
    """Compute the usual QD metrics that one can retrieve
    from a MAP Elites repertoire.

    Args:
        repertoire: a MAP-Elites repertoire
        qd_offset: an offset used to ensure that the QD score
            will be positive and increasing with the number
            of individuals.

    Returns:
        a dictionary containing the QD score (sum of fitnesses
            modified to be all positive), the max fitness of the
            repertoire, the coverage (number of niche filled in
            the repertoire).
    """

    # get metrics
    repertoire_empty = repertoire.fitnesses == -jnp.inf
    qd_score = jnp.sum(repertoire.fitnesses, where=~repertoire_empty)
    qd_score += qd_offset * jnp.sum(1.0 - repertoire_empty)
    coverage = 100 * jnp.mean(1.0 - repertoire_empty)
    max_fitness = jnp.max(repertoire.fitnesses)

    return {"qd_score": qd_score, "max_fitness": max_fitness, "coverage": coverage}


def qd_metrics_with_bi_tracking(
    repertoire: MapElitesRepertoire,
    qd_offset: float,
    centroids1: jnp.ndarray,
    centroids2: jnp.ndarray,
    descriptors_indexes1: jnp.ndarray,
    descriptors_indexes2: jnp.ndarray
) -> Metrics:
    def _compute_coverage(full_descriptors: jnp.ndarray,
                          centroids: jnp.ndarray,
                          descriptors_indexes: jnp.ndarray
                          ) -> jnp.ndarray:
        descriptors = full_descriptors.take(descriptors_indexes, axis=1)
        indices = get_cells_indices(descriptors, centroids)
        binary_array = jnp.where(jnp.isin(jnp.arange(len(centroids)), indices), 1, 0)
        return 100 * jnp.sum(binary_array) / len(centroids)

    # get metrics
    coverage1 = _compute_coverage(repertoire.descriptors, centroids1, descriptors_indexes1)
    coverage2 = _compute_coverage(repertoire.descriptors, centroids2, descriptors_indexes2)

    metrics = default_qd_metrics(repertoire, qd_offset)
    metrics["coverage1"] = coverage1
    metrics["coverage2"] = coverage2

    return metrics


def default_biqd_metrics(bi_repertoire: MapElitesBiRepertoire, qd_offset: float) -> Metrics:
    qd_metrics1 = default_qd_metrics(bi_repertoire.repertoire1, qd_offset)
    qd_metrics2 = default_qd_metrics(bi_repertoire.repertoire2, qd_offset)

    return {
        "qd_score1": qd_metrics1["qd_score"],
        "coverage1": qd_metrics1["coverage"],
        "qd_score2": qd_metrics2["qd_score"],
        "coverage2": qd_metrics2["coverage"],
        "max_fitness": qd_metrics1["max_fitness"],
    }


def default_moqd_metrics(
    repertoire: MOMERepertoire, reference_point: jnp.ndarray
) -> Metrics:
    """Compute the MOQD metric given a MOME repertoire and a reference point.

    Args:
        repertoire: a MOME repertoire.
        reference_point: the hypervolume of a pareto front has to be computed
            relatively to a reference point.

    Returns:
        A dictionary containing all the computed metrics.
    """
    repertoire_empty = repertoire.fitnesses == -jnp.inf
    repertoire_empty = jnp.all(repertoire_empty, axis=-1)
    repertoire_not_empty = ~repertoire_empty
    repertoire_not_empty = jnp.any(repertoire_not_empty, axis=-1)
    coverage = 100 * jnp.mean(repertoire_not_empty)
    hypervolume_function = partial(compute_hypervolume, reference_point=reference_point)
    moqd_scores = jax.vmap(hypervolume_function)(repertoire.fitnesses)
    moqd_scores = jnp.where(repertoire_not_empty, moqd_scores, -jnp.inf)
    max_hypervolume = jnp.max(moqd_scores)
    max_scores = jnp.max(repertoire.fitnesses, axis=(0, 1))
    max_sum_scores = jnp.max(jnp.sum(repertoire.fitnesses, axis=-1), axis=(0, 1))
    num_solutions = jnp.sum(~repertoire_empty)
    (
        pareto_front,
        _,
    ) = repertoire.compute_global_pareto_front()

    global_hypervolume = compute_hypervolume(
        pareto_front, reference_point=reference_point
    )
    metrics = {
        "moqd_score": moqd_scores,
        "max_hypervolume": max_hypervolume,
        "max_scores": max_scores,
        "max_sum_scores": max_sum_scores,
        "coverage": coverage,
        "number_solutions": num_solutions,
        "global_hypervolume": global_hypervolume,
    }

    return metrics
