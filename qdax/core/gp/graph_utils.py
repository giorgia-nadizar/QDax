from functools import partial
from typing import Dict, Tuple, Callable, List

from jax import numpy as jnp, vmap
from jax.lax import fori_loop

from qdax.core.gp.functions import function_arities, available_functions, arithmetic_functions, logical_functions, \
    trigonometric_functions
from qdax.types import Descriptor, Genotype


def get_graph_descriptor_extractor(config: Dict) -> Tuple[Callable[[Genotype], Descriptor], int]:
    indexes = {
        "complexity": [0],
        "inputs_usage": [1],
        "function_arities": [2, 3],
        "function_types": [4, 5, 6],
        "function_arities_fraction": [7]
    }
    list_of_descriptors = config["graph_descriptors"] if isinstance(config["graph_descriptors"], list) \
        else [config["graph_descriptors"]]
    descr_indexes = jnp.asarray([idx for desc_name in list_of_descriptors for idx in indexes[desc_name]])
    if config["solver"] == "cgp":
        single_genome_descriptor_function = partial(compute_cgp_descriptors, config=config,
                                                    descriptors_indexes=descr_indexes)
    elif config["solver"] == "lgp":
        single_genome_descriptor_function = partial(compute_lgp_descriptors, config=config,
                                                    descriptors_indexes=descr_indexes)
    else:
        raise ValueError("Solver must be either cgp or lgp.")
    return vmap(single_genome_descriptor_function), len(descr_indexes)


def get_complexity_fn(config: Dict) -> Callable[[jnp.ndarray], float]:
    if config["solver"] == "cgp":
        compute_active_fn = compute_cgp_active_graph
    elif config["solver"] == "lgp":
        compute_active_fn = compute_lgp_coding_lines
    else:
        raise ValueError("Solver must be either cgp or lgp.")
    return partial(_compute_complexity, config=config, compute_active_fn=compute_active_fn)


def compute_cgp_descriptors(genome: jnp.ndarray, config: Dict, descriptors_indexes: jnp.ndarray) -> jnp.ndarray:
    active = compute_cgp_active_graph(genome, config)
    active_fraction = jnp.sum(active[config["n_in"]:]) / config["n_nodes"]
    inputs_fraction = jnp.sum(active[:config["n_in"]]) / config["n_in"]
    functions_count = _cgp_count_functions(active, genome, config)
    functions_descriptor = _functions_descriptor(functions_count, config["n_nodes"])
    descriptors = jnp.concatenate([jnp.asarray([active_fraction, inputs_fraction]), functions_descriptor])
    return jnp.take(descriptors, descriptors_indexes)


def compute_lgp_descriptors(genome: jnp.ndarray, config: Dict, descriptors_indexes: jnp.ndarray) -> jnp.ndarray:
    active = compute_lgp_coding_lines(genome, config)
    active_fraction = jnp.sum(active) / len(active)
    inputs_fraction = jnp.sum(_lgp_inputs_usage(active, genome, config)) / config["n_in"]
    functions_count = _lgp_count_functions(active, genome)
    functions_descriptor = _functions_descriptor(functions_count, len(active))
    descriptors = jnp.concatenate([jnp.asarray([active_fraction, inputs_fraction]), functions_descriptor])
    return jnp.take(descriptors, descriptors_indexes)


def compute_cgp_active_graph(
        genome: jnp.ndarray,
        config: Dict
) -> jnp.ndarray:
    n_nodes, n_out, n_in = config["n_nodes"], config["n_out"], config["n_in"]
    x_genes, y_genes, f_genes, out_genes = jnp.split(genome, [n_nodes, 2 * n_nodes, 3 * n_nodes])
    active = jnp.zeros(n_in + n_nodes)
    active = active.at[out_genes].set(1)

    active, _, _, _, _ = fori_loop(
        lower=0,
        upper=n_nodes,
        body_fun=_compute_active_graph,
        init_val=(active, x_genes, y_genes, f_genes, n_in),
    )
    return active


def compute_lgp_coding_lines(
        genome: jnp.ndarray,
        config: Dict
) -> jnp.ndarray:
    n_rows, n_registers, n_out, n_in = config["n_rows"], config["n_registers"], config["n_out"], config["n_in"]
    lhs_genes, x_genes, y_genes, f_genes = jnp.split(genome, 4)
    lhs_genes = lhs_genes + n_in
    registers_mask = jnp.where(jnp.arange(n_registers) >= (n_registers - n_out), 1, 0)
    active = jnp.zeros(n_rows)
    active, _, _, _, _, _ = fori_loop(
        lower=0,
        upper=n_rows,
        body_fun=_set_used_lines,
        init_val=(active, registers_mask, lhs_genes, x_genes, y_genes, f_genes)
    )
    return active


def _cgp_count_functions(active: jnp.ndarray, genome: jnp.ndarray, config: Dict) -> jnp.ndarray:
    _, _, f_genes, _ = jnp.split(genome, [config["n_nodes"], 2 * config["n_nodes"], 3 * config["n_nodes"]])

    def _count_cgp_functions(
            idx: int,
            carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        active, f_genes, f_counter = carry
        n_in = len(active) - len(f_genes)
        f_id = f_genes.at[idx].get()
        f_counter = f_counter.at[f_id].set(f_counter.at[f_id].get() + active.at[idx + n_in].get())
        return active, f_genes, f_counter

    _, _, functions_count = fori_loop(
        lower=0,
        upper=len(f_genes),
        body_fun=_count_cgp_functions,
        init_val=(active, f_genes, jnp.zeros(len(available_functions)))
    )
    return functions_count


def _lgp_count_functions(active: jnp.ndarray, genome: jnp.ndarray) -> jnp.ndarray:
    _, _, _, f_genes = jnp.split(genome, 4)

    def _count_lgp_functions(
            idx: int,
            carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        active, f_genes, f_counter = carry
        f_id = f_genes.at[idx].get()
        f_counter = f_counter.at[f_id].set(f_counter.at[f_id].get() + active.at[idx].get())
        return active, f_genes, f_counter

    _, _, functions_count = fori_loop(
        lower=0,
        upper=len(f_genes),
        body_fun=_count_lgp_functions,
        init_val=(active, f_genes, jnp.zeros(len(available_functions)))
    )
    return functions_count


def _functions_descriptor(functions_count: jnp.ndarray, max_n_functions: int) -> jnp.ndarray:
    one_arity_total = jnp.sum(jnp.where(function_arities == 1, functions_count, 0))
    two_arity_total = jnp.sum(jnp.where(function_arities == 2, functions_count, 0))

    arithmetic_total = jnp.sum(jnp.take(functions_count, arithmetic_functions))
    logical_total = jnp.sum(jnp.take(functions_count, logical_functions))
    trigonometric_total = jnp.sum(jnp.take(functions_count, trigonometric_functions))
    function_totals = jnp.asarray(
        [one_arity_total, two_arity_total, arithmetic_total, logical_total, trigonometric_total])

    arity_fraction = one_arity_total / (one_arity_total + two_arity_total)

    return jnp.concatenate([function_totals / max_n_functions, jnp.asarray([arity_fraction])])


def _lgp_inputs_usage(
        active: jnp.ndarray,
        genome: jnp.ndarray,
        config: Dict
) -> float:
    lhs_genes, x_genes, y_genes, f_genes = jnp.split(genome, 4)
    lhs_genes = lhs_genes + config["n_in"]
    used_registers, _, _, _, _, _, _ = fori_loop(
        lower=0,
        upper=config["n_rows"],
        body_fun=_compute_lgp_used_inputs,
        init_val=(
            jnp.zeros(config["n_registers"]), jnp.zeros(config["n_registers"]), active, lhs_genes, x_genes,
            y_genes, f_genes,
        ),
    )
    return used_registers[:config["n_in"]]


def _compute_lgp_used_inputs(
        line_id: int,
        carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    used_regs, rewr_regs, active_lines, lhs_genes, x_genes, y_genes, f_genes = carry
    active_line = active_lines.at[line_id].get()
    lhs_pos = lhs_genes.at[line_id].get()
    x_pos = x_genes.at[line_id].get()
    rewr_x = rewr_regs.at[x_pos].get()
    y_pos = y_genes.at[line_id].get()
    rewr_y = rewr_regs.at[y_pos].get()
    arity = function_arities.at[f_genes[line_id]].get()
    # set the used registers
    used_regs = used_regs.at[x_pos].set(jnp.ceil((used_regs.at[x_pos].get() + active_line) / 2) * (1 - rewr_x))
    used_regs = used_regs.at[y_pos].set(
        jnp.ceil((used_regs.at[y_pos].get() + (active_line * (arity == 2))) / 2) * (1 - rewr_y))
    # set the registers which are written before content usage
    used_target = used_regs.at[lhs_pos].get()
    rewr_regs = rewr_regs.at[lhs_pos].set(jnp.ceil((rewr_regs.at[lhs_pos].get() + (1 - used_target)) / 2))
    return used_regs, rewr_regs, active_lines, lhs_genes, x_genes, y_genes, f_genes


def _compute_complexity(
        genome: jnp.ndarray,
        config: Dict,
        compute_active_fn: Callable[[jnp.ndarray, Dict], jnp.ndarray]
) -> float:
    active = compute_active_fn(genome, config)
    return jnp.sum(active) / len(active)


def _compute_active_graph(
        opposite_idx: int,
        carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    active, x_genes, y_genes, f_genes, n_in = carry
    idx = len(active) - opposite_idx - 1
    x_idx = x_genes.at[idx - n_in].get().astype(int)
    y_idx = y_genes.at[idx - n_in].get().astype(int)
    arity = function_arities.at[f_genes[idx - n_in]].get()
    active = active.at[x_idx].set(jnp.ceil((active.at[x_idx].get() + active.at[idx].get()) / 2))
    active = active.at[y_idx].set(jnp.ceil((active.at[y_idx].get() + (active.at[idx].get() * (arity == 2))) / 2))
    return active, x_genes, y_genes, f_genes, n_in


def _set_used_lines(
        opposite_idx: int,
        carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    active, regs_mask, lhs_genes, x_genes, y_genes, f_genes = carry
    row_idx = len(active) - opposite_idx - 1
    line_use = regs_mask.at[lhs_genes.at[row_idx].get()].get()
    active = active.at[row_idx].set(line_use)

    x_reg = x_genes.at[row_idx].get()
    y_reg = y_genes.at[row_idx].get()
    arity = function_arities.at[f_genes[row_idx]].get()
    regs_mask = regs_mask.at[row_idx].set(0)
    regs_mask = regs_mask.at[x_reg].set(jnp.ceil((line_use + regs_mask.at[x_reg].get()) / 2))
    regs_mask = regs_mask.at[y_reg].set(jnp.ceil((line_use * (arity == 2) + regs_mask.at[y_reg].get()) / 2))

    return active, regs_mask, lhs_genes, x_genes, y_genes, f_genes
