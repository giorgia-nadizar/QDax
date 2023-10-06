from functools import partial
from typing import Dict, Tuple, Any, Callable

from jax import numpy as jnp
from jax.lax import scan

from qdax.core.gp.functions import function_arities


def get_complexity_fn(config: Dict) -> Callable[[jnp.ndarray], float]:
    if config["solver"] == "cgp":
        compute_active_fn = compute_cgp_active_graph
    elif config["solver"] == "lgp":
        compute_active_fn = compute_lgp_coding_lines
    else:
        raise ValueError("Solver must be either cgp or lgp.")
    return partial(compute_complexity, config=config, compute_active_fn=compute_active_fn)


def compute_complexity(
    genome: jnp.ndarray,
    config: Dict,
    compute_active_fn: Callable[[jnp.ndarray, Dict], jnp.ndarray]
) -> float:
    active = compute_active_fn(genome, config)
    return jnp.sum(active) / len(active)


def compute_cgp_active_graph(
    genome: jnp.ndarray,
    config: Dict
) -> jnp.ndarray:
    n_nodes, n_out, n_in = config["n_nodes"], config["n_out"], config["n_in"]
    x_genes, y_genes, f_genes, out_genes = jnp.split(genome, [n_nodes, 2 * n_nodes, 3 * n_nodes])
    active = jnp.where(jnp.arange(n_in + n_nodes) < n_in, 1, 0)
    active = active.at[out_genes].set(1)
    (active, _, _, _, _, _), _ = scan(
        _compute_active_graph,
        init=(active, x_genes, y_genes, f_genes, n_in, n_in + n_nodes - 1),
        xs=None,
        length=n_nodes
    )
    return active


def _compute_active_graph(
    carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, int],
    unused_args: Any
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, int], Any]:
    active, x_genes, y_genes, f_genes, n_in, idx = carry
    x_idx = x_genes.at[idx - n_in].get().astype(int)
    y_idx = y_genes.at[idx - n_in].get().astype(int)
    arity = function_arities.at[f_genes[idx - n_in]].get()
    active = active.at[x_idx].set(jnp.ceil((active.at[x_idx].get() + active.at[idx].get()) / 2))
    active = active.at[y_idx].set(jnp.ceil((active.at[y_idx].get() + (active.at[idx].get() * (arity == 2))) / 2))
    return (active, x_genes, y_genes, f_genes, n_in, idx - 1), None


def compute_lgp_coding_lines(
    genome: jnp.ndarray,
    config: Dict
) -> jnp.ndarray:
    n_rows, n_registers, n_out, n_in = config["n_rows"], config["n_registers"], config["n_out"], config["n_in"]
    lhs_genes, x_genes, y_genes, f_genes = jnp.split(genome, 4)
    lhs_genes = lhs_genes + n_in
    registers_mask = jnp.where(jnp.arange(n_registers) >= (n_registers - n_out), 1, 0)
    active = jnp.zeros(n_rows)
    (active, _, _, _, _, _, _), _ = scan(
        _set_used_lines,
        init=(active, registers_mask, lhs_genes, x_genes, y_genes, f_genes, n_rows - 1),
        xs=None,
        length=n_rows
    )
    return active


def _set_used_lines(
    carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int],
    unused_args: Any
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int], Any]:
    active, regs_mask, lhs_genes, x_genes, y_genes, f_genes, row_id = carry
    line_use = regs_mask.at[lhs_genes.at[row_id].get()].get()
    active = active.at[row_id].set(line_use)

    x_reg = x_genes.at[row_id].get()
    y_reg = y_genes.at[row_id].get()
    arity = function_arities.at[f_genes[row_id]].get()
    regs_mask = regs_mask.at[row_id].set(0)
    regs_mask = regs_mask.at[x_reg].set(jnp.ceil((line_use + regs_mask.at[x_reg].get()) / 2))
    regs_mask = regs_mask.at[y_reg].set(jnp.ceil((line_use * (arity == 2) + regs_mask.at[y_reg].get()) / 2))

    return (active, regs_mask, lhs_genes, x_genes, y_genes, f_genes, row_id - 1), None
