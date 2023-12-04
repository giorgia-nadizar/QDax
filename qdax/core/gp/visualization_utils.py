from typing import Dict

import jax.numpy as jnp

from qdax.core.gp.graph_utils import compute_cgp_active_graph, compute_lgp_coding_lines
from qdax.core.gp.functions import available_functions


def expression_from_genome(genome: jnp.ndarray, config: Dict) -> str:
    if config["solver"] == "cgp":
        return _cgp_expression_from_genome(genome, config)
    if config["solver"] == "lgp":
        return _lgp_expression_from_genome(genome, config)
    raise ValueError("Solver must be either cgp or lgp.")


def program_from_genome(genome: jnp.ndarray, config: Dict, active_only: bool = True) -> str:
    if config["solver"] == "cgp":
        return _cgp_program_from_genome(genome, config, active_only)
    if config["solver"] == "lgp":
        return _lgp_program_from_genome(genome, config, active_only)
    raise ValueError("Solver must be either cgp or lgp.")


def _cgp_expression_from_genome(genome: jnp.ndarray, config: Dict) -> str:
    n_in, n_out = config["n_in"], config["n_out"]
    n_nodes = config["n_nodes"]
    x_genes, y_genes, f_genes, out_genes = jnp.split(genome, jnp.asarray([n_nodes, 2 * n_nodes, 3 * n_nodes]))
    target = ""
    for i, out in enumerate(out_genes):
        target = target + f"o{i} = {_replace_cgp_expression(x_genes, y_genes, f_genes, n_in, out)}\n"
    return target


def _replace_cgp_expression(
    x_genes: jnp.ndarray,
    y_genes: jnp.ndarray,
    f_genes: jnp.ndarray,
    n_in: int,
    idx: int
) -> str:
    if idx < n_in:
        return f"i{idx}"
    functions = list(available_functions.values())
    gene_idx = idx - n_in
    function = functions[f_genes[gene_idx]]
    if function.arity == 1:
        return f"{function.symbol}({_replace_cgp_expression(x_genes, y_genes, f_genes, n_in, int(x_genes[gene_idx]))})"
    else:
        return f"({_replace_cgp_expression(x_genes, y_genes, f_genes, n_in, int(x_genes[gene_idx]))}" \
               f"{function.symbol}{_replace_cgp_expression(x_genes, y_genes, f_genes, n_in, int(y_genes[gene_idx]))})"


def _lgp_expression_from_genome(genome: jnp.ndarray, config: Dict) -> str:
    lhs_genes, x_genes, y_genes, f_genes = jnp.split(genome, 4)
    lhs_genes += config["n_in"]
    target = ""
    for output_id in range(config["n_out"]):
        register_id = config["n_registers"] - config["n_out"] + output_id
        expr = _replace_lgp_expr(lhs_genes, x_genes, y_genes, f_genes, register_id, len(lhs_genes), config['n_in'])
        target = target + f"o{output_id} = {expr}\n"
    return target


def _replace_lgp_expr(
    lhs_genes: jnp.ndarray,
    x_genes: jnp.ndarray,
    y_genes: jnp.ndarray,
    f_genes: jnp.ndarray,
    register_number: int,
    max_row_id: int,
    n_in: int
) -> str:
    for row_id in range(max_row_id - 1, -1, -1):
        if int(lhs_genes[row_id]) == register_number:
            function = list(available_functions.values())[f_genes[row_id]]
            if function.arity == 1:
                expr = _replace_lgp_expr(lhs_genes, x_genes, y_genes, f_genes, int(x_genes[row_id]), row_id, n_in)
                return f"{function.symbol}({expr})"
            else:
                expr1 = _replace_lgp_expr(lhs_genes, x_genes, y_genes, f_genes, int(x_genes[row_id]), row_id, n_in)
                expr2 = _replace_lgp_expr(lhs_genes, x_genes, y_genes, f_genes, int(y_genes[row_id]), row_id, n_in)
                return f"({expr1}{function.symbol}{expr2})"
    return f"i{register_number}" if register_number < n_in else "0"


def _cgp_program_from_genome(genome: jnp.ndarray, config: Dict, active_only: bool = True) -> str:
    n_in, n_out, n_nodes = config["n_in"], config["n_out"], config["n_nodes"]
    x_genes, y_genes, f_genes, out_genes = jnp.split(genome, jnp.asarray([n_nodes, 2 * n_nodes, 3 * n_nodes]))
    active = compute_cgp_active_graph(genome, config) if active_only else jnp.ones(n_nodes + n_in)
    functions = list(available_functions.values())
    text_function = f"def program(inputs, buffer):\n" \
                    f"  buffer[{list(range(n_in))}] = inputs\n"

    # execution
    for buffer_idx in range(n_in, len(active)):
        if active[buffer_idx]:
            idx = buffer_idx - n_in
            function = functions[f_genes[idx]]
            text_function += f"  buffer[{buffer_idx}] = {function.symbol}(buffer[{x_genes[idx]}]"
            if function.arity > 1:
                text_function += f", buffer[{y_genes[idx]}]"
            text_function += ")\n"

    # output selection
    text_function += f"  outputs = buffer[{out_genes}]\n"
    return text_function


def _lgp_program_from_genome(genome: jnp.ndarray, config: Dict, active_only: bool = True) -> str:
    lhs_genes, x_genes, y_genes, f_genes = jnp.split(genome, 4)
    lhs_genes += config["n_in"]
    functions = list(available_functions.values())
    text_function = f"def program(inputs, r):\n" \
                    f"  r[{list(range(config['n_in']))}] = inputs\n"

    active = compute_lgp_coding_lines(genome, config) if active_only else jnp.ones(config["n_rows"])
    # execution
    for row_idx in range(config["n_rows"]):
        if active[row_idx]:
            function = functions[f_genes[row_idx]]
            text_function += f"  r[{lhs_genes[row_idx]}] = {function.symbol}(r[{x_genes[row_idx]}]"
            if function.arity > 1:
                text_function += f", r[{y_genes[row_idx]}]"
            text_function += ")\n"

    # output selection
    text_function += f"  outputs = r[-{config['n_out']}:]\n"
    return text_function
