import jax
import jax.numpy as jnp
import pytest

from qdax.core.graphs.cartesian_genetic_programming import CGP, cgp_mutation


def test_genome_bounds() -> None:
    """Test that a CGP genome has all elements in the correct bounds.
    Tests both at initialization and after mutation.
    """
    # define genome structure
    cgp = CGP(
        n_inputs=2,
        n_outputs=1,
        n_nodes=5
    )
    key = jax.random.key(42)

    # define expected bounds
    connections_bounds = jnp.arange(
        start=cgp.n_inputs + len(cgp.input_constants),
        stop=cgp.n_inputs + len(cgp.input_constants) + cgp.n_nodes
    )
    functions_bound = len(cgp.function_set)
    outputs_bound = cgp.n_inputs + len(cgp.input_constants) + cgp.n_nodes

    # init genome
    key, init_key = jax.random.split(key)
    initial_cgp_genome = cgp.init(init_key)

    # check if bounds are respected at initialization
    pytest.assume(jnp.all(initial_cgp_genome["params"]["x_connections_genes"] < connections_bounds))
    pytest.assume(jnp.all(initial_cgp_genome["params"]["y_connections_genes"] < connections_bounds))
    pytest.assume(jnp.all(initial_cgp_genome["params"]["functions_genes"] < functions_bound))
    pytest.assume(jnp.all(initial_cgp_genome["params"]["output_connections_genes"] < outputs_bound))

    # mutate genome
    key, mut_key = jax.random.split(key)
    mutated_cgp_genome = cgp_mutation(
        genotype=initial_cgp_genome,
        rnd_key=mut_key,
        cgp=cgp
    )

    # check if bounds are respected after mutation
    pytest.assume(jnp.all(mutated_cgp_genome["params"]["x_connections_genes"] < connections_bounds))
    pytest.assume(jnp.all(mutated_cgp_genome["params"]["y_connections_genes"] < connections_bounds))
    pytest.assume(jnp.all(mutated_cgp_genome["params"]["functions_genes"] < functions_bound))
    pytest.assume(jnp.all(mutated_cgp_genome["params"]["output_connections_genes"] < outputs_bound))


