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


def test_known_genome_execution() -> None:
    """Test that a CGP genome behaves as expected.
    The chosen genome takes as outputs:
    - input 0
    - constant 0
    - input 0 + input 1
    - (input 0 + input 1) * input 1
    All outputs are wrapped by the tanh function.
        """
    # define genome structure
    cgp = CGP(
        n_inputs=2,
        n_outputs=3,
        n_nodes=5
    )
    cgp_genome = {
        "params": {
            "x_connections_genes": jnp.asarray([0, 0, 4, 0, 0]),
            "y_connections_genes": jnp.ones(cgp.n_nodes, dtype=jnp.int32),
            "functions_genes": jnp.asarray([0, 0, 2, 0, 0]),
            "output_connections_genes": jnp.asarray([0, 2, 4, 6])
        }
    }

    input_test_range = jnp.arange(start=-1, stop=1, step=.2)
    for x in input_test_range:
        for y in input_test_range:
            inputs = jnp.asarray([x, y])
            outputs = cgp.apply(
                cgp_genome,
                inputs,
            )
            expected_outputs = jnp.tanh(jnp.asarray([x, cgp.input_constants[0], x + y, (x + y) * y]))
            pytest.assume(jnp.allclose(outputs, expected_outputs, rtol=1e-5, atol=1e-8))


def test_active_graph() -> None:
    """Test that a CGP genomes has the correct active nodes.
        """
    # define genome structure
    cgp = CGP(
        n_inputs=2,
        n_outputs=3,
        n_nodes=5
    )
    cgp_genome = {
        "params": {
            "x_connections_genes": jnp.asarray([0, 0, 4, 0, 0]),
            "y_connections_genes": jnp.asarray([1, 1, 5, 1, 1]),
            "functions_genes": jnp.asarray([0, 0, 4, 0, 0]),
            "output_connections_genes": jnp.asarray([0, 2, 4, 6])
        }
    }
    expected_active_nodes = jnp.asarray([1, 0, 1, 0, 0])
    active_nodes = cgp.compute_active_nodes(cgp_genome)
    pytest.assume(jnp.array_equal(active_nodes, expected_active_nodes))
