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
        n_nodes=5,
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
    pytest.assume(jnp.all(initial_cgp_genome["params"]["inputs1"] < connections_bounds))
    pytest.assume(jnp.all(initial_cgp_genome["params"]["inputs2"] < connections_bounds))
    pytest.assume(jnp.all(initial_cgp_genome["params"]["functions"] < functions_bound))
    pytest.assume(jnp.all(initial_cgp_genome["params"]["outputs"] < outputs_bound))
    pytest.assume(jnp.all(initial_cgp_genome["params"]["weights"] == 1))

    # mutate genome
    key, mut_key = jax.random.split(key)
    mutated_cgp_genome = cgp_mutation(
        genotype=initial_cgp_genome,
        rnd_key=mut_key,
        cgp=cgp
    )

    # check if bounds are respected after mutation
    pytest.assume(jnp.all(mutated_cgp_genome["params"]["inputs1"] < connections_bounds))
    pytest.assume(jnp.all(mutated_cgp_genome["params"]["inputs2"] < connections_bounds))
    pytest.assume(jnp.all(mutated_cgp_genome["params"]["functions"] < functions_bound))
    pytest.assume(jnp.all(mutated_cgp_genome["params"]["outputs"] < outputs_bound))
    pytest.assume(jnp.all(initial_cgp_genome["params"]["weights"] == 1))


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
        n_outputs=4,
        n_nodes=5
    )
    cgp_genome = {
        "params": {
            "inputs1": jnp.asarray([0, 0, 4, 0, 0]),
            "inputs2": jnp.ones(cgp.n_nodes, dtype=jnp.int32),
            "functions": jnp.asarray([0, 0, 2, 0, 0]),
            "outputs": jnp.asarray([0, 2, 4, 6]),
            "weights": jnp.ones(cgp.n_nodes),
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
        n_outputs=4,
        n_nodes=5
    )
    cgp_genome = {
        "params": {
            "inputs1": jnp.asarray([0, 0, 4, 0, 0]),
            "inputs2": jnp.asarray([1, 1, 5, 1, 1]),
            "functions": jnp.asarray([0, 0, 4, 0, 0]),
            "outputs": jnp.asarray([0, 2, 4, 6]),
            "weights": jnp.ones(cgp.n_nodes),
        }
    }
    expected_active_nodes = jnp.asarray([1, 0, 1, 0, 0])
    active_nodes = cgp.compute_active_nodes(cgp_genome)
    pytest.assume(jnp.array_equal(active_nodes, expected_active_nodes))


def test_active_graph_jit() -> None:
    """Test that the computation of the active graph is jittable.
    """
    key = jax.random.key(42)
    cgp = CGP(
        n_inputs=3,
        n_outputs=2,
    )

    # Init the population of CGP genomes
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=10)
    init_cgp_genomes = jax.vmap(cgp.init)(keys)

    # Check it runs
    jax.vmap(cgp.compute_active_nodes)(init_cgp_genomes)


def test_readable_expression() -> None:
    cgp = CGP(
        n_inputs=2,
        n_outputs=4,
        n_nodes=5,
        weighted_graph=False
    )
    cgp_genome = {
        "params": {
            "inputs1": jnp.asarray([0, 0, 4, 0, 0]),
            "inputs2": jnp.asarray([1, 1, 5, 1, 1]),
            "functions": jnp.asarray([0, 0, 4, 0, 0]),
            "outputs": jnp.asarray([0, 2, 4, 6]),
            "weights": jnp.ones(cgp.n_nodes),
        }
    }
    print(cgp.get_readable_expression(cgp_genome), "\n")

    inputs_mapping_fn = lambda x: f"i_{{{x}}}"
    print(cgp.get_readable_expression(cgp_genome, inputs_mapping=inputs_mapping_fn), "\n")

    inputs_mapping_dict = {0: "a", 1: "b"}
    print(cgp.get_readable_expression(cgp_genome, inputs_mapping=inputs_mapping_dict), "\n")

    outputs_mapping_fn = lambda x: f"o_{{{x}}}"
    print(cgp.get_readable_expression(cgp_genome, outputs_mapping=outputs_mapping_fn), "\n")

    outputs_mapping_dict = {0: "x", 1: "y"}
    print(cgp.get_readable_expression(cgp_genome, outputs_mapping=outputs_mapping_dict), "\n")
