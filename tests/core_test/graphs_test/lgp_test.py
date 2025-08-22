import jax
import jax.numpy as jnp
import pytest

from qdax.core.graphs.linear_genetic_programming import LGP, lgp_mutation, lgp_crossover


def test_genome_bounds() -> None:
    """Test that a LGP genome has all elements in the correct bounds.
    Tests both at initialization and after mutation.
    """
    # define genome structure
    lgp = LGP(
        n_inputs=2,
        n_outputs=1,
    )
    key = jax.random.key(42)

    # define expected bounds
    lhs_lower_bound = lgp.n_inputs + len(lgp.input_constants)
    assignments_upper_bounds = lgp.n_registers
    functions_bound = len(lgp.function_set)

    # init genome
    key, init_key = jax.random.split(key)
    initial_lgp_genome = lgp.init(init_key)

    # check if bounds are respected at initialization
    pytest.assume(jnp.all(initial_lgp_genome["genes"]["targets"] >= lhs_lower_bound))
    pytest.assume(jnp.all(initial_lgp_genome["genes"]["targets"] < assignments_upper_bounds))
    pytest.assume(jnp.all(initial_lgp_genome["genes"]["inputs1"] < assignments_upper_bounds))
    pytest.assume(jnp.all(initial_lgp_genome["genes"]["inputs2"] < assignments_upper_bounds))
    pytest.assume(jnp.all(initial_lgp_genome["genes"]["functions"] < functions_bound))

    # mutate genome
    key, mut_key = jax.random.split(key)
    mutated_lgp_genome = lgp_mutation(
        genotype=initial_lgp_genome,
        rnd_key=mut_key,
        lgp=lgp
    )

    # check if bounds are respected after mutation
    pytest.assume(jnp.all(mutated_lgp_genome["genes"]["targets"] >= lhs_lower_bound))
    pytest.assume(jnp.all(mutated_lgp_genome["genes"]["targets"] < assignments_upper_bounds))
    pytest.assume(jnp.all(mutated_lgp_genome["genes"]["inputs1"] < assignments_upper_bounds))
    pytest.assume(jnp.all(mutated_lgp_genome["genes"]["inputs2"] < assignments_upper_bounds))
    pytest.assume(jnp.all(mutated_lgp_genome["genes"]["functions"] < functions_bound))

    # init another genome and perform crossover
    key, another_init_key = jax.random.split(key)
    another_lgp_genome = lgp.init(another_init_key)
    key, xover_key = jax.random.split(key)
    crossed_genome = lgp_crossover(mutated_lgp_genome, another_lgp_genome, xover_key, lgp)

    # check if bounds are respected after crossover
    pytest.assume(jnp.all(crossed_genome["genes"]["targets"] >= lhs_lower_bound))
    pytest.assume(jnp.all(crossed_genome["genes"]["targets"] < assignments_upper_bounds))
    pytest.assume(jnp.all(crossed_genome["genes"]["inputs1"] < assignments_upper_bounds))
    pytest.assume(jnp.all(crossed_genome["genes"]["inputs2"] < assignments_upper_bounds))
    pytest.assume(jnp.all(crossed_genome["genes"]["functions"] < functions_bound))


def test_known_genome_execution() -> None:
    """Test that a lGP genome behaves as expected.
    The chosen genome takes as outputs:
    - input 0
    - constant 0
    - input 0 + input 1
    - (input 0 + input 1) * input 1
    All outputs are wrapped by the tanh function.
        """
    # define genome structure
    lgp = LGP(
        n_inputs=2,
        n_outputs=4,
        n_program_lines=4,
        n_computation_registers=4
    )
    lgp_genome = {
        "genes": {
            "targets": jnp.asarray([8, 9, 10, 11]),
            "inputs1": jnp.asarray([0, 2, 0, 10]),
            "inputs2": jnp.asarray([3, 3, 1, 1]),
            "functions": jnp.asarray([2, 2, 0, 2]),
        }
    }

    input_test_range = jnp.arange(start=-1, stop=1, step=.2)
    for x in input_test_range:
        for y in input_test_range:
            inputs = jnp.asarray([x, y])
            outputs = lgp.apply(
                lgp_genome,
                inputs,
            )
            expected_outputs = jnp.tanh(jnp.asarray([x, lgp.input_constants[0], x + y, (x + y) * y]))
            pytest.assume(jnp.allclose(outputs, expected_outputs, rtol=1e-5, atol=1e-8))


def test_active_lines() -> None:
    """Test that a LGP genomes has the correct active nodes.
        """
    # define genome structure
    lgp = LGP(
        n_inputs=2,
        n_outputs=4,
        n_program_lines=5,
        n_computation_registers=4
    )
    lgp_genome = {
        "genes": {
            "targets": jnp.asarray([8, 9, 10, 11, 4]),
            "inputs1": jnp.asarray([0, 2, 0, 10, 2]),
            "inputs2": jnp.asarray([3, 3, 1, 1, 10]),
            "functions": jnp.asarray([2, 2, 0, 2, 1]),
        }
    }
    expected_active_lines = jnp.asarray([1, 1, 1, 1, 0])
    active_lines = lgp.compute_active_lines(lgp_genome)
    pytest.assume(jnp.array_equal(active_lines, expected_active_lines))

    # define genome structure
    lgp2 = LGP(
        n_inputs=2,
        n_outputs=2,
        n_program_lines=2,
        n_computation_registers=4
    )
    lgp_genome2 = {
        "genes": {
            "targets": jnp.asarray([5, 9]),
            "inputs1": jnp.asarray([0, 0]),
            "inputs2": jnp.asarray([1, 5]),
            "functions": jnp.asarray([2, 5]),
        }
    }
    expected_active_lines2 = jnp.asarray([0, 1])
    active_lines2 = lgp2.compute_active_lines(lgp_genome2)
    pytest.assume(jnp.array_equal(active_lines2, expected_active_lines2))


def test_active_lines_jit() -> None:
    """Test that the computation of the active lines is jittable.
    """
    key = jax.random.key(42)
    lgp = LGP(
        n_inputs=3,
        n_outputs=2,
    )

    # Init the population of LGP genomes
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=10)
    init_lgp_genomes = jax.vmap(lgp.init)(keys)

    # Check it runs
    jax.vmap(lgp.compute_active_lines)(init_lgp_genomes)


def test_readable_program() -> None:
    lgp = LGP(
        n_inputs=2,
        n_outputs=4,
        n_program_lines=5,
        n_computation_registers=4
    )
    lgp_genome = {
        "genes": {
            "targets": jnp.asarray([8, 9, 10, 11, 4]),
            "inputs1": jnp.asarray([0, 2, 0, 10, 2]),
            "inputs2": jnp.asarray([3, 3, 1, 1, 10]),
            "functions": jnp.asarray([2, 2, 0, 2, 1]),
        }
    }
    print(lgp.get_readable_program(lgp_genome))

    lgp2 = LGP(
        n_inputs=2,
        n_outputs=2,
        n_program_lines=2,
        n_computation_registers=4
    )
    lgp_genome2 = {
        "genes": {
            "targets": jnp.asarray([5, 9]),
            "inputs1": jnp.asarray([0, 0]),
            "inputs2": jnp.asarray([1, 5]),
            "functions": jnp.asarray([2, 5]),
        }
    }
    print(lgp2.get_readable_program(lgp_genome2))


def test_readable_expression() -> None:
    lgp = LGP(
        n_inputs=2,
        n_outputs=4,
        n_program_lines=5,
        n_computation_registers=4
    )
    lgp_genome = {
        "genes": {
            "targets": jnp.asarray([8, 9, 10, 11, 4]),
            "inputs1": jnp.asarray([0, 2, 0, 10, 2]),
            "inputs2": jnp.asarray([6, 3, 1, 1, 10]),
            "functions": jnp.asarray([2, 2, 0, 2, 1]),
        }
    }
    print(lgp.get_readable_program(lgp_genome), "\n")

    print(lgp.get_readable_expression(lgp_genome), "\n")

    inputs_mapping_fn = lambda x: f"i_{{{x}}}"
    print(lgp.get_readable_expression(lgp_genome, inputs_mapping=inputs_mapping_fn), "\n")

    inputs_mapping_dict = {0: "a", 1: "b"}
    print(lgp.get_readable_expression(lgp_genome, inputs_mapping=inputs_mapping_dict), "\n")

    outputs_mapping_fn = lambda x: f"o_{{{x}}}"
    print(lgp.get_readable_expression(lgp_genome, outputs_mapping=outputs_mapping_fn), "\n")

    outputs_mapping_dict = {0: "x", 1: "y"}
    print(lgp.get_readable_expression(lgp_genome, outputs_mapping=outputs_mapping_dict), "\n")
