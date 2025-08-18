import jax
import jax.numpy as jnp
import pytest

from qdax.core.graphs.linear_genetic_programming import LGP, lgp_mutation


def test_genome_bounds() -> None:
    """Test that a CGP genome has all elements in the correct bounds.
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
    pytest.assume(jnp.all(initial_lgp_genome["params"]["target_registers_genes"] >= lhs_lower_bound))
    pytest.assume(jnp.all(initial_lgp_genome["params"]["target_registers_genes"] < assignments_upper_bounds))
    pytest.assume(jnp.all(initial_lgp_genome["params"]["x_connections_genes"] < assignments_upper_bounds))
    pytest.assume(jnp.all(initial_lgp_genome["params"]["y_connections_genes"] < assignments_upper_bounds))
    pytest.assume(jnp.all(initial_lgp_genome["params"]["functions_genes"] < functions_bound))

    # mutate genome
    key, mut_key = jax.random.split(key)
    mutated_lgp_genome = lgp_mutation(
        genotype=initial_lgp_genome,
        rnd_key=mut_key,
        lgp=lgp
    )

    # check if bounds are respected after mutation
    pytest.assume(jnp.all(mutated_lgp_genome["params"]["target_registers_genes"] >= lhs_lower_bound))
    pytest.assume(jnp.all(mutated_lgp_genome["params"]["target_registers_genes"] < assignments_upper_bounds))
    pytest.assume(jnp.all(mutated_lgp_genome["params"]["x_connections_genes"] < assignments_upper_bounds))
    pytest.assume(jnp.all(mutated_lgp_genome["params"]["y_connections_genes"] < assignments_upper_bounds))
    pytest.assume(jnp.all(mutated_lgp_genome["params"]["functions_genes"] < functions_bound))


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
        "params": {
            "target_registers_genes": jnp.asarray([8, 9, 10, 11]),
            "x_connections_genes": jnp.asarray([0, 2, 0, 10]),
            "y_connections_genes": jnp.asarray([3, 3, 1, 1]),
            "functions_genes": jnp.asarray([2, 2, 0, 2]),
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
