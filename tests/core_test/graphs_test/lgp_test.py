import jax
import jax.numpy as jnp
import pytest

from qdax.core.graphs.linear_genetic_programming import LGP


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
    initial_cgp_genome = lgp.init(init_key)

    # check if bounds are respected at initialization
    pytest.assume(jnp.all(initial_cgp_genome["params"]["target_registers_genes"] >= lhs_lower_bound))
    pytest.assume(jnp.all(initial_cgp_genome["params"]["target_registers_genes"] < assignments_upper_bounds))
    pytest.assume(jnp.all(initial_cgp_genome["params"]["x_connections_genes"] < assignments_upper_bounds))
    pytest.assume(jnp.all(initial_cgp_genome["params"]["y_connections_genes"] < assignments_upper_bounds))
    pytest.assume(jnp.all(initial_cgp_genome["params"]["functions_genes"] < functions_bound))
