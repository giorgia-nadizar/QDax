import jax
import jax.numpy as jnp
import optax
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
    pytest.assume(jnp.all(initial_cgp_genome["params"]["node_weights"] == 1))
    pytest.assume(jnp.all(initial_cgp_genome["params"]["input_weights1"] == 1))
    pytest.assume(jnp.all(initial_cgp_genome["params"]["input_weights2"] == 1))

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
    pytest.assume(jnp.all(mutated_cgp_genome["params"]["node_weights"] == 1))
    pytest.assume(jnp.all(mutated_cgp_genome["params"]["input_weights1"] == 1))
    pytest.assume(jnp.all(mutated_cgp_genome["params"]["input_weights2"] == 1))


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
            "node_weights": jnp.ones(cgp.n_nodes),
            "input_weights1": jnp.ones(cgp.n_nodes),
            "input_weights2": jnp.ones(cgp.n_nodes),
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
            "node_weights": jnp.ones(cgp.n_nodes),
            "input_weights1": jnp.ones(cgp.n_nodes),
            "input_weights2": jnp.ones(cgp.n_nodes), }
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
    )
    cgp_genome = {
        "params": {
            "inputs1": jnp.asarray([0, 0, 4, 0, 0]),
            "inputs2": jnp.asarray([1, 1, 5, 1, 1]),
            "functions": jnp.asarray([0, 0, 4, 0, 0]),
            "outputs": jnp.asarray([0, 2, 4, 6]),
            "node_weights": jnp.ones(cgp.n_nodes),
            "input_weights1": jnp.ones(cgp.n_nodes),
            "input_weights2": jnp.ones(cgp.n_nodes),
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


def test_gradient_optimization_of_node_weights() -> None:
    # Generate genome
    cgp = CGP(
        n_inputs=3,
        input_constants=jnp.asarray([]),
        n_outputs=2,
        n_nodes=4,
        weighted_nodes=True,
        weighted_connections=False
    )
    target_weights = jnp.asarray([.2, -.5, .4, -.3])
    cgp_genome = {
        "params": {
            "inputs1": jax.lax.stop_gradient(jnp.asarray([0, 1, 3, 0])),
            "inputs2": jax.lax.stop_gradient(jnp.asarray([0, 2, 4, 1])),
            "functions": jax.lax.stop_gradient(jnp.asarray([6, 2, 0, 0])),
            "outputs": jax.lax.stop_gradient(jnp.asarray([3, 5])),
            "node_weights": target_weights,
            "input_weights1": jnp.ones(cgp.n_nodes),
            "input_weights2": jnp.ones(cgp.n_nodes),
        }
    }
    active = cgp.compute_active_nodes(cgp_genome)
    print(target_weights * active)

    # Generate synthetic dataset
    n_samples = 500
    key = jax.random.key(0)
    x_key, y_key, z_key, noise_key, weights_key = jax.random.split(key, 5)
    x = jax.random.uniform(x_key, (n_samples,), minval=0, maxval=2 * jnp.pi)
    y = jax.random.normal(y_key, (n_samples,))
    z = jax.random.normal(z_key, (n_samples,))
    observations = jnp.vstack((x, y, z)).T
    noise = 0.01 * jax.random.normal(noise_key, (n_samples, cgp.n_outputs))
    target_outputs = (jax.vmap(cgp.apply, (None, 0, None))
                      (cgp_genome, observations, {"node_weights": target_weights}) + noise)

    # Initialize weights to random values
    cgp_weights = jax.random.uniform(key=weights_key, shape=(cgp.n_nodes,)) * 2 - 1

    # Loss = mean squared error
    def loss_fn(weights, genome, inputs, target_y):
        pred_y = jax.vmap(cgp.apply, (None, 0, None))(genome, inputs, {"node_weights": weights})
        return jnp.mean((pred_y - target_y) ** 2)

    @jax.jit
    def step(genome, weights, opt_st, inputs, targets):
        loss, grads = jax.value_and_grad(loss_fn)(weights, genome, inputs, targets)
        updates, opt_st = optimizer.update(grads, opt_st)
        params = optax.apply_updates(weights, updates)
        return params, opt_state, loss

    # Optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(cgp_weights)

    # Training loop
    for i in range(50_000):
        cgp_weights, opt_state, train_loss = step(cgp_genome, cgp_weights, opt_state, observations, target_outputs)
        # if i % 1_000 == 0:
        # print(f"Step {i}, Loss {train_loss}, Params {cgp_weights}")
    print(cgp_weights * active)


def test_gradient_optimization_of_connection_weights() -> None:
    # Generate genome
    cgp = CGP(
        n_inputs=3,
        input_constants=jnp.asarray([]),
        n_outputs=2,
        n_nodes=4,
        weighted_nodes=False,
        weighted_connections=True
    )
    target_weights1 = jnp.asarray([.2, -.5, .4, -.3])
    target_weights2 = jnp.asarray([-.3, .7, .1, -1.])
    cgp_genome = {
        "params": {
            "inputs1": jax.lax.stop_gradient(jnp.asarray([0, 1, 3, 0])),
            "inputs2": jax.lax.stop_gradient(jnp.asarray([0, 2, 4, 1])),
            "functions": jax.lax.stop_gradient(jnp.asarray([6, 2, 0, 0])),
            "outputs": jax.lax.stop_gradient(jnp.asarray([3, 5])),
            "node_weights": jnp.ones(cgp.n_nodes),
            "input_weights1": target_weights1,
            "input_weights2": target_weights2,
        }
    }
    active = cgp.compute_active_nodes(cgp_genome)

    # Generate synthetic dataset
    n_samples = 500
    key = jax.random.key(0)
    x_key, y_key, z_key, noise_key, weights_key = jax.random.split(key, 5)
    x = jax.random.uniform(x_key, (n_samples,), minval=0, maxval=2 * jnp.pi)
    y = jax.random.normal(y_key, (n_samples,))
    z = jax.random.normal(z_key, (n_samples,))
    observations = jnp.vstack((x, y, z)).T
    noise = 0.01 * jax.random.normal(noise_key, (n_samples, cgp.n_outputs))
    target_outputs = (jax.vmap(cgp.apply, (None, 0, None))
                      (cgp_genome, observations,
                       {"input_weights1": target_weights1, "input_weights2": target_weights2}) )

    # Initialize weights to random values
    cgp_weights = jax.random.uniform(key=weights_key, shape=(cgp.n_nodes * 2,)) * 2 - 1
    weights1, weights2 = jnp.split(cgp_weights, 2)
    optimizable_weights = {"input_weights1": weights1, "input_weights2": weights2}

    # Loss = mean squared error
    def loss_fn(weights_dict, genome, inputs, target_y):
        pred_y = jax.vmap(cgp.apply, (None, 0, None))(genome, inputs, weights_dict)
        return jnp.mean((pred_y - target_y) ** 2)

    @jax.jit
    def step(genome, weights_dict, opt_st, inputs, targets):
        loss, grads = jax.value_and_grad(loss_fn)(weights_dict, genome, inputs, targets)
        updates, opt_st = optimizer.update(grads, opt_st)
        weights_dict = optax.apply_updates(weights_dict, updates)
        return weights_dict, opt_state, loss

    # Optimizer
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(optimizable_weights)

    # Training loop
    for i in range(50_000):
        optimizable_weights, opt_state, train_loss = step(cgp_genome, optimizable_weights, opt_state, observations, target_outputs)
        # if i % 1_000 == 0:
        # print(f"Step {i}, Loss {train_loss}, Params {cgp_weights}")
    print(train_loss)
    print(cgp.get_readable_expression(cgp_genome))
    print(optimizable_weights["input_weights1"] * active, target_weights1 * active)
    print(optimizable_weights["input_weights2"] * active, target_weights2 * active)
