# run PGA-ME first with few centroids with ant-omni
# TODO clean up
import functools
import os
import time

import jax
import jax.numpy as jnp

from qdax import environments
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.pga_me_emitter import PGAMEConfig, PGAMEEmitter
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.brax_envs import scoring_function_brax_envs as scoring_function
from qdax.utils.metrics import CSVLogger, default_qd_metrics

env_name = 'ant_omni'  # @param['ant_uni', 'hopper_uni', 'walker_uni', 'halfcheetah_uni', 'humanoid_uni', 'ant_omni', 'humanoid_omni']
seed = 0  # @param {type:"integer"}
file_path = f"me_{env_name}_{seed}"
episode_length = 250  # @param {type:"integer"}
num_iterations = 10_000  # @param {type:"integer"}
policy_hidden_layer_sizes = (256, 256)  # @param {type:"raw"}
iso_sigma = 0.005  # @param {type:"number"}
line_sigma = 0.05  # @param {type:"number"}
# num_init_cvt_samples = 50000 #@param {type:"integer"}
# num_centroids = 100 #@param {type:"integer"}
n_descriptors_per_dimension = 5
min_bd = -15.0  # @param {type:"number"}
max_bd = 15.0  # @param {type:"number"}
early_stopping = True
batch_size = 100

proportion_mutation_ga = 0.5  # @param {type:"number"}

# TD3 params
env_batch_size = 100  # @param {type:"number"}
replay_buffer_size = 1000000  # @param {type:"number"}
critic_hidden_layer_size = (256, 256)  # @param {type:"raw"}
critic_learning_rate = 3e-4  # @param {type:"number"}
greedy_learning_rate = 3e-4  # @param {type:"number"}
policy_learning_rate = 1e-3  # @param {type:"number"}
noise_clip = 0.5  # @param {type:"number"}
policy_noise = 0.2  # @param {type:"number"}
discount = 0.99  # @param {type:"number"}
reward_scaling = 1.0  # @param {type:"number"}
transitions_batch_size = 256  # @param {type:"number"}
soft_tau_update = 0.005  # @param {type:"number"}
num_critic_training_steps = 300  # @param {type:"number"}
num_pg_training_steps = 100  # @param {type:"number"}
policy_delay = 2  # @param {type:"number"}

# Init environment
env = environments.create(env_name, episode_length=episode_length)

# Init a random key
random_key = jax.random.PRNGKey(seed)

# Init policy network
policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
policy_network = MLP(
    layer_sizes=policy_layer_sizes,
    kernel_init=jax.nn.initializers.lecun_uniform(),
    final_activation=jnp.tanh,
)

# Init population of controllers
random_key, subkey = jax.random.split(random_key)
keys = jax.random.split(subkey, num=env_batch_size)
fake_batch = jnp.zeros(shape=(env_batch_size, env.observation_size))
init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

# Create the initial environment states
random_key, subkey = jax.random.split(random_key)
keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=env_batch_size, axis=0)
reset_fn = jax.jit(jax.vmap(env.reset))
init_states = reset_fn(keys)


# Define the fonction to play a step with the policy in the environment
def play_step_fn(
        env_state,
        policy_params,
        random_key,
):
    """
    Play an environment step and return the updated state and the transition.
    """

    actions = policy_network.apply(policy_params, env_state.obs)

    state_desc = env_state.info["state_descriptor"]
    next_state = env.step(env_state, actions)

    transition = QDTransition(
        obs=env_state.obs,
        next_obs=next_state.obs,
        rewards=next_state.reward,
        dones=next_state.done,
        actions=actions,
        truncations=next_state.info["truncation"],
        state_desc=state_desc,
        next_state_desc=next_state.info["state_descriptor"],
    )

    return next_state, policy_params, random_key, transition


# Prepare the scoring function
bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]
scoring_fn = functools.partial(
    scoring_function,
    init_states=init_states,
    episode_length=episode_length,
    play_step_fn=play_step_fn,
    behavior_descriptor_extractor=bd_extraction_fn,
)

# Get minimum reward value to make sure qd_score are positive
reward_offset = environments.reward_offset[env_name]

# Define a metrics function
metrics_function = functools.partial(
    default_qd_metrics,
    qd_offset=reward_offset * episode_length,
)

# # Define the PG-emitter config
# pga_emitter_config = PGAMEConfig(
#     env_batch_size=env_batch_size,
#     batch_size=transitions_batch_size,
#     proportion_mutation_ga=proportion_mutation_ga,
#     critic_hidden_layer_size=critic_hidden_layer_size,
#     critic_learning_rate=critic_learning_rate,
#     greedy_learning_rate=greedy_learning_rate,
#     policy_learning_rate=policy_learning_rate,
#     noise_clip=noise_clip,
#     policy_noise=policy_noise,
#     discount=discount,
#     reward_scaling=reward_scaling,
#     replay_buffer_size=replay_buffer_size,
#     soft_tau_update=soft_tau_update,
#     num_critic_training_steps=num_critic_training_steps,
#     num_pg_training_steps=num_pg_training_steps,
#     policy_delay=policy_delay,
# )
#
# # Get the emitter
# variation_fn = functools.partial(
#     isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma
# )
#
# pg_emitter = PGAMEEmitter(
#     config=pga_emitter_config,
#     policy_network=policy_network,
#     env=env,
#     variation_fn=variation_fn,
# )

# Define emitter
variation_fn = functools.partial(
    isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma
)
mixing_emitter = MixingEmitter(
    mutation_fn=None,
    variation_fn=variation_fn,
    variation_percentage=1.0,
    batch_size=batch_size
)

# Instantiate MAP Elites
map_elites = MAPElites(
    scoring_function=scoring_fn,
    # emitter=pg_emitter,
    emitter=mixing_emitter,
    metrics_function=metrics_function,
)

# Compute the centroids
centroids = compute_euclidean_centroids(
    grid_shape=tuple(n_descriptors_per_dimension for _ in range(env.behavior_descriptor_length)),
    minval=min_bd,
    maxval=max_bd,
)

# compute initial repertoire
repertoire, emitter_state, random_key = map_elites.init(
    init_variables, centroids, random_key
)

log_period = 10
num_loops = int(num_iterations / log_period)

csv_logger = CSVLogger(
    f"../results/{file_path}.csv",
    header=["loop", "iteration", "qd_score", "max_fitness", "coverage", "time"]
)
all_metrics = {}

repertoire_path = f"../results/{file_path}/"
os.makedirs(repertoire_path, exist_ok=True)

# main loop
map_elites_scan_update = map_elites.scan_update
for i in range(num_loops):
    start_time = time.time()
    # main iterations
    (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
        map_elites_scan_update,
        (repertoire, emitter_state, random_key),
        (),
        length=log_period,
    )
    timelapse = time.time() - start_time

    # log metrics
    logged_metrics = {"time": timelapse, "loop": 1 + i, "iteration": 1 + i * log_period}
    for key, value in metrics.items():
        # take last value
        logged_metrics[key] = value[-1]

        # take all values
        if key in all_metrics.keys():
            all_metrics[key] = jnp.concatenate([all_metrics[key], value])
        else:
            all_metrics[key] = value

    csv_logger.log(logged_metrics)

    print(f"Loop: {i + 1}, "
          f"max_fitness: {logged_metrics['max_fitness']}, "
          f"time: {logged_metrics['time']}, "
          f"coverage: {logged_metrics['coverage']}")

    if logged_metrics['coverage'] > 99.9 and early_stopping:
        break

    if (num_loops + 1) % 5 == 0:
        repertoire.save(path=repertoire_path)

repertoire.save(path=repertoire_path)

# Create the plots and the grid
# fig, axes = plot_map_elites_results(env_steps=env_steps, metrics=metrics, repertoire=repertoire,
#                                     min_descriptor=min_descriptor, max_descriptor=max_descriptor)
