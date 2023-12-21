import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids, compute_cvt_centroids_from_samples

n_init_cvt_samples = 500000
n_centroids = 1024
random_key = jax.random.PRNGKey(0)

behavior_centroids_pointmaze, random_key = compute_cvt_centroids(
    num_descriptors=2,
    num_init_cvt_samples=n_init_cvt_samples,
    num_centroids=n_centroids,
    minval=-1,
    maxval=1,
    random_key=random_key,
)
jnp.save("../../docker-res/behavior_centroids_pointmaze.npy", behavior_centroids_pointmaze)
jnp.save("../../results/behavior_centroids_pointmaze.npy", behavior_centroids_pointmaze)
print("behavior pointmaze")

random_key, subkey = jax.random.split(random_key)
xs = jax.random.uniform(key=subkey, shape=(3 * n_init_cvt_samples, 4), minval=jnp.array([-1, -1, 0, 0]),
                        maxval=jnp.ones(4))
pointmaze_graph_samples_list = []
for x in xs:
    if x[3] + x[2] <= 1:
        pointmaze_graph_samples_list.append(x)
    if len(pointmaze_graph_samples_list) >= n_init_cvt_samples:
        break

pointmaze_graph_centroids, random_key = compute_cvt_centroids_from_samples(jnp.asarray(pointmaze_graph_samples_list),
                                                                           n_centroids, random_key)
jnp.save("../../docker-res/me_centroids_pointmaze.npy", pointmaze_graph_centroids)
jnp.save("../../results/me_centroids_pointmaze.npy", pointmaze_graph_centroids)
print("me pointmaze")

behavior_centroids_2d, random_key = compute_cvt_centroids(
    num_descriptors=2,
    num_init_cvt_samples=n_init_cvt_samples,
    num_centroids=n_centroids,
    minval=0,
    maxval=1,
    random_key=random_key,
)
jnp.save("../../docker-res/behavior_centroids_2d.npy", behavior_centroids_2d)
print("behavior 2d")

behavior_centroids_1d, random_key = compute_cvt_centroids(
    num_descriptors=1,
    num_init_cvt_samples=n_init_cvt_samples,
    num_centroids=n_centroids,
    minval=0,
    maxval=1,
    random_key=random_key,
)
jnp.save("../../docker-res/behavior_centroids_1d.npy", behavior_centroids_1d)
print("behavior 1d")

# create graph samples
random_key, subkey = jax.random.split(random_key)
xs = jax.random.uniform(key=subkey, shape=(3 * n_init_cvt_samples, 2))
graph_samples_list = []
for x in xs:
    if x[0] + x[1] <= 1:
        graph_samples_list.append(x)
    if len(graph_samples_list) >= n_init_cvt_samples:
        break

graph_centroids, random_key = compute_cvt_centroids_from_samples(jnp.asarray(graph_samples_list), n_centroids,
                                                                 random_key)
jnp.save("../../docker-res/graph_centroids.npy", graph_centroids)
print("graph")

# create behavior1d + graph samples
random_key, subkey = jax.random.split(random_key)
xs = jax.random.uniform(key=subkey, shape=(3 * n_init_cvt_samples, 3))
b1d_graph_samples_list = []
for x in xs:
    if x[1] + x[2] <= 1:
        b1d_graph_samples_list.append(x)
    if len(b1d_graph_samples_list) >= n_init_cvt_samples:
        break

b1d_graph_centroids, random_key = compute_cvt_centroids_from_samples(jnp.asarray(b1d_graph_samples_list), n_centroids,
                                                                     random_key)
jnp.save("../../docker-res/me_centroids_3d.npy", b1d_graph_centroids)
print("me 3d")

# create behavior2d + graph samples
random_key, subkey = jax.random.split(random_key)
xs = jax.random.uniform(key=subkey, shape=(3 * n_init_cvt_samples, 4))
b2d_graph_samples_list = []
for x in xs:
    if x[3] + x[2] <= 1:
        b2d_graph_samples_list.append(x)
    if len(b2d_graph_samples_list) >= n_init_cvt_samples:
        break

b2d_graph_centroids, random_key = compute_cvt_centroids_from_samples(jnp.asarray(b2d_graph_samples_list), n_centroids,
                                                                     random_key)
jnp.save("../../docker-res/me_centroids_4d.npy", b2d_graph_centroids)
print("me 4d")
