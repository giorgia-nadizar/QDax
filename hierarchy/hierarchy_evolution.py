import jax.numpy as jnp

# evolve a CGP with two outputs: degree to which we want to go left / right, and forward / back

base_path = "../results/pgame_ant_omni_0"
fitnesses = jnp.load(f"{base_path}/fitnesses.npy")
print(fitnesses)