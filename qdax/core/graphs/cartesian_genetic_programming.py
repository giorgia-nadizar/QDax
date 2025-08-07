from typing import Callable, Union, Dict, Any, Tuple

import jax.numpy as jnp
from chex import PRNGKey
from flax import struct
from flax.typing import RNGSequences, FrozenVariableDict
from jax import random, jit
from jax._src.lax.control_flow import fori_loop

from qdax.core.graphs.functions import FunctionSet


@struct.dataclass
class CGP:
    n_inputs: int
    n_nodes: int
    n_outputs: int
    function_set: FunctionSet
    input_constants: jnp.ndarray = jnp.asarray([0.1, 1.0])
    outputs_wrapper: Callable = jnp.tanh
    fixed_outputs: bool = False

    def init(self, rngs: Union[PRNGKey, RNGSequences], *args, ) -> Union[FrozenVariableDict, Dict[str, Any]]:
        # determine bounds for genes for each section of the genome
        in_mask = jnp.arange(self.n_inputs + len(self.input_constants),
                             self.n_inputs + len(self.input_constants) + self.n_nodes)
        f_mask = len(self.function_set) * jnp.ones(self.n_nodes)
        out_mask = (self.n_inputs + len(self.input_constants) + self.n_nodes) * jnp.ones(self.n_outputs)

        # generate the random float values for each section of the genome
        x_key, y_key, f_key, out_key = random.split(rngs, 4)
        random_x = random.uniform(key=x_key, shape=in_mask.shape)
        random_y = random.uniform(key=y_key, shape=in_mask.shape)
        random_f = random.uniform(key=f_key, shape=f_mask.shape)
        random_out = random.uniform(key=out_key, shape=out_mask.shape)

        # rescale, cast to integer and store the random genome parts
        return {
            "params": {
                "x_connections_genes": jnp.floor(random_x * in_mask).astype(int),
                "y_connections_genes": jnp.floor(random_y * in_mask).astype(int),
                "functions_genes": jnp.floor(random_f * f_mask).astype(int),
                "output_connections_genes": out_mask if self.fixed_outputs else
                jnp.floor(random_out * out_mask).astype(int)
            }
        }

    def apply(self,
              cgp_genome_params: Union[FrozenVariableDict, Dict[str, Any]],
              obs: jnp.ndarray,
              ) -> jnp.ndarray:
        # define function to update buffer in a certain position: get inputs from the x and y connections
        # then apply the function
        @jit
        def _update_buffer(buffer_idx: int,
                           carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) \
                -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            x_genes, y_genes, f_genes, buff = carry
            n_in = len(buff) - len(x_genes)
            idx = buffer_idx - n_in
            f_idx = f_genes.at[idx].get()
            x_arg = buff.at[x_genes.at[idx].get()].get()
            y_arg = buff.at[y_genes.at[idx].get()].get()
            f_computed = self.function_set.apply(f_idx, x_arg, y_arg)
            buff = buff.at[buffer_idx].set(f_computed)
            return x_genes, y_genes, f_genes, buff

        # initialize the buffer with inputs and constants and use zeros as placeholders for computation
        buffer = jnp.concatenate([obs, self.input_constants, jnp.zeros(self.n_nodes)])
        # apply the buffer update function for all positions of the buffer to update it
        _, _, _, buffer = fori_loop(self.n_inputs, len(buffer), _update_buffer,
                                    (cgp_genome_params["params"]["x_connections_genes"],
                                     cgp_genome_params["params"]["y_connections_genes"],
                                     cgp_genome_params["params"]["functions_genes"], buffer))
        outputs = jnp.take(buffer, cgp_genome_params["params"]["output_connections_genes"])

        # apply wrapper to constraint the outputs in the correct domain
        return self.outputs_wrapper(outputs)
