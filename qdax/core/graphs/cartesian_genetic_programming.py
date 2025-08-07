from typing import Callable, Union, Dict
import jax.numpy as jnp
from chex import PRNGKey

from flax import struct
from flax.typing import RNGSequences
from jax import random

from qdax.core.graphs.functions import JaxFunction


@struct.dataclass
class CGP:
    n_inputs: int
    n_nodes: int
    n_outputs: int
    function_set: Dict[str, JaxFunction]
    input_constants: jnp.ndarray = jnp.asarray([0.1, 1.0])
    # buffer_update_fn: Callable = struct.field(pytree_node=False)   # non-array → static
    outputs_wrapper: Callable = jnp.tanh
    fixed_outputs: bool = False

    def init(self, rngs: Union[PRNGKey, RNGSequences], *args, ):
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
