import jax
from jax import jit
from typing import Any, Callable, Sequence
from jax import lax, random, numpy as jnp
from flax.core import freeze, unfreeze
from flax import linen as nn


class ExplicitMLP(nn.Module):
  features: Sequence[int]

  def setup(self):
    # we automatically know what to do with lists, dicts of submodules
    self.layers = [nn.Dense(feat) for feat in self.features]
    # for single submodules, we would just write:
    # self.layer1 = nn.Dense(feat1)

  def __call__(self, inputs):
    x = inputs
    for i, lyr in enumerate(self.layers):
      x = lyr(x)
      if i != len(self.layers) - 1:
        x = nn.relu(x)
    return x


if __name__ == '__main__':

    key1, key2 = random.split(random.PRNGKey(0), 2)
    x = random.uniform(key1, (4,4))

    model = ExplicitMLP(features=[3,4,5])
    params = model.init(key2, x)
    y = model.apply(params, x)

    print('input:\n', x)
    print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(params)))
    print('output:\n', y)