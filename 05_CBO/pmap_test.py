from os import environ
CORES = 12
environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={CORES}'

from jax import pmap, numpy as jnp
import numpy as np

x = jnp.arange(10)
y = jnp.arange(-5,5)

f = lambda x, y: x*y
out = pmap(f)(x, y)
print(out)