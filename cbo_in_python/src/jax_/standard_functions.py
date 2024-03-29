import torch
import jax.numpy as jnp

def rastrigin(v):
    return rastrigin_c()(v)


def rastrigin_c(c=10):
    # lambda v: tf.reduce_sum(v ** 2 - c * tf.math.cos(2 * np.pi * v)) + tf.cast(c, tf.float32)
    return lambda v: (v ** 2 - c * jnp.cos(2 * jnp.pi * v)).sum() + c


def square(v):
    return v ** 2
