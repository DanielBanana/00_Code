import jax
from jax import jit
from typing import Any, Callable, Sequence
from jax import lax, random, numpy as jnp
from flax.core import freeze, unfreeze
from flax import linen as nn


@jit
def net(params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    hidden_layers = params[:-1]
    activation = x
    for w, b in hidden_layers:
        activation = jax.nn.relu(jnp.dot(activation, w) + b)
    w_last, b_last = params[-1]
    output = jnp.dot(activation, w_last) + b_last
    return output
batched_net = jax.vmap(net, in_axes=(0, None))

def net_init(layer_widths, parent_key, scale=1):
    # Row based initialization: Vectors are represented as row vectors
    # like in JAX and numpy
    # Size of input is the first layer_width
    params = []
    keys = jax.random.split(parent_key, num=len(layer_widths)-1)
    for in_width, out_width, key in zip(layer_widths[:-1], layer_widths[1:], keys):
        weight_key, bias_key = jax.random.split(key)
        params.append([
                       scale*jax.random.normal(weight_key, shape=(in_width, out_width)),
                       scale*jax.random.normal(bias_key, shape=(out_width,))
                       ]
        )
    return params

@jit
def mse_flax(params, input_batched, output_batched):
    # Define the squared loss for a single pair(input, output)
    def squared_error(input, output):
        pred = model.apply(params, input)
        return jnp.inner(output-pred, output-pred) / 2.0
    # Vectorize the previous to compute the average of the loss on all samples.
    return jnp.mean(jax.vmap(squared_error)(input_batched, output_batched), axis=0)


@jit
def mse_hand(params, input_batched, output_batched):
    # Define the squared loss for a single pair(input, output)
    def squared_error(input, output):
        pred = net(params, input)
        return jnp.inner(output-pred, output-pred) / 2.0
    # Vectorize the previous to compute the average of the loss on all samples.
    return jnp.mean(jax.vmap(squared_error)(input_batched, output_batched), axis=0)

@jit
def update_params(params, learning_rate, grads):
    params = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g, params, grads)
    return params



if __name__ == '__main__':
    key1, key2 = random.split(random.PRNGKey(0))
    key3, key4 = random.split(key1)
    x = random.normal(key1, (10,)) # Dummy input data

    # Create a Model with the Function from above
    layer_widths = [10,5]
    params_h = net_init(layer_widths, key4)
    print('Implementation by Hand')
    print(jax.tree_util.tree_map(lambda x: x.shape, params_h))
    hand_prd = net(params_h, x)
    print(hand_prd)

    print('###########################################################')

    # Create a Model with Flax
    print('Implementation via flax')
    model = nn.Dense(features=5)
    params = model.init(key2, x) # Initialization call
    print(jax.tree_util.tree_map(lambda x: x.shape, params)) # Checking output shapes
    flax_prd = model.apply(params, x)
    print(flax_prd)


    # Gradient Descent
    n_samples = 20
    x_dim = 10
    y_dim = 5

    # Generate example data via a NN with one layer
    key = random.PRNGKey(0)
    keys = random.split(key, num = 10)

    W = random.normal(keys[0], (x_dim, y_dim))
    b = random.normal(keys[1], (y_dim, ))
    # Store the parameters in a FrozenDict pytree (FrozenDict = immutable Python Dict)
    true_params = freeze({'params': {'bias': b, 'kernel': W}})
    x_samples = random.normal(keys[2], (n_samples, x_dim))
    y_samples = jnp.dot(x_samples, W) + b + 0.1 * random.normal(keys[3], (n_samples, y_dim))
    print('x shape:', x_samples.shape, '; y shape:', y_samples.shape)


    # Perform Gradient Descent 
    learning_rate = 0.3
    # The prediction with the true parameters has a loss since we introdued a random error
    print('Loss for "true" W,b: ', mse_flax(true_params, x_samples, y_samples))
    loss_grad_fnc_flax = jax.value_and_grad(mse_flax)
    loss_grad_fnc_hand = jax.value_and_grad(mse_hand)

    # FLAX
    for i in range(101):
        # Perform one gradient update.
        loss_val, grads = loss_grad_fnc_flax(params, x_samples, y_samples)
        params = update_params(params, learning_rate, grads)
        if i % 10 == 0:
            print(f'FLAX: Loss step {i}: ', loss_val)


    # NN by Hand
    for i in range(101):
        # Perform one gradient update.
        loss_val, grads = loss_grad_fnc_hand(params_h, x_samples, y_samples)
        params_h = update_params(params_h, learning_rate, grads)
        if i % 10 == 0:
            print(f'Hand: Loss step {i}: ', loss_val)