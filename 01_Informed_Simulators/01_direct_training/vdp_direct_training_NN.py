import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import random, jit, numpy as jnp
from flax import linen as nn
from typing import Any, Callable, Sequence
from flax.core import freeze, unfreeze
import os
import sys

# To use the plot_results file we need to add the uppermost folder to the PYTHONPATH
# Only Works if file gets called from 00_Code
sys.path.insert(0, os.getcwd())
from plot_results import plot_results, get_plot_path


# The Neural Network structure class
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

# The true ODE
def vdp(z, t, args):
    kappa, mu, m = args
    x = z[0]
    v = z[1]
    return np.array([v, (spring(x, kappa) + damping(x, v, mu))/m])

def spring(x, kappa):
    return -kappa * x

def damping(x, v, mu):
    return mu*(1-x**2)*v

# The hybrid ode where the damping is approximated by the NN
def hybrid_model(z, t, args, model, params):
    kappa, mu, m = args[0]
    x = z[0]
    v = z[1]
    return np.array([v, (spring(x, kappa)/m) + model.apply(params, [x, v])[0]])

# Forward Euler implementation to solve the ode
def euler(fun, z0, t0, t1, t_span, args):
    z = [z0]
    z_old = z0
    t_old = t0
    for t_new in t_span[1:]:
        dt = t_new - t_old
        z_new = z_old + dt * fun(z_old, t_old, *args)
        z.append(z_new)
        t_old = t_new
        z_old = z_new
    return np.array(z)


# The residuals are 1 shorter than the z_ref since we do
# finite differences on the reference values to get the derivatives
def create_residuals(z_ref, t_span, args_ref):
    kappa_ref, mu_ref, m_ref = args_ref
    x_ref = z_ref[:,0]
    v_ref = z_ref[:,1]
    # true_v_dot = vdp(z_ref.T, t_span, args_ref)[1]
    v_dot = (v_ref[1:] - v_ref[:-1])/(t_span[1:] - t_span[:-1])
    residual = v_dot - spring(x_ref, kappa_ref)[:-1]/m_ref
    return residual

# Loss for the Neural Network
def J_NN(params, model, input_batched, output_batched):
    # Define the squared loss for a single pair(input, output)
    def squared_error(input, output):
        pred = model.apply(params, input)
        return jnp.inner(output-pred, output-pred) / 2.0
    return jnp.sum(jax.vmap(squared_error)(input_batched, output_batched), axis=0)

@jit
def update_params(params, learning_rate, grads):
    params = jax.tree_util.tree_map(
    lambda p, g: p - learning_rate * g, params, grads)
    return params

if __name__ == '__main__':
    # ODE Parameters
    args_ref = [[3.0, 5.0, 1.0],]
    t0 = 0.0
    t1 = 10.0
    steps = 401
    t_span = np.linspace(t0, t1, steps)
    z0 = np.array([1.0, 0.0])
    learning_rate = 0.0001


    #NN Parameters
    key1, key2 = random.split(random.PRNGKey(0), 2)
    # Input size is guess from input during init
    model = ExplicitMLP(features=[20, 1])
    params = model.init(key2, np.zeros((1, 2)))
    print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(params)))

    # Create the Reference Solution
    z_ref = euler(vdp, z0, t0, t1, t_span, args_ref)
    # Create the residuals from which the Network is trained
    outputs_batched = create_residuals(z_ref, t_span, *args_ref)
    inputs_batched = z_ref[:-1]

    prd_args = [args_ref, model, params]

    z_prd = euler(hybrid_model, z0, t0, t1, t_span, prd_args)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.plot(t_span, z_ref[:,0], label='Reference Position')
    ax1.plot(t_span, z_ref[:,1], label='Reference Velocity')
    ax1.plot(t_span, z_prd[:,0], label='Prediction Position')
    ax1.plot(t_span, z_prd[:,1], label='Prediction Velocity')
    ax1.legend()
    ax1.grid()
    ax1.set_title('Start')
    


    losses = []
    for epoch in range(4000):
        loss_grad_fnc = jax.value_and_grad(J_NN)
        loss_val, grads = loss_grad_fnc(params, model, inputs_batched, outputs_batched)
        params = update_params(params, learning_rate, grads)
        print(f'Iteration: {epoch}, Loss: {loss_val}')
        losses.append(loss_val)

    prd_args = [args_ref, model, params]
    z_prd = euler(hybrid_model, z0, t0, t1, t_span, prd_args)

    ax2.plot(t_span, z_ref[:,0], label='Reference Position')
    ax2.plot(t_span, z_ref[:,1], label='Reference Velocity')
    ax2.plot(t_span, z_prd[:,0], label='Prediction Position')
    ax2.plot(t_span, z_prd[:,1], label='Prediction Velocity')
    ax2.legend()
    ax2.grid()
    ax2.set_title('Finish')
    
    ax3.plot(losses, label='Learning loss')
    ax3.legend()
    ax3.grid()
    ax3.set_title('Loss')
    plt.show()
    path = os.path.abspath(__file__)
    plot_path = get_plot_path(path)
    fig.savefig(plot_path)


