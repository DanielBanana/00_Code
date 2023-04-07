import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, integrate
import jax
from jax import random, jit, numpy as jnp
from functools import partial
from flax import linen as nn
from typing import Any, Callable, Sequence
from flax.core import freeze, unfreeze
from jax.tree_util import tree_structure
from optax import adam
import optax
from ode import euler, heun
import os
import sys

# To use the plot_results file we need to add the uppermost folder to the PYTHONPATH
# Only Works if file gets called from 00_Code
sys.path.insert(0, os.getcwd())
from plot_results import plot_results, get_plot_path

# this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)

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

@jit
def vdp(z, t, args):
    kappa, mu, m = args
    x = z[0]
    v = z[1]
    return jnp.array([v, (spring(x, kappa) + damping(x, v, mu))/m])

# The hybrid ode where the damping is approximated by the NN
# @partial(jit, static_argnums=(3,))
def hybrid_model(z, t, args, params):
    kappa, mu, m = args[0]
    x = z[0]
    v = z[1]
    return jnp.array([v, (spring(x, kappa)/m) + model.apply(params, z)[0]])

@jit
def adjoint_model(s, t, args, z_ref, t_span):
    """
    This solves a system of 2*N differential equations in reverse, first the
    original ODE system (which is needed since we require the solution to
    evaluate the Jacobian) and then the adjoint system.

    This is the general version, that works if the loss function is evaluated
    over the entire trajectory. It therefore requires the reference solution to
    be given as part of the args, since it requires an interpolation of such.
    Additionally, the temporal mesh on which the reference solution is given has
    to be provided.
    """

    # Unpack the state vector
    z = s.reshape((2, 2))[:, 0]
    adjoint_variable = s.reshape((2, 2))[:, 1]

    # Interpolate the reference solution
    # z_at_current_t_ref = interpolate.interp1d(t, z_ref, axis=-1)(t)
    x_at_current_t_ref = jax.numpy.interp(t, t_span, z_ref[0])
    v_at_current_t_ref = jax.numpy.interp(t, t_span, z_ref[1])

    z_at_current_t_ref = jnp.array([x_at_current_t_ref, v_at_current_t_ref])

    # Form the Jacobian of f wrt to the function parameters
    del_f__del_z = jax.jacobian(hybrid_model, argnums=0)(z, t, *args)

    # Form the gradient of the loss function g wrt to the function parameters
    del_g__del_z = (z - z_at_current_t_ref).T

    original_rhs = hybrid_model(z, t, *args).reshape((-1, 1))

    # calculate the rhs of the adjoint problem
    adjoint_rhs =  (- del_f__del_z.T @ adjoint_variable - del_g__del_z.T).reshape((-1, 1))

    return jnp.concatenate((original_rhs, adjoint_rhs), axis=1).flatten()

def spring(x, kappa):
    return -kappa * x

def damping(x, v, mu):
    return -mu*(1-x**2)*v

def g_entire_trajectory(z_at_t, z_at_t_ref, t_span):
    difference_at_t = (z_at_t - z_at_t_ref)**2
    quadratic_loss_at_t = 0.5 * jnp.sum(difference_at_t, axis=0)
    return integrate.trapezoid(quadratic_loss_at_t, t_span, axis=0)

@jit
def update_params(params, learning_rate, grads):
    params = unfreeze(params)
    params = jax.tree_util.tree_map(
    lambda p, g: p - learning_rate * g, params, grads)
    return freeze(params)

if __name__ == '__main__':
    args_ref = [[3.0, 8.53, 1.0],]
    t0 = 0.0
    t1 = 10.0
    steps = 1601
    t_span = np.linspace(t0, t1, steps)
    z0 = np.array([1.0, 0.0])
    learning_rate_base  = 0.01
    epochs = 1000
    integration_method = heun
    use_optimizer = True
    layers = [20, 1]

    #NN Parameters
    key1, key2 = random.split(random.PRNGKey(0), 2)
    # Input size is guess from input during init
    model = ExplicitMLP(features=layers)
    params = model.init(key2, np.zeros((1, 2)))
    print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, unfreeze(params)))
    optimizer = adam(learning_rate = learning_rate_base)
    opt_state = optimizer.init(unfreeze(params))


    # Create a Reference Solution
    prd_args = [args_ref, params]
    z_ref = integration_method(vdp, z0, t0, t1, t_span, args_ref)
    z_prd = integration_method(hybrid_model, z0, t0, t1, t_span, prd_args)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.plot(t_span, z_ref[0], label='Reference Position')
    ax1.plot(t_span, z_ref[1], label='Reference Velocity')
    ax1.plot(t_span, z_prd[0], label='Prediction Position')
    ax1.plot(t_span, z_prd[1], label='Prediction Velocity')
    ax1.legend()
    ax1.grid()
    ax1.set_title('Start')

    losses = []
    
    # Vector setup for d_f__d_theta jacobian
    dynamic_sensitivity_jacobian = lambda y, t, args, params: jax.jacrev(hybrid_model, argnums=3)(y, t, args, params)
    vectorized_dynamic_sensitivity_jacobian = jit(jax.vmap(dynamic_sensitivity_jacobian, in_axes=(1, 0, None, None), out_axes=1))

    for epoch in range(epochs):
        # lr = learning_rate_base/(np.log(epoch+2))
        prd_args = [args_ref, params]
        # learning_rate = 1/(epoch+1) * learning_rate_base
        learning_rate = learning_rate_base
        z_prd = integration_method(hybrid_model, z0, t0, t1, t_span, prd_args)
        loss = g_entire_trajectory(z_prd, z_ref, t_span)
        losses.append(loss)
        # loss = g_entire_trajectory_with_prediction(z_ref, z0, t0, t1, t_span, theta_prd)
        print(f'Epoch: {epoch}, Loss: {loss:.3f}')
        terminal_condition = np.zeros((2, 2))
        terminal_condition[:, 0] = z_prd[:, -1]
        solution_and_adjoint_variable_at_t = integration_method(adjoint_model,
                                                terminal_condition.flatten(),
                                                t1,
                                                t0,
                                                np.flip(t_span),
                                                args=(prd_args, z_ref, t_span))
        solution_and_adjoint_variable_at_t = np.flip(solution_and_adjoint_variable_at_t.reshape((2, 2, steps)), axis=2)

        solution_variable_at_t = np.flip(solution_and_adjoint_variable_at_t[:, 0, :], axis=1)
        adjoint_variable_at_t = solution_and_adjoint_variable_at_t[:, 1, :]

        # Initial condition did not depend on theta
        d_z0__d_theta = np.zeros((2, 3))

        # Calculate the Jacobian of f with respect to the neural network parameters
        del_f__del_theta__at_t = vectorized_dynamic_sensitivity_jacobian(z_prd, t_span, args_ref, params)

        # Now we need to map the jacobians (there is one matrix for each bias-vector and weight matrix)
        # The dimensions are as follows:
        # i: Dimensions of f
        # N: Number of samples, time_steps
        # j: First dimension of NN part (bias and weights)
        # k: Second dimension of NN part (only weights)
        # bias jacobian: i, N, j
        # weight jacobian: i, N, j, k
        # Resulting gradients should have form N, j and N, j, k
        # kernel = weight
        
        # For loop probably not the fastest; Pytree probably better
        # Matrix multiplication of adjoint variable with jacobian
        del_f__del_theta__at_t = unfreeze(del_f__del_theta__at_t)

        for layer in del_f__del_theta__at_t['params']:
            adjoint_matmul_jacobian_bias = np.einsum("iN,iNj->Nj", adjoint_variable_at_t, del_f__del_theta__at_t['params'][layer]['bias'])
            adjoint_matmul_jacobian_kernel = np.einsum("iN,iNjk->Njk", adjoint_variable_at_t, del_f__del_theta__at_t['params'][layer]['kernel'])

            # Integrate the matmul result over the entire time_span to get the final gradients
            # We save the results in the old jacobian since it already has the right dictionary structure
            del_f__del_theta__at_t['params'][layer]['bias'] = integrate.trapezoid(adjoint_matmul_jacobian_bias, t_span, axis=0) 
            del_f__del_theta__at_t['params'][layer]['kernel'] = integrate.trapezoid(adjoint_matmul_jacobian_kernel, t_span, axis=0) 

        grads = del_f__del_theta__at_t


        # Derivative of loss function with respect to parameters is zero
        # d_g__d_theta = jax.grad(g_entire_trajectory, argnums=None)(z_prd, z_ref, t_span)      

        # params = update_params(params, lr, grads)
        if epoch == 0:
            update_fnc = jit(optimizer.update)
        
        if use_optimizer:
            updates, opt_state = update_fnc(unfreeze(grads), opt_state)
            params = optax.apply_updates(unfreeze(params), unfreeze(updates))
        else:
            params = update_params(params, learning_rate, grads)

        if epoch % 750 == 0 or epoch == epochs-1:
            fig2, (ax4, ax5, ax6, ax7) = plt.subplots(1, 4, figsize=(24, 6))
            ax4.plot(t_span, z_ref[0], label='Reference Position')
            ax4.plot(t_span, z_ref[1], label='Reference Velocity')
            ax4.plot(t_span, z_prd[0], label='Prediction Position')
            ax4.plot(t_span, z_prd[1], label='Prediction Velocity')
            ax4.legend()
            ax4.grid()
            ax4.set_title(f'Epoch: {epoch}')

            ax5.plot(t_span, adjoint_variable_at_t[0], label = 'adjoint, x')
            ax5.plot(t_span, adjoint_variable_at_t[1], label = 'adjoint, v')
            ax5.legend()
            ax5.grid()
            ax5.set_title('Adjoint')

            ax6.plot(t_span, z_prd[0], label = 'Prediction, x')
            ax6.plot(t_span, z_prd[1], label = 'Prediction, v')
            ax6.plot(t_span, np.flip(solution_variable_at_t[0]), label = 'Reverse Prediction, x')
            ax6.plot(t_span, np.flip(solution_variable_at_t[1]), label = 'Reverse Prediction, v')
            ax6.legend()
            ax6.grid()
            
            ax7.plot(losses)
            ax7.grid()
            ax7.set_title('Losses')

            plt.show()

    ax2.plot(t_span, z_ref[0], label='Reference Position')
    ax2.plot(t_span, z_ref[1], label='Reference Velocity')
    ax2.plot(t_span, z_prd[0], label='Prediction Position')
    ax2.plot(t_span, z_prd[1], label='Prediction Velocity')
    ax2.legend()
    ax2.grid()
    ax2.set_title('Final')

    ax3.plot(losses, label='Learning loss')
    ax3.legend()
    ax3.grid()
    ax3.set_title('Losses')
    plt.show()
    fig.savefig(f'adjoint_NN_mu_({args_ref[0][1]})_{steps}.png')


    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(t_span, z_prd[0], label = 'Prediction, x')
    ax1.plot(t_span, z_prd[1], label = 'Prediction, v')
    ax1.plot(t_span, np.flip(solution_variable_at_t[0]), label = 'Reverse Prediction, x')
    ax1.plot(t_span, np.flip(solution_variable_at_t[1]), label = 'Reverse Prediction, v')
    ax1.legend()
    ax1.grid()

    ax2.plot(t_span, adjoint_variable_at_t[0], label = 'adjoint, x')
    ax2.plot(t_span, adjoint_variable_at_t[1], label = 'adjoint, v')
    ax2.legend()
    ax2.grid()
    fig2.savefig(f'adjoint_NN_mu_({args_ref[0][1]})_{steps}_extra.png')