import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import jax
from jax import random, jit, flatten_util, numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
from typing import Sequence
import os
import sys

# To use the plot_results file we need to add the uppermost folder to the PYTHONPATH
# Only Works if file gets called from 00_Code
sys.path.insert(0, os.getcwd())
from plot_results import plot_results, get_plot_path

'''
Naming Conventions:
    z       refers to the state
    x       refers to the location variable of the state
    v       refers to the velocity variable of the state
    t       refers to time
    f       refers to the ode function
    g       refers to the inner part of the loss function: loss = sum(g) / loss = integral(g)
    d       refers to a total derivative
    del     refers to a partial derivative
    adj     refers to the adjoint state
    phi     collection of physical parameters (kappa, mu, mass)
    theta   collection of neural network parameters
    '''

# this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)


@jit
def ode_stim(z, t, ode_parameters):
    '''Calculates the right hand side of the original ODE.'''
    kappa = ode_parameters[0]
    mu = ode_parameters[1]
    mass = ode_parameters[2]
    derivative = jnp.array([z[1],
                           -kappa*z[0]/mass + (mu*(1-z[0]**2)*z[1])/mass + 1.2*jnp.cos(0.628*t)])
    return derivative

@jit
def hybrid_ode_stim(z, t, ode_parameters, nn_parameters):
    '''Calculates the right hand side of the hybrid ODE, where
    the damping term is replaced by the neural network'''
    kappa = ode_parameters[0]
    mu = ode_parameters[1]
    mass = ode_parameters[2]
    derivative = jnp.array([jnp.array((z[1],)),
                            jnp.array((-kappa*z[0]/mass,)) + model.apply(nn_parameters, z) + jnp.array(1.2*jnp.cos(0.628*t))] ).flatten()
    return derivative

@jit
def ode(z, t, ode_parameters):
    '''Calculates the right hand side of the original ODE.'''
    kappa = ode_parameters[0]
    mu = ode_parameters[1]
    mass = ode_parameters[2]
    derivative = jnp.array([z[1],
                           -kappa*z[0]/mass + (mu*(1-z[0]**2)*z[1])/mass])
    return derivative

@jit
def hybrid_ode(z, t, ode_parameters, nn_parameters):
    '''Calculates the right hand side of the hybrid ODE, where
    the damping term is replaced by the neural network'''
    kappa = ode_parameters[0]
    mu = ode_parameters[1]
    mass = ode_parameters[2]
    derivative = jnp.array([jnp.array((z[1],)),
                            jnp.array((-kappa*z[0]/mass,)) + model.apply(nn_parameters, z)]).flatten()
    return derivative

@jit
def adjoint_ode(adj, z, z_ref, t, ode_parameters, nn_parameters):
    '''Calculates the right hand side of the adjoint system.'''
    df_dz = jax.jacobian(hybrid_ode, argnums=0)(z, t, ode_parameters, nn_parameters)
    dg_dz = jax.grad(g, argnums=0)(z, z_ref, ode_parameters, nn_parameters)
    # d_adj = - df_dz(z, ode_parameters).T @ adj - dg_dz(z, z_ref)
    d_adj = - df_dz.T @ adj - dg_dz
    return d_adj

def g(z, z_ref, ode_parameters, nn_parameters):
    '''Calculates the inner part of the loss function.

    This function can either take individual floats for z
    and z_ref or whole numpy arrays'''
    return jnp.sum(0.5 * (z_ref - z)**2, axis = 0)

def J(z, z_ref, ode_parameters, nn_parameters):
    '''Calculates the complete loss of a trajectory w.r.t. a reference trajectory'''
    return np.sum(g(z, z_ref, ode_parameters, nn_parameters))

def f_euler(z0, t, ode_parameters):
    '''Applies forward Euler to the original ODE and returns the trajectory'''
    z = np.zeros((t.shape[0], 2))
    z[0] = z0
    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        z[i+1] = z[i] + dt * ode(z[i], t[i], ode_parameters)
    return z

def hybrid_euler(z0, t, ode_parameters, nn_parameters):
    '''Applies forward Euler to the hybrid ODE and returns the trajectory'''
    z = np.zeros((t.shape[0], 2))
    z[0] = z0
    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        z[i+1] = z[i] + dt * hybrid_ode(z[i], t[i], ode_parameters, nn_parameters)
    return z

def adj_euler(a0, z, z_ref, t, ode_parameters, nn_parameters):
    '''Applies forward Euler to the adjoint ODE and returns the trajectory'''
    a = np.zeros((t.shape[0], 2))
    a[0] = a0
    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        a[i+1] = a[i] + dt * adjoint_ode(a[i], z[i], z_ref[i], t[i], ode_parameters, nn_parameters)
    return a

# Vectorize the  jacobian df_dtheta for all time points
df_dtheta_function = lambda z, t, phi, theta: jax.jacobian(hybrid_ode, argnums=3)(z, t, phi, theta)
vectorized_df_dtheta_function = jit(jax.vmap(df_dtheta_function, in_axes=(0, 0, None, None)))

# Vectorize the  jacobian dg_dtheta for all time points
dg_dtheta_function = lambda z, z_ref, phi: jax.grad(g, argnums=3)(z, z_ref, phi)
vectorized_dg_dtheta_function = jit(jax.vmap(dg_dtheta_function, in_axes=(0, 0, None, None)))

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


def function_wrapper(nn_parameters, args):
    '''This is a function wrapper for the optimisation function. It returns the
    loss and the jacobian'''

    # Unpack the arguments
    t = args[0]
    z0 = args[1]
    ode_parameters_ref = args[2]
    ode_parameters = args[3]
    unravel_pytree = args[4]
    epoch = args[5]

    # Get the parameters out of the neural network tree structure into an array
    nn_parameters = unravel_pytree(nn_parameters)

    # Calculate the reference solution (could do this out of the the function wrapper)
    z_ref = f_euler(z0, t, ode_parameters_ref)
    # calculate the prediction of the hybrid model and calculate the loss w.r.t. the reference
    z = hybrid_euler(z0, t, ode_parameters, nn_parameters)
    loss = J(z, z_ref, ode_parameters, nn_parameters)

    # adjoint always has the initial condition of all 0s.
    a0 = np.array([0, 0])
    # Calculate the adjoint solution to the problem
    adjoint = adj_euler(a0, np.flip(z, axis=0), np.flip(z_ref, axis=0), np.flip(t), ode_parameters, nn_parameters)
    adjoint = np.flip(adjoint, axis=0)

    # Calculate the gradient of the hybrid ode with respect to the nn_parameters
    df_dtheta_at_t = vectorized_df_dtheta_function(z, t, ode_parameters, nn_parameters)

    # For loop probably not the fastest; Pytree probably better
    # Matrix multiplication of adjoint variable with jacobian
    df_dtheta_at_t = unfreeze(df_dtheta_at_t)

    for layer in df_dtheta_at_t['params']:
        # Sum the matmul result over the entire time_span to get the final gradients
        df_dtheta_at_t['params'][layer]['bias'] = np.einsum("iN,iNj->j", adjoint, df_dtheta_at_t['params'][layer]['bias'])
        df_dtheta_at_t['params'][layer]['kernel'] = np.einsum("iN,iNjk->jk", adjoint, df_dtheta_at_t['params'][layer]['kernel'])

    df_dtheta = df_dtheta_at_t

    dJ_dtheta = df_dtheta

    flat_dJ_dtheta, _ = flatten_util.ravel_pytree(dJ_dtheta)

    print(f'Epoch: {epoch}, Loss: {loss:.5f}')
    epoch += 1
    args[5] = epoch
    return loss, flat_dJ_dtheta

t = np.linspace(0.0, 10.0, 601)
z0 = np.array([1.0, 0.0])
ode_parameters_ref = np.asarray([1.0, 5.0, 1.0])
ode_parameters = np.asarray([1.0, 1.0, 1.0])


layers = [5, 1]
#NN Parameters
key1, key2 = random.split(random.PRNGKey(0), 2)
# Input size is guess from input during init
model = ExplicitMLP(features=layers)
nn_parameters = model.init(key2, np.zeros((1, 2)))
# nn_parameters = unfreeze(nn_parameters)
flat_nn_parameters, unravel_pytree = flatten_util.ravel_pytree(nn_parameters)
epoch = 0

# Put all arguments the optimization needs into one array for the minimize function
args = [t, z0, ode_parameters_ref, ode_parameters, unravel_pytree, epoch]

# Possible methods include: BFGS, SLSQP, L-BFGS-B, CG
method = 'BFGS'
res = minimize(function_wrapper, flat_nn_parameters, method=method, jac=True, args=args, options={'maxiter': 1000})

flat_nn_parameters = res['x']
nn_parameters = unravel_pytree(flat_nn_parameters)

print(res)

z_ref = f_euler(z0, t, ode_parameters_ref)
z = hybrid_euler(z0, t, ode_parameters_ref, nn_parameters)

fig = plt.figure()
x_ax, v_ax = fig.subplots(2,1)
x_ax.set_title('Position')
x_ax.plot(t, z_ref[:,0], label='ref')
v_ax.plot(t, z_ref[:,1], label='ref')
v_ax.set_title('Velocity')
x_ax.plot(t, z[:,0], label='sol')
v_ax.plot(t, z[:,1], label='sol')
x_ax.legend()
v_ax.legend()
fig.tight_layout()

path = os.path.abspath(__file__)
plot_path = get_plot_path(path)
plot_results(t, z, z_ref, plot_path)