import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import jax
from jax import jit, numpy as jnp

'''
Naming Conventions:
    z   refers to the state
    x   refers to the location variable of the state
    v   refers to the velocity variable of the state
    t   refers to time
    f   refers to the ode function
    g   refers to the inner part of the loss function: loss = sum(g) / loss = integral(g)
    d   refers to a total derivative
    del refers to a partial derivative
    adj refers to the adjoint state
    phi collection of physical parameters (kappa, mu, mass)
    '''

@jit
def ode(z, t, ode_parameters):
    '''Calculates the right hand side of the original ODE.'''
    kappa = ode_parameters[0]
    mu = ode_parameters[1]
    mass = ode_parameters[2]
    derivative = jnp.array([z[1],
                           -kappa*z[0]/mass - (mu*(1-z[0]**2)*z[1])/mass])
    return derivative

@jit
def adjoint_ode(adj, z, z_ref, t, ode_parameters):
    '''Calculates the right hand side of the adjoint system.'''
    df_dz = jax.jacobian(ode, argnums=0)(z, t, ode_parameters)
    dg_dz = jax.grad(g, argnums=0)(z, z_ref, ode_parameters)
    # d_adj = - df_dz(z, ode_parameters).T @ adj - dg_dz(z, z_ref)
    d_adj = - df_dz.T @ adj - dg_dz
    return d_adj

def df_dphi(z, ode_parameters):
    '''Calculates the derivative of f w.r.t. the physical parameters phi.'''
    kappa = ode_parameters[0]
    mu = ode_parameters[1]
    mass = ode_parameters[2]
    df_dkappa = np.array([0, -z[0]/mass])
    df_dmu = np.array([0, -(1-z[0]**2)*z[1]/mass])
    df_dmass = np.array([0, + kappa*z[0]/mass**2 + mu*(1-z[0]**2)*z[1]/mass**2])
    return np.vstack([df_dkappa, df_dmu, df_dmass])

def g(z, z_ref, ode_parameters):
    '''Calculates the inner part of the loss function.
    
    This function can either take individual floats for z
    and z_ref or whole numpy arrays'''
    return jnp.sum(0.5 * (z_ref - z)**2, axis = 0)

def J(z, z_ref, ode_parameters):
    '''Calculates the complete loss of a trajectory w.r.t. a reference trajectory'''
    return np.sum(g(z, z_ref, ode_parameters))

def f_euler(z0, t, ode_parameters):
    '''Applies forward Euler to the original ODE and returns the trajectory'''
    z = np.zeros((t.shape[0], 2))
    z[0] = z0
    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        z[i+1] = z[i] + dt * ode(z[i], t[i], ode_parameters)
    return z

def adj_euler(a0, z, z_ref, t, ode_parameters):
    '''Applies forward Euler to the adjoint ODE and returns the trajectory'''
    a = np.zeros((t.shape[0], 2))
    a[0] = a0
    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        a[i+1] = a[i] + dt * adjoint_ode(a[i], z[i], z_ref[i], t[i], ode_parameters)
    return a

# Vectorize the  jacobian df_dphi for all time points
df_dphi_function = lambda z, t, phi: jnp.array(jax.jacobian(ode, argnums=2)(z, t, phi))
vectorized_df_dphi_function = jit(jax.vmap(df_dphi_function, in_axes=(0, 0, None)))

# Vectorize the  jacobian dg_dphi for all time points
dg_dphi_function = lambda z, z_ref, phi: jnp.array(jax.grad(g, argnums=2)(z, z_ref, phi))
vectorized_dg_dphi_function = jit(jax.vmap(dg_dphi_function, in_axes=(0, 0, None)))


def function_wrapper(ode_parameters, args):
    '''This is a function wrapper for the optimisation function. It returns the 
    loss and the jacobian'''
    t = args[0]
    z0 = args[1]
    ode_parameters_ref = args[2]

    z_ref = f_euler(z0, t, ode_parameters_ref)
    z = f_euler(z0, t, ode_parameters)
    loss = J(z, z_ref, ode_parameters)
    # plt.plot(t, z_ref)
    # plt.plot(t, z)
    # plt.show()
    print(ode_parameters)

    a0 = np.array([0, 0])
    adjoint = adj_euler(a0, np.flip(z, axis=0), np.flip(z_ref, axis=0), np.flip(t), ode_parameters)
    adjoint = np.flip(adjoint, axis=0)

    df_dphi_at_t = vectorized_df_dphi_function(z, t, ode_parameters)
    # if there is only one parameter to optimise we need to manually add dimension here
    if len(df_dphi_at_t.shape) == 2:
        df_dphi_at_t = jnp.expand_dims(df_dphi_at_t, 2)
        df_dphi = float(np.einsum("Ni,Nij->j", adjoint, df_dphi_at_t))
    else:
        df_dphi = jnp.einsum("Ni,Nij->j", adjoint, df_dphi_at_t)

    # This evaluates only to zeroes, but for completeness sake
    dg_dphi_at_t = vectorized_dg_dphi_function(z, z_ref, ode_parameters)
    dg_dphi = jnp.einsum("Ni->i", dg_dphi_at_t)
    
    dJ_dphi = dg_dphi + df_dphi

    return loss, dJ_dphi

t = np.linspace(0.0, 10.0, 401)
z0 = np.array([1.0, 0.0])
ode_parameters_ref = np.asarray([3.0, 8.53, 1.0])
ode_parameters = np.asarray([1.0, 1.0, 1.0])
args = [t, z0, ode_parameters_ref]

# Sometimes either boundaries have to be defined or the numer of steps has to be increased.
res = minimize(function_wrapper, ode_parameters, method='BFGS', jac=True, args=args, bounds=[(0, 10), (0, 10), (0.1, 10)])
print(res)