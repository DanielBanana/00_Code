from jax import jit
from jax import numpy as jnp
from types import SimpleNamespace



@jit
def ode(z, t, variables):
    '''Calculates the right hand side of the original ODE.'''
    kappa = variables['kappa']
    mu = variables['mu']
    mass = variables['mass']
    derivative = jnp.array([z[1],
                           -kappa*z[0]/mass + (mu*(1-z[0]**2))/mass])
    return derivative

@jit
def ode_res(z, t, variables):
    '''Calculates the right hand side of the deficient ODE.'''
    kappa = variables['kappa']
    mu = variables['mu']
    mass = variables['mass']
    derivative = jnp.array([z[1],
                           -kappa*z[0]/mass])
    return derivative

@jit
def ode_stim(z, t, variables):
    '''Calculates the right hand side of the original ODE.'''
    kappa = variables['kappa']
    mu = variables['mu']
    mass = variables['mass']
    derivative = jnp.array([z[1],
                           -kappa*z[0]/mass + (mu*(1-z[0]**2)*z[1])/mass + 1.2*jnp.cos(jnp.pi/5*t)])
    return derivative

@jit
def ode_stim_res(z, t, variables):
    '''Calculates the right hand side of the original ODE.'''
    kappa = variables['kappa']
    mu = variables['mu']
    mass = variables['mass']
    derivative = jnp.array([z[1],
                           -kappa*z[0]/mass + 1.2*jnp.cos(jnp.pi/5*t)])
    return derivative

d = {'ode': ode, 'ode_res': ode_res, 'ode_stim': ode_stim, 'ode_stim_res': ode_stim_res}
VdP = SimpleNamespace(**d)