from jax import jit
from jax import numpy as jnp
import numpy as np
from types import SimpleNamespace



@jit
def ode(z, t, variables):
    '''Calculates the right hand side of the original ODE.'''
    kappa = variables['kappa']
    mu = variables['mu']
    mass = variables['mass']
    derivative = jnp.array([z[1],
                           -kappa*z[0]/mass + (mu*(1-z[0]**2))*z[1]/mass])
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

def ode_hybrid(z, t, variables, parameters, model_function):
    '''Calculates the right hand side of the hybrid ODE, where
    the damping term is replaced by the neural network'''
    kappa = variables['kappa']
    mu = variables['mu']
    mass = variables['mass']
    derivative = jnp.array([jnp.array((z[1],)),
                 jnp.array((-kappa*z[0]/mass,)) + model_function(parameters, z)]).flatten()
    return derivative

def ode_hybrid_stim(z, t, variables, parameters, model_function):
    '''Calculates the right hand side of the hybrid ODE, where
    the damping term is replaced by the neural network'''
    kappa = variables['kappa']
    mu = variables['mu']
    mass = variables['mass']
    derivative = np.array([np.array((z[1],)),
                            np.array((-kappa*z[0]/mass,)) + model_function(parameters, z) + np.array(1.2*np.cos(np.pi/5*t))]).flatten()
    return derivative

# For calculation of the reference solution we need the correct behaviour of the VdP
def missing_terms(parameters, inputs):
    kappa = parameters['kappa']
    mu = parameters['mu']
    mass = parameters['mass']
    return mu * (1 - inputs[0]**2) * inputs[1]

def zero(parameters, inputs):
    return 0.0

d = {'ode': ode,
     'ode_res': ode_res,
     'ode_stim': ode_stim,
     'ode_stim_res': ode_stim_res,
     'ode_hybrid': ode_hybrid,
     'ode_hybrid_stim': ode_hybrid_stim,
     'missing_terms': missing_terms,
     'zero': zero}

VdP = SimpleNamespace(**d)
