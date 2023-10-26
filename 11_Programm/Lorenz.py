from jax import jit
from jax import numpy as jnp
import numpy as np
from types import SimpleNamespace



@jit
def ode(z, t, variables):
    '''Calculates the right hand side of the original ODE.'''
    sigma = variables['sigma']
    rho = variables['rho']
    beta = variables['beta']
    derivative = jnp.array([sigma*(z[1] - z[0]),
                           z[0]*(rho - z[2]) - z[1],
                           z[0]*z[1] - beta * z[2]])
    return derivative

@jit
def ode_res(z, t, variables):
    '''Calculates the right hand side of the deficient ODE.'''
    sigma = variables['sigma']
    rho = variables['rho']
    beta = variables['beta']
    derivative = jnp.array([sigma*(z[1] - z[0]),
                           z[0]*(rho - z[2]) - z[1],
                           z[0]*z[1] - z[2]])
    return derivative



def ode_hybrid(z, t, variables, parameters, model_function):
    '''Calculates the right hand side of the hybrid ODE, where
    the damping term is replaced by the neural network'''
    sigma = variables['sigma']
    rho = variables['rho']
    beta = variables['beta']
    derivative = jnp.array([jnp.array(sigma*(z[1] - z[0]),),
                 jnp.array(z[0]*(rho - z[2]) - z[1],),
                 jnp.array(z[0]*z[1]) - z[2]*model_function(parameters, z)]).flatten()
    return derivative



# For calculation of the reference solution we need the correct behaviour of the VdP
def missing_terms(parameters, inputs):
    sigma = parameters['sigma']
    rho = parameters['rho']
    beta = parameters['beta']
    return beta

def zero(parameters, inputs):
    return 0.0

d = {'ode': ode,
     'ode_res': ode_res,
     'ode_hybrid': ode_hybrid,
     'missing_terms': missing_terms,
     'zero': zero}

Lorenz = SimpleNamespace(**d)
