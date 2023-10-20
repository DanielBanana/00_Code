from jax import jit
from jax import numpy as jnp
import numpy as np
from types import SimpleNamespace



@jit
def ode(z, t, variables):
    '''Calculates the right hand side of the original ODE.'''

    q = variables['q']  # L/min
    cA_i = variables['cA_i']  # mol/L
    T_i = variables['T_i']  # K
    V = variables['V']  # L
    rho = variables['rho'] # g/L
    C = variables['C'] # J/(g K)
    Hr = variables['Hr']  # J/(g K)
    E_over_R = variables['E_over_R']  # K
    k0 = variables['k0']  # 1/min
    UA = variables['UA']  # J/(min K)
    Tc = variables['Tc']

    k = k0*jnp.exp(-E_over_R/z[1])
    w = q*rho
    derivative = jnp.array([q*(cA_i - z[0])/V - k*z[0],
                            1/(V*rho*C)*(w*C*(T_i - z[1]) - Hr*V*k*z[0] + UA*(Tc - z[1]))])
    return derivative

@jit
def ode_res(z, t, variables):
    '''Calculates the right hand side of the deficient ODE.'''
    q = variables['q']  # L/min
    cA_i = variables['cA_i']  # mol/L
    T_i = variables['T_i']  # K
    V = variables['V']  # L
    rho = variables['rho'] # g/L
    C = variables['C'] # J/(g K)
    Hr = variables['Hr']  # J/(g K)

    UA = variables['UA']  # J/(min K)
    Tc = variables['Tc']

    w = q*rho
    derivative = jnp.array([q*(cA_i - z[0])/V * z[0],
                            1/(V*rho*C)*(w*C*(T_i - z[1]) - Hr*V*z[0]  + UA*(Tc - z[1]))])
    return derivative

@jit
def ode_hybrid(z, t, variables, parameters, model_function):
    '''Calculates the right hand side of the hybrid ODE, where
    the damping term is replaced by the neural network'''
    q = variables['q']  # L/min
    cA_i = variables['cA_i']  # mol/L
    T_i = variables['T_i']  # K
    V = variables['V']  # L
    rho = variables['rho'] # g/L
    C = variables['C'] # J/(g K)
    Hr = variables['Hr']  # J/(g K)

    UA = variables['UA']  # J/(min K)
    Tc = variables['Tc']

    k = model_function(parameters, z)
    w = q*rho
    derivative = jnp.array([q*(cA_i - z[0])/V - k * z[0],
                            1/(V*rho*C)*(w*C*(T_i - z[1]) - Hr*V*k*z[0] + UA*(Tc - z[1]))]).flatten()
    return derivative

# For calculation of the reference solution we need the correct behaviour of the VdP
def missing_terms(parameters, inputs):
    q = parameters['q']  # L/min
    cA_i = parameters['cA_i']  # mol/L
    T_i = parameters['T_i']  # K
    V = parameters['V']  # L
    rho = parameters['rho'] # g/L
    C = parameters['C'] # J/(g K)
    Hr = parameters['Hr']  # J/(g K)
    E_over_R = parameters['E_over_R']  # K
    k0 = parameters['k0']  # 1/min
    UA = parameters['UA']  # J/(min K)
    Tc = parameters['Tc']
    return k0*jnp.exp(-E_over_R/inputs[0])

def zero(parameters, inputs):
    return 0.0

d = {'ode': ode,
     'ode_res': ode_res,
    #  'ode_stim': ode_stim,
    #  'ode_stim_res': ode_stim_res,
     'ode_hybrid': ode_hybrid,
    #  'ode_hybrid_stim': ode_hybrid_stim,
     'missing_terms': missing_terms,
     'zero': zero}

CSTR = SimpleNamespace(**d)
