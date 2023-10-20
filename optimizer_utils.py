import jax
from jax import jit, numpy as jnp
import numpy as np

@jit
def g(z, z_ref, model_parameters):
    '''Calculates the inner part of the loss function.

    This function can either take individual floats for z
    and z_ref or whole numpy arrays'''
    return jnp.mean(0.5 * (z_ref - z)**2, axis = 0)

@jit
def J(z, z_ref, optimisation_parameters):
    '''Calculates the complete loss of a trajectory w.r.t. a reference trajectory'''
    return np.mean(g(z, z_ref, optimisation_parameters))

def create_residuals(z_ref, t, variables, ode_res):
    z_dot = (z_ref[1:] - z_ref[:-1])/(t[1:] - t[:-1]).reshape(-1,1)
    v_ode = jax.vmap(lambda z_ref, t, variables: ode_res(z_ref, t, variables), in_axes=(0, 0, None))
    residual = z_dot - v_ode(z_ref[:-1], t[:-1], variables)
    return residual

def create_residuals_fmu(z_ref, t, z_dot_fmu):
    z_dot = (z_ref[1:] - z_ref[:-1])/(t[1:] - t[:-1]).reshape(-1,1)
    residual = z_dot - z_dot_fmu
    return np.asarray(residual)