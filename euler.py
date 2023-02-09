import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit
import numpy as np


def euler(fun, t_span, y0, t_eval, args):
    step_size = t_eval[1] - t_eval[0]
    y = jnp.array(y0)
    sol = [y]
    for time in t_eval[:-1]:
        y_next = euler_step(fun, y, time, step_size, args)
        sol.append(y_next)
        y = y_next
    return jnp.array(sol).T

def euler_step(fun, y, time, step_size, args):
    return y + step_size * fun(time, y, args)

def euler_NN(fun, t_span, y0, t_eval, params, args):
    step_size = t_eval[1] - t_eval[0]
    y = jnp.array(y0)
    sol = [y]
    for time in t_eval[:-1]:
        y_next = euler_step_NN(fun, y, time, step_size, params, args)
        sol.append(y_next)
        y = y_next
    return jnp.array(sol).T

def euler_step_NN(fun, y, time, step_size, params, args):
    return y + step_size * fun(time, y, params, args)