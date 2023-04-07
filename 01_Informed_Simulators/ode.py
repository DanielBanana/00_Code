import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit
import numpy as np
from functools import partial


@partial(jit, static_argnums=(0,))
def euler_step(fun, z_old, t_old, dt, args):
    z_new = z_old + dt * fun(z_old, t_old, *args)
    return z_new

def euler(fun, z0, t0, t1, t, args):
    z = np.zeros((t.shape[0], 2))
    z[0] = z0
    args[0].append(0)
    # z_old = z0
    # t_old = t0
    for i in range(len(t)-1):
        args[0][-1] = i
        dt = t[i+1] - t[i]
        z[i+1] = euler_step(fun, z[i], t[i], dt, args)
    args[0].pop(-1)
    return jnp.array(z).T


@partial(jit, static_argnums=(0,))
def heun_step(fun, z_old, t_old, t_new, args):
    dt = t_new - t_old
    z_temp = fun(z_old, t_old, *args)
    z_1 = z_old + dt * z_temp
    z_new = z_old + dt/2 * (z_temp + fun(z_1, t_new, *args)) 
    return z_new
def heun(fun, z0, t0, t1, t_span, args):
    z = [z0]
    z_old = z0
    t_old = t0
    for t_new in t_span[1:]:
        z_new = heun_step(fun, z_old, t_old, t_new, args)
        z.append(z_new)
        t_old = t_new
        z_old = z_new
    return jnp.array(z).T