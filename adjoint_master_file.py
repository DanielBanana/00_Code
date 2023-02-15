import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


def f(z, t, mu):
    '''Calculates the right hand side of the original ODE.'''
    x = z[1]
    v = -3*z[0] - float(mu*(1-z[0]**2)*z[1])
    return np.array([x, v])

def adj(a, z, z_ref, t, mu):
    '''Calculates the right hand side of the adjoint system.'''
    da = - df_dz(z, mu).T @ a - dg_dz(z, z_ref)
    return da

def df_dz(z, mu):
    '''Calculates the jacobian of f w.r.t. z derived by hand.'''
    df_dx = np.array([0, 1])
    df_dv = np.array([-3 + float(2*mu*z[0]*z[1]), float(-mu*(1-z[0]**2))])
    return np.array([df_dx, df_dv])

def df_dmu(z):
    '''Calculates the derivative of f w.r.t. the damping parameter mu.'''
    return np.array([0, -(1-z[0]**2)*z[1]])

def g(z, z_ref):
    '''Calculates the inner part of the loss function.
    
    This function can either take individual floats for z
    and z_ref or whole numpy arrays'''
    return 0.5 * (z_ref - z)**2

def dg_dz(z, z_ref):
    '''Calculates the derivative of g w.r.t. z, derived by hand.'''
    return z - z_ref

def dg_dmu(z, z_ref):
    return 0.0

def J(z, z_ref):
    '''Calculates the complete loss of a trajectory w.r.t. a reference trajectory'''
    return np.sum(g(z, z_ref))

def f_euler(z0, t, mu):
    '''Applies forward Euler to the original ODE and returns the trajectory'''
    z = np.zeros((t.shape[0], 2))
    z[0] = z0
    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        z[i+1] = z[i] + dt * f(z[i], t[i], mu)
    return z

def adj_euler(a0, z, z_ref, t, mu):
    '''Applies forward Euler to the adjoint ODE and returns the trajectory'''
    a = np.zeros((t.shape[0], 2))
    a[0] = a0
    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        a[i+1] = a[i] + dt * adj(a[i], z[i], z_ref[i], t[i], mu)
    return a


def function_wrapper(mu):
    '''This is a function wrapper for the optimisation function. It returns the 
    loss and the jacobian'''
    t = np.linspace(0.0, 10.0, 101)
    z0 = np.array([1.0, 0.0])
    mu_ref = 8.53

    z_ref = f_euler(z0, t, mu_ref)
    z = f_euler(z0, t, mu)

    a0 = np.array([0, 0])
    adjoint = adj_euler(a0, np.flip(z, axis=0), np.flip(z_ref, axis=0), np.flip(t), mu)
    adjoint = np.flip(adjoint, axis=0)

    df_dmu_at_t = []
    for z_at_t in z:
        df_dmu_at_t.append(df_dmu(z_at_t))
    df_dmu_at_t = np.array(df_dmu_at_t)
    loss = J(z, z_ref)
    dJ_dmu = 0
    # for i in range(t.shape[0]):
    #     dJ_dmu += dg_dmu(z, z_ref) + a.T[:,i] @ df_dmu_at_t[i]
    # # a = np.expand_dims(a, 1)
    df_dmu_at_t = np.expand_dims(df_dmu_at_t, 1)
    dJ_dmu = float(np.einsum("Ni,Nji->j", adjoint, df_dmu_at_t))

    return loss, dJ_dmu

mu_init = 1.0
res = minimize(function_wrapper, mu_init, method='BFGS', jac=True)
# Nelder-Mead is a good alternative if BFGS fails
print(res)