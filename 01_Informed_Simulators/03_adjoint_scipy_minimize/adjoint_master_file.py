import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import os
import sys

# To use the plot_results file we need to add the uppermost folder to the PYTHONPATH
# Only Works if file gets called from the 00_Code directory
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


def f(z, t, mu):
    '''Calculates the right hand side of the original ODE. In this case the 
    Van der Pol Oscilator'''
    x = z[1]
    v = -1.0*z[0] + float(mu*(1-z[0]**2)*z[1])
    return np.array([x, v])

def adj(a, z, z_ref, t, mu):
    '''Calculates the right hand side of the adjoint system.'''
    da = - df_dz(z, mu).T @ a - dg_dz(z, z_ref)
    return da

def df_dz(z, mu):
    '''Calculates the jacobian of f w.r.t. z derived by hand.'''
    df1_dz = np.array([0, 1])
    df2_dz = np.array([-1.0 - float(2*mu*z[0]*z[1]), float(mu*(1-z[0]**2))])
    return np.array([df1_dz, df2_dz])

def df_dmu(z):
    '''Calculates the derivative of f w.r.t. the damping parameter mu.'''
    return np.array([0, (1-z[0]**2)*z[1]])

def g(z, z_ref):
    '''Calculates the inner part of the loss function.
    
    This function can either take individual floats for z
    and z_ref or whole numpy arrays'''
    return 0.5 * (z_ref - z)**2

def dg_dz(z, z_ref):
    '''Calculates the derivative of g w.r.t. z.'''
    return z - z_ref

def dg_dmu(z, z_ref):
    '''Calculates the derivative of g w.r.t. mu.'''
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

def function_wrapper(mu, args):
    '''This is a function wrapper for the optimisation function. It returns the 
    loss and the jacobian'''

    t = args[0]
    z0 = args[1]
    z_ref = args[2]

    # Calculate the current prediction for the trajectory
    z = f_euler(z0, t, mu)

    # Calculate the adjoint solution. Starting with the initial condition 0, 0
    # For that we need to flip all the ingoing trajectories since we calculate the adjoint backward
    a0 = np.array([0, 0])
    adjoint = adj_euler(a0, np.flip(z, axis=0), np.flip(z_ref, axis=0), np.flip(t), mu)
    # Flip the solution back since we need to
    adjoint = np.flip(adjoint, axis=0)

    df_dmu_at_t = []
    for z_at_t in z:
        df_dmu_at_t.append(df_dmu(z_at_t))
    df_dmu_at_t = np.array(df_dmu_at_t)
    loss = J(z, z_ref)
    df_dmu_at_t = np.expand_dims(df_dmu_at_t, 1)
    dJ_dmu = float(np.einsum("Ni,Nji->j", adjoint, df_dmu_at_t))

    print(f'mu: {mu}, Loss: {loss}')
    return loss, dJ_dmu

# Define Problem
mu_ref = 8.53
mu_init = 1.0
z0 = np.array([1.0, 0.0])
t = np.linspace(0.0, 10.0, 801)

# Get Reference Solution
z_ref = f_euler(z0, t, mu_ref)


# Optimise mu
args = [t, z0, z_ref]
res = minimize(function_wrapper, mu_init, method='SLSQP', jac=True, args=args)
print(res)

# Plot approximated mu solution
mu = res.x
z = f_euler(z0, t, mu)


path = os.path.abspath(__file__)
plot_path = get_plot_path(path)
plot_results(t, z, z_ref, plot_path)

