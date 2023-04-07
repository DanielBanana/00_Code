import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import os
import sys

# To use the plot_results file we need to add the uppermost folder to the PYTHONPATH
# Only Works if file gets called from 00_Code
sys.path.insert(0, os.getcwd())
from plot_results import plot_results, get_plot_path

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


def ode(z, t, ode_parameters):
    '''Calculates the right hand side of the original ODE.'''
    kappa = ode_parameters[0]
    mu = ode_parameters[1]
    mass = ode_parameters[2]
    derivative = np.array([z[1],
                           -kappa*z[0]/mass + (mu*(1-z[0]**2)*z[1])/mass])
    return derivative

def adjoint_ode(adj, z, z_ref, t, ode_parameters):
    '''Calculates the right hand side of the adjoint system.'''
    d_adj = - df_dz(z, ode_parameters).T @ adj - dg_dz(z, z_ref)
    return d_adj

def df_dz(z, ode_parameters):
    '''Calculates the jacobian of f w.r.t. z derived by hand.'''
    kappa = ode_parameters[0]
    mu = ode_parameters[1]
    mass = ode_parameters[2]
    df1_dz = np.array([0, 1])
    df2_dz = np.array([-kappa/mass - float(2*mu*z[0]*z[1])/mass,
                      float(mu*(1-z[0]**2))/mass])
    return np.array([df1_dz, df2_dz])

def df_dphi(z, ode_parameters):
    '''Calculates the derivative of f w.r.t. the physical parameters phi.'''
    kappa = ode_parameters[0]
    mu = ode_parameters[1]
    mass = ode_parameters[2]
    df_dkappa = np.array([0, -z[0]/mass])
    df_dmu = np.array([0, (1-z[0]**2)*z[1]/mass])
    df_dmass = np.array([0, + kappa*z[0]/mass**2 - mu*(1-z[0]**2)*z[1]/mass**2])
    return np.vstack([df_dkappa, df_dmu, df_dmass])

def g(z, z_ref):
    '''Calculates the inner part of the loss function.
    
    This function can either take individual floats for z
    and z_ref or whole numpy arrays'''
    return np.sum(0.5 * (z_ref - z)**2, axis=1)

def dg_dz(z, z_ref):
    '''Calculates the derivative of g w.r.t. z, derived by hand.'''
    return z - z_ref

def dg_dphi(z, z_ref):
    return 0.0

def J(z, z_ref):
    '''Calculates the complete loss of a trajectory w.r.t. a reference trajectory'''
    return np.sum(g(z, z_ref))

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

def function_wrapper(ode_parameters, args):
    '''This is a function wrapper for the optimisation function. It returns the 
    loss and the jacobian'''
    t = args[0]
    z0 = args[1]
    ode_parameters_ref = args[2]

    z_ref = f_euler(z0, t, ode_parameters_ref)
    z = f_euler(z0, t, ode_parameters)
    # plt.plot(t, z_ref)
    # plt.plot(t, z)
    # plt.show()
    print(ode_parameters)

    a0 = np.array([0, 0])
    adjoint = adj_euler(a0, np.flip(z, axis=0), np.flip(z_ref, axis=0), np.flip(t), ode_parameters)
    adjoint = np.flip(adjoint, axis=0)

    df_dphi_at_t = []
    for z_at_t in z:
        df_dphi_at_t.append(df_dphi(z_at_t, ode_parameters))
    df_dphi_at_t = np.array(df_dphi_at_t)
    loss = J(z, z_ref)
    dJ_dphi = 0
    # for i in range(t.shape[0]):
    #     dJ_dmu += dg_dmu(z, z_ref) + a.T[:,i] @ df_dphi_at_t[i]
    # # a = np.expand_dims(a, 1)
    
    # if there is only one parameter to optimise we need to manually add dimension here
    if len(df_dphi_at_t.shape) == 2:
        df_dphi_at_t = np.expand_dims(df_dphi_at_t, 1)
        dJ_dphi = float(np.einsum("Ni,Nji->j", adjoint, df_dphi_at_t))
    else:
        dJ_dphi = np.einsum("Ni,Nji->j", adjoint, df_dphi_at_t)

    return loss, dJ_dphi

t = np.linspace(0.0, 10.0, 801)
z0 = np.array([1.0, 0.0])
ode_parameters_ref = np.asarray([3.0, 5.0, 1.0])
ode_parameters = np.asarray([1.0, 1.0, 1.0])
args = [t, z0, ode_parameters_ref]

# Get Reference Solution
z_ref = f_euler(z0, t, ode_parameters_ref)
res = minimize(function_wrapper, ode_parameters, method='BFGS', jac=True, args=args)
print(res)

# Plot approximated mu solution
ode_parameters = res.x
z = f_euler(z0, t, ode_parameters)

path = os.path.abspath(__file__)
plot_path = get_plot_path(path)
plot_results(t, z, z_ref, plot_path)
