import numpy as np
import matplotlib.pyplot as plt
# import jax
import os
import sys

# To use the plot_results file we need to add the uppermost folder to the PYTHONPATH
# Only Works if file gets called from 00_Code
sys.path.insert(0, os.getcwd())
from plot_results import plot_results, get_plot_path


def vdp(z, t, args):
    """
    The Van der Pol Oscillator is a ODE of Order 2, which is transformed into a system of ODEs of order 1.
    We want to model this system and it represents a function we usually not know. It's used to generate
    reference data.
    """
    kappa, mu, m = args
    x = z[0]
    v = z[1]
    return np.array([v, (spring(x, kappa) + damping(x, v, mu))/m])

def spring(x, kappa):
    """
    Part of the VdP model responsible for the spring part.
    """
    return -kappa * x

def damping(x, v, mu):
    """
    Part of the VdP model responsible for the damping of the spring oscillation.
    """
    return mu*(1-x**2)*v

def damping_dmu(x, v, mu):
    return (1-x**2)*v

def euler(fun, z0, t0, t1, t_span, args):
    """
    Implements the forward euler method to solve systems of ODEs. Returns the entire trajectory as np.array
    """
    z = [z0]
    z_old = z0
    t_old = t0
    for t_new in t_span[1:]:
        dt = t_new - t_old
        z_new = z_old + dt * fun(z_old, t_old, args)
        z.append(z_new)
        t_old = t_new
        z_old = z_new
    return np.array(z)

def J(z_ref, t_span, args_ref, mu_prd):
    """
    The direct Training loss function. The ML model should play the part of the damping part from above.
    Here we train the model on the residual between the reference velocity derivative and the part acounted
    for by the spring part itself.
    """
    kappa, mu, m = args_ref
    x_ref = z_ref[:,0]
    v_ref = z_ref[:,1]
    # true_v_dot = vdp(z_ref.T, t_span, args_ref)[1]
    v_dot = (v_ref[1:] - v_ref[:-1])/(t_span[1:] - t_span[:-1])
    residual = v_dot - spring(x_ref, kappa)[:-1]/m
    prd = damping(x_ref, v_ref, mu_prd)[:-1]/m
    return 0.5 * np.sum((residual - prd)**2)

def dJ_dmu(z_ref, args_ref, t_span, mu_prd):
    kappa, __, m = args_ref
    x_ref = z_ref[:,0]
    v_ref = z_ref[:,1]
    v_dot = (v_ref[1:] - v_ref[:-1])/(t_span[1:] - t_span[:-1])
    residual = v_dot - spring(x_ref, kappa)[:-1]/m - damping(x_ref, v_ref, mu_prd)[:-1]/m
    # residual = (v_ref[1:] - v_ref[:-1])/(t_span[1:] - t_span[:-1]) - (spring(x_ref, kappa)[:-1]/m + damping(x_ref, v_ref, mu_prd)[:-1]/m)
    # return np.sum(residual - ((damping_dmu(x_ref, v_ref, mu_prd))/m)[:-1])
    return np.sum(residual)

if __name__ == '__main__':
    args = [3.0, 8.53, 1.0]
    args_prd = [3.0, 1.0, 1.0]
    t0 = 0.0
    t1 = 10.0
    steps = 801
    t_span = np.linspace(t0, t1, steps)
    z0 = np.array([1.0, 0.0])

    mu = args_prd[1]
    z_ref = euler(vdp, z0, t0, t1, t_span, args)
    z_prd = euler(vdp, z0, t0, t1, t_span, args_prd)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.plot(t_span, z_ref[:,0], label='Reference Position')
    ax1.plot(t_span, z_ref[:,1], label='Reference Velocity')
    ax1.plot(t_span, z_prd[:,0], label='Prediction Position')
    ax1.plot(t_span, z_prd[:,1], label='Prediction Velocity')
    ax1.legend()
    ax1.grid()
    ax1.set_title('Start')

    print(f'Loss: {J(z_ref, t_span, args, args_prd[1])}')
    print(f'Gradient: {dJ_dmu(z_ref, args, t_span, mu)}')
    

    lr = 0.001
    losses = []
    for epoch in range(4000):
        loss = J(z_ref, t_span, args, mu)
        gradient = dJ_dmu(z_ref, args, t_span, mu)
        mu = mu - lr * gradient
        print(f'Loss: {loss}')
        print(f'Gradient: {gradient}')
        print(f'Mu: {mu}')
        losses.append(loss)

    args_prd[1] = mu
    z_prd = euler(vdp, z0, t0, t1, t_span, args_prd)
    ax2.plot(t_span, z_ref[:,0], label='Reference Position')
    ax2.plot(t_span, z_ref[:,1], label='Reference Velocity')
    ax2.plot(t_span, z_prd[:,0], label='Prediction Position')
    ax2.plot(t_span, z_prd[:,1], label='Prediction Velocity')
    ax2.legend()
    ax2.grid()
    ax2.set_title('Finish')

    ax3.plot(losses, label='Learning loss')
    ax3.legend()
    ax3.grid()
    ax3.set_title('Loss')
    plt.show()
    
    path = os.path.abspath(__file__)
    plot_path = get_plot_path(path)
    fig.savefig(plot_path)


