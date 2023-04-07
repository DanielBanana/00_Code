import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, integrate
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import optax
from optax import adam
from ode import euler, heun
import os
import sys

# To use the plot_results file we need to add the uppermost folder to the PYTHONPATH
# Only Works if file gets called from 00_Code
sys.path.insert(0, os.getcwd())
from plot_results import plot_results, get_plot_path

# this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)

@jit
def vdp(z, t, kappa, mu, m):
    # kappa, mu, m = args
    x = z[0]
    v = z[1]
    return jnp.array([v, (spring(x, kappa) + damping(x, v, mu))/m])

@jit
def adjoint_model(s, t, args, z_ref, t_span):
    """
    This solves a system of 2*N differential equations in reverse, first the
    original ODE system (which is needed since we require the solution to
    evaluate the Jacobian) and then the adjoint system.

    This is the general version, that works if the loss function is evaluated
    over the entire trajectory. It therefore requires the reference solution to
    be given as part of the args, since it requires an interpolation of such.
    Additionally, the temporal mesh on which the reference solution is given has
    to be provided.
    """
    kappa, mu, m = args
    dt = t_span[1]-t_span[0]

    # Unpack the state vector
    z = s.reshape((2, 2))[:, 0]
    adjoint_variable = s.reshape((2, 2))[:, 1]

    # Interpolate the reference solution
    # z_at_current_t_ref = interpolate.interp1d(t, z_ref, axis=-1)(t)
    x_at_current_t_ref = jax.numpy.interp(t, t_span, z_ref[0])
    v_at_current_t_ref = jax.numpy.interp(t, t_span, z_ref[1])

    z_at_current_t_ref = jnp.array([x_at_current_t_ref, v_at_current_t_ref])

    # Form the Jacobian of f wrt to the function parameters
    # FOR AUTODIFF THIS NEEDS TO BE REPLACED
    # del_f__del_z = np.array([[0, 1], 
    #                          [-kappa/m + 2*mu*z[0], -mu*(1-z[0]**2)*z[1]/m]])
    del_f__del_z = jax.jacobian(vdp, argnums=0)(z, t, kappa, mu, m)

    # Form the gradient of the loss function g wrt to the function parameters
    del_g__del_z = (z - z_at_current_t_ref).T

    original_rhs = vdp(z, t, kappa, mu, m).reshape((-1, 1))

    # calculate the rhs of the adjoint problem
    adjoint_rhs =  (- del_f__del_z.T @ adjoint_variable - del_g__del_z.T).reshape((-1, 1))

    return jnp.concatenate((original_rhs, adjoint_rhs), axis=1).flatten()

def spring(x, kappa):
    return -kappa * x

def damping(x, v, mu):
    return -mu*(1-x**2)*v

# # @partial(jit, static_argnums=(0,))
# def euler(fun, z0, t0, t1, t_span, args):
#     z = [z0]
#     z_old = z0
#     t_old = t0
#     for t_new in t_span[1:]:
#         dt = t_new - t_old
#         z_new = z_old + dt * fun(z_old, t_old, *args)
#         z.append(z_new)
#         t_old = t_new
#         z_old = z_new
#     return jnp.array(z).T

# def heun(fun, z0, t0, t1, t_span, args):
#     z = [z0]
#     z_old = z0
#     t_old = t0
#     for t_new in t_span[1:]:
#         dt = t_new - t_old
#         z_temp = z_old + dt * fun(z_old, t_old, *args)
#         z_new = z_old + dt * (fun(z_old, t_old, *args) + fun(z_temp, t_new, *args)) /2
#         z.append(z_new)
#         t_old = t_new
#         z_old = z_new
#     return jnp.array(z).T


def g_entire_trajectory(z_at_t, z_at_t_ref, t_span):
    difference_at_t = z_at_t - z_at_t_ref
    quadratic_loss_at_t = 0.5 * jnp.einsum("iN,iN->N", difference_at_t, difference_at_t)
    return integrate.trapezoid(quadratic_loss_at_t, t_span, axis=0)

def g_entire_trajectory_with_prediction(z_at_t_ref, z0, t0, t1, t_span, args):
    z_at_t = euler(vdp, z0, z0, t1, t_span, args)
    difference_at_t = z_at_t - z_at_t_ref
    quadratic_loss_at_t = 0.5 * jnp.einsum("iN,iN->N", difference_at_t, difference_at_t)
    return integrate.trapezoid(quadratic_loss_at_t, t_span, axis=0)

# REPLACE THIS BY AUTODIFF
# def dg_dmu(z_at_t, z_at_t_ref, t_span, mu):
#     difference_at_t = (z_at_t - z_at_t_ref)[1,1:]
#     nach_diff = (1-z_at_t[0,:-1]**2)*z_at_t[1,:-1]*(t_span[1:]-t_span[:-1])
#     return (difference_at_t + nach_diff).reshape(-1,1)


if __name__ == '__main__':
    theta = [3.0, 4, 1.0]
    theta_prd = [3.0, 1.0, 1.0]
    t0 = 0.0
    t1 = 10.0
    steps = 1601
    t_span = np.linspace(t0, t1, steps)
    z0 = np.array([1.0, 0.0])
    alpha  = 0.1
    epochs = 1000
    integration_method = euler
    use_optimizer = True

    mu = theta_prd[1]
    m = theta_prd[2]
    z_ref = integration_method(vdp, z0, t0, t1, t_span, theta)
    z_prd = integration_method(vdp, z0, t0, t1, t_span, theta_prd)
    optimizer = adam(learning_rate = alpha)
    opt_state = optimizer.init(theta_prd[1])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.plot(t_span, z_ref[0], label='Reference Position')
    ax1.plot(t_span, z_ref[1], label='Reference Velocity')
    ax1.plot(t_span, z_prd[0], label='Prediction Position')
    ax1.plot(t_span, z_prd[1], label='Prediction Velocity')
    ax1.legend()
    ax1.grid()
    ax1.set_title('Start')

    losses = []

    # if optimisation over multiple parameters; parameters in list
    dynamic_sensitivity_jacobian = lambda y, t, kappa, mu, m: jnp.array(jax.jacobian(vdp, argnums=2)(y, t, kappa, mu, m)).T
    # if paramters in list out_axes = 2
    vectorized_dynamic_sensitivity_jacobian = jit(jax.vmap(dynamic_sensitivity_jacobian, in_axes=(1, 0, None, None, None), out_axes=1))

    for epoch in range(epochs):
        # lr = alpha/(np.log(epoch+2))
        lr = alpha
        z_prd = integration_method(vdp, z0, t0, t1, t_span, theta_prd)
        loss = g_entire_trajectory(z_prd, z_ref, t_span)
        print(f'Epoch: {epoch}, Loss: {loss:.3f},  Mu:{mu:.3f}')
        losses.append(loss)
        # loss = g_entire_trajectory_with_prediction(z_ref, z0, t0, t1, t_span, theta_prd)
        # print(f'Loss: {loss}')
        terminal_condition = np.zeros((2, 2))
        terminal_condition[:, 0] = z_prd[:, -1]
        solution_and_adjoint_variable_at_t = integration_method(adjoint_model,
                                                terminal_condition.flatten(),
                                                t1,
                                                t0,
                                                np.flip(t_span),
                                                args=(theta_prd, z_ref, t_span))
        solution_and_adjoint_variable_at_t = np.flip(solution_and_adjoint_variable_at_t.reshape((2, 2, steps)), axis=2)

        solution_variable_at_t = np.flip(solution_and_adjoint_variable_at_t[:, 0, :], axis=1)
        adjoint_variable_at_t = solution_and_adjoint_variable_at_t[:, 1, :]

        # Initial condition did not depend on theta
        d_z0__d_theta = np.zeros((2, 1))

        # NEEDS TO BE REPLACED FOR AUTODIFF
        # del_f__del_theta__at_t = np.array([[np.zeros((steps))], [-(1-z_prd[0]**2)*z_prd[1]/m]])
        kappa_prd, mu_prd, m_prd = theta_prd
        del_f__del_theta__at_t = vectorized_dynamic_sensitivity_jacobian(z_prd, t_span, kappa_prd, mu_prd, m_prd)

        # If optimisation over only a parameter del_f__del_theta__at_t has form (i,N)
        # otherwise (i, j, N)
        del_f__del_theta__at_t = jnp.expand_dims(del_f__del_theta__at_t, 1)

        adjoint_variable_matmul_del_f__del_theta_at_t = np.einsum("iN,ijN->jN", adjoint_variable_at_t, del_f__del_theta__at_t)

        # d_J__d_theta__entire_trajectory = (dJ_dmu(z_ref, t_span, *args, mu) + 
        #     integrate.trapezoid(adjoint_variable_matmul_del_f__del_theta_at_t, t_span, axis=-1))
        # d_g__d_mu = dg_dmu(z_prd, z_ref, t_span, mu)

        # Derivative of g w.r.t. theta is zero
        # d_g__d_theta = jax.grad(g_entire_trajectory, argnums=3)(z_prd, z_ref, t_span, mu)

        # d_g__d_theta = jax.grad(g_entire_trajectory_with_prediction, argnums=5)(z_ref, z0, t0, t1, t_span, theta_prd)
        
        d_J__d_theta__entire_trajectory = (integrate.trapezoid(adjoint_variable_matmul_del_f__del_theta_at_t, t_span, axis=-1)) 

        # Eliminate the gradients for kappa und mu since we know they are right
        # d_J__d_theta__entire_trajectory = d_J__d_theta__entire_trajectory.at[0].set(0.0)
        # d_J__d_theta__entire_trajectory = d_J__d_theta__entire_trajectory.at[2].set(0.0)

        # print(f'Gradient: {d_J__d_theta__entire_trajectory}')

        if use_optimizer:
            updates, opt_state = optimizer.update(-d_J__d_theta__entire_trajectory, opt_state)
            theta_prd[1] = float(optax.apply_updates(theta_prd[1], updates))
        else:
            theta_prd[1] = (theta_prd[1] + lr * d_J__d_theta__entire_trajectory[0])

        mu = theta_prd[1]
        
        if epoch % 500 == 0:
            fig2, (ax4, ax5, ax6, ax7) = plt.subplots(1, 4, figsize=(24, 6))
            ax4.plot(t_span, z_ref[0], label='Reference Position')
            ax4.plot(t_span, z_ref[1], label='Reference Velocity')
            ax4.plot(t_span, z_prd[0], label='Prediction Position')
            ax4.plot(t_span, z_prd[1], label='Prediction Velocity')
            ax4.legend()
            ax4.grid()
            ax4.set_title(f'Epoch: {epoch}')

            ax5.plot(t_span, adjoint_variable_at_t[0], label = 'adjoint, x')
            ax5.plot(t_span, adjoint_variable_at_t[1], label = 'adjoint, v')
            ax5.legend()
            ax5.grid()
            ax5.set_title('Adjoint')

            ax6.plot(t_span, z_prd[0], label = 'Prediction, x')
            ax6.plot(t_span, z_prd[1], label = 'Prediction, v')
            ax6.plot(t_span, np.flip(solution_variable_at_t[0]), label = 'Reverse Prediction, x')
            ax6.plot(t_span, np.flip(solution_variable_at_t[1]), label = 'Reverse Prediction, v')
            ax6.legend()
            ax6.grid()
            
            ax7.plot(losses)
            ax7.grid()
            ax7.set_title('Losses')

            plt.show()

    ax2.plot(t_span, z_ref[0], label='Reference Position')
    ax2.plot(t_span, z_ref[1], label='Reference Velocity')
    ax2.plot(t_span, z_prd[0], label='Prediction Position')
    ax2.plot(t_span, z_prd[1], label='Prediction Velocity')
    ax2.legend()
    ax2.grid()
    ax2.set_title('Final')

    ax3.plot(losses, label='Learning loss')
    ax3.legend()
    ax3.grid()
    ax3.set_title('Losses')
    plt.show()
    fig.savefig(f'adjoint_mu_({theta[1]})_autodiff_heun_{steps}_steps.png')


    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(t_span, z_prd[0], label = 'Prediction, x')
    ax1.plot(t_span, z_prd[1], label = 'Prediction, v')
    ax1.plot(t_span, np.flip(solution_variable_at_t[0]), label = 'Reverse Prediction, x')
    ax1.plot(t_span, np.flip(solution_variable_at_t[1]), label = 'Reverse Prediction, v')
    ax1.legend()
    ax1.grid()

    ax2.plot(t_span, adjoint_variable_at_t[0], label = 'adjoint, x')
    ax2.plot(t_span, adjoint_variable_at_t[1], label = 'adjoint, v')
    ax2.legend()
    ax2.grid()
    fig2.savefig(f'adjoint_mu_({theta[1]})_autodiff_heun_{steps}_steps_extra.png')