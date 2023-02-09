import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, integrate
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

@jit
def vdp(z, t, args):
    kappa, mu, m = args
    x = z[0]
    v = z[1]
    return jnp.array([v, (spring(x, kappa) + damping(x, v, mu))/m])

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
    del_f__del_z = jax.jacobian(vdp, argnums=0)(z, t, args)

    # Form the gradient of the loss function g wrt to the function parameters
    del_g__del_z = (z - z_at_current_t_ref).T

    original_rhs = vdp(z, t, args).reshape((-1, 1))

    # calculate the rhs of the adjoint problem
    adjoint_rhs =  (- del_f__del_z.T @ adjoint_variable - del_g__del_z.T).reshape((-1, 1))

    return jnp.concatenate((original_rhs, adjoint_rhs), axis=1).flatten()

def spring(x, kappa):
    return -kappa * x

def damping(x, v, mu):
    return -mu*(1-x**2)*v

# @partial(jit, static_argnums=(0,))
def euler(fun, z0, t0, t1, t_span, args):
    z = [z0]
    z_old = z0
    t_old = t0
    for t_new in t_span[1:]:
        dt = t_new - t_old
        z_new = z_old + dt * fun(z_old, t_old, *args)
        z.append(z_new)
        t_old = t_new
        z_old = z_new
    return jnp.array(z).T

def g_entire_trajectory(z_at_t, z_at_t_ref, t_span, args):
    difference_at_t = (z_at_t - z_at_t_ref)**2
    quadratic_loss_at_t = 0.5 * jnp.sum(difference_at_t, axis=0)
    return integrate.trapezoid(quadratic_loss_at_t, t_span, axis=0)

def g_entire_trajectory_with_prediction(z_at_t_ref, z0, t0, t1, t_span, args):
    z_at_t = euler(vdp, z0, z0, t1, t_span, args)
    difference_at_t = (z_at_t - z_at_t_ref)**2
    quadratic_loss_at_t = 0.5 * jnp.sum(difference_at_t, axis=0)
    return jax.numpy.trapz(quadratic_loss_at_t, t_span, axis=0)

# REPLACE THIS BY AUTODIFF
# def dg_dmu(z_at_t, z_at_t_ref, t_span, mu):
#     difference_at_t = (z_at_t - z_at_t_ref)[1,1:]
#     nach_diff = (1-z_at_t[0,:-1]**2)*z_at_t[1,:-1]*(t_span[1:]-t_span[:-1])
#     return (difference_at_t + nach_diff).reshape(-1,1)


if __name__ == '__main__':
    theta = [[1.0, 0.5, 1.0],]
    theta_prd = [[1.0, 1.0, 1.0],]
    t0 = 0.0
    t1 = 10.0
    steps = 201
    t_span = np.linspace(t0, t1, steps)
    z0 = np.array([1.0, 0.0])
    alpha  = 0.5
    epochs = 1000

    mu = theta_prd[0][1]
    m = theta_prd[0][2]
    z_ref = euler(vdp, z0, t0, t1, t_span, theta)

    dynamic_sensitivity_jacobian = lambda y, t, params: jnp.array(jax.jacobian(vdp, argnums=2)(y, t, *params)).T
    vectorized_dynamic_sensitivity_jacobian = jit(jax.vmap(dynamic_sensitivity_jacobian, in_axes=(1, 0, None), out_axes=2))

    for epoch in range(epochs):
        # lr = alpha/(np.log(epoch+2))
        lr = alpha
        z_prd = euler(vdp, z0, z0, t1, t_span, theta_prd)
        loss = g_entire_trajectory(z_prd, z_ref, t_span, theta_prd)
        # loss = g_entire_trajectory_with_prediction(z_ref, z0, t0, t1, t_span, theta_prd)
        print(f'Loss: {loss}')
        terminal_condition = np.zeros((2, 2))
        terminal_condition[:, 0] = z_prd[:, -1]
        solution_and_adjoint_variable_at_t = euler(adjoint_model,
                                                terminal_condition.flatten(),
                                                t1,
                                                t0,
                                                np.flip(t_span),
                                                args=(*theta_prd, z_ref, t_span))
        solution_and_adjoint_variable_at_t = np.flip(solution_and_adjoint_variable_at_t.reshape((2, 2, steps)), axis=2)

        solution_variable_at_t = np.flip(solution_and_adjoint_variable_at_t[:, 0, :], axis=1)
        adjoint_variable_at_t = solution_and_adjoint_variable_at_t[:, 1, :]

        # Initial condition did not depend on theta
        d_z0__d_theta = np.zeros((2, 3))

        # NEEDS TO BE REPLACED FOR AUTODIFF
        # del_f__del_theta__at_t = np.array([[np.zeros((steps))], [-(1-z_prd[0]**2)*z_prd[1]/m]])
        del_f__del_theta__at_t = vectorized_dynamic_sensitivity_jacobian(z_prd, t_span, theta_prd)

        adjoint_variable_matmul_del_f__del_theta_at_t = np.einsum("iN,ijN->jN", adjoint_variable_at_t, del_f__del_theta__at_t)

        # d_J__d_theta__entire_trajectory = (dJ_dmu(z_ref, t_span, *args, mu) + 
        #     integrate.trapezoid(adjoint_variable_matmul_del_f__del_theta_at_t, t_span, axis=-1))
        # d_g__d_mu = dg_dmu(z_prd, z_ref, t_span, mu)

        d_g__d_theta = jax.grad(g_entire_trajectory, argnums=3)(z_prd, z_ref, t_span, mu)

        # d_g__d_theta = jax.grad(g_entire_trajectory_with_prediction, argnums=5)(z_ref, z0, t0, t1, t_span, theta_prd)
        
        d_J__d_theta__entire_trajectory = (integrate.trapezoid(adjoint_variable_matmul_del_f__del_theta_at_t, t_span, axis=-1) + 
            d_g__d_theta) 

        # Eliminate the gradients for kappa und mu since we know they are right
        d_J__d_theta__entire_trajectory = d_J__d_theta__entire_trajectory.at[0].set(0.0)
        d_J__d_theta__entire_trajectory = d_J__d_theta__entire_trajectory.at[2].set(0.0)

        print(f'Gradient: {d_J__d_theta__entire_trajectory}')

        theta_prd = (jnp.array(theta_prd) - lr * d_J__d_theta__entire_trajectory).tolist()
        # theta_prd[0][1] = mu
        mu = theta_prd[0][1]
        print(f'Mu:{mu}')
        if epoch % 100 == 0:
            plt.plot(t_span, z_ref[0], label='Reference Position')
            plt.plot(t_span, z_ref[1], label='Reference Velocity')
            plt.plot(t_span, z_prd[0], label='Prediction Position')
            plt.plot(t_span, z_prd[1], label='Prediction Velocity')
            plt.legend()
            plt.grid()
            fig = plt.figure()
            ax = fig.subplots()
            ax.plot(t_span, adjoint_variable_at_t[0], label = 'adjoint, x')
            ax.plot(t_span, adjoint_variable_at_t[1], label = 'adjoint, v')
            ax.legend()
            ax.grid()
            fig = plt.figure()
            ax = fig.subplots()
            ax.plot(t_span, z_prd[0], label = 'Prediction, x')
            ax.plot(t_span, z_prd[1], label = 'Prediction, v')
            ax.plot(t_span, np.flip(solution_variable_at_t[0]), label = 'Reverse Prediction, x')
            ax.plot(t_span, np.flip(solution_variable_at_t[1]), label = 'Reverse Prediction, v')

            ax.legend()
            ax.grid()
            plt.show()

    plt.plot(t_span, z_ref[0], label='Reference Position')
    plt.plot(t_span, z_ref[1], label='Reference Velocity')
    plt.plot(t_span, z_prd[0], label='Prediction Position')
    plt.plot(t_span, z_prd[1], label='Prediction Velocity')
    plt.legend()
    plt.grid()
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(t_span, solution_variable_at_t[0], label = 'solution, x')
    ax.plot(t_span, solution_variable_at_t[1], label = 'solution, v')
    ax.legend()
    ax.grid()
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(t_span, adjoint_variable_at_t[0], label = 'adjoint, x')
    ax.plot(t_span, adjoint_variable_at_t[1], label = 'adjoint, v')
    ax.legend()
    ax.grid()
    plt.show()