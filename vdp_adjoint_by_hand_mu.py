import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, integrate
import jax
from jax import numpy as jnp
from ode import euler, heun

# this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)

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
    x_at_current_t_ref = jax.numpy.interp(t, t_span, z_ref[0])
    v_at_current_t_ref = jax.numpy.interp(t, t_span, z_ref[1])

    z_at_current_t_ref = jnp.array([x_at_current_t_ref, v_at_current_t_ref])

    del_f__del_z = jnp.array([[0, 1], 
                             [-kappa/m + 2*mu*z[0]*z[1], -mu*(1-z[0]**2)/m]])

    del_g__del_z = (z - z_at_current_t_ref).T

    original_rhs = vdp(z, t, args).reshape((-1, 1))
    adjoint_rhs =  (- (del_f__del_z.T @ adjoint_variable) - del_g__del_z.T).reshape((-1, 1))

    return jnp.concatenate((original_rhs, adjoint_rhs), axis=1).flatten()

def spring(x, kappa):
    return -kappa * x

def damping(x, v, mu):
    return -mu*(1-x**2)*v

def J(z_at_t, z_at_t_ref, t_span, mu):
    difference_at_t = z_at_t_ref - z_at_t
    quadratic_loss_at_t = 0.5*difference_at_t**2
    # quadratic_loss_at_t = 0.5 * jnp.einsum("iN,iN->N", difference_at_t, difference_at_t) # Just the scalarproduct
    # return integrate.trapezoid(quadratic_loss_at_t, t_span, axis=0)
    return jnp.sum(quadratic_loss_at_t)


if __name__ == '__main__':
    theta = [[3.0, 8.53, 1.0],]
    theta_prd = [[3.0, 1.0, 1.0],]
    start_time = 0.0
    end_time = 10.0
    steps = 401
    t_span = np.linspace(start_time, end_time, steps)
    initial_condition = np.array([1.0, 0.0])
    learning_rate_initial = 0.001
    epochs = 1000
    integration_method = euler

    mu = theta_prd[0][1]
    m = theta_prd[0][2]
    z_ref = integration_method(vdp, initial_condition, start_time, end_time, t_span, theta)
    z_prd = integration_method(vdp, initial_condition, start_time, end_time, t_span, theta_prd)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.plot(t_span, z_ref[0], label='Reference Position')
    ax1.plot(t_span, z_ref[1], label='Reference Velocity')
    ax1.plot(t_span, z_prd[0], label='Prediction Position')
    ax1.plot(t_span, z_prd[1], label='Prediction Velocity')
    ax1.legend()
    ax1.grid()
    ax1.set_title('Start')

    losses = []

    for epoch in range(epochs):
        # lr = alpha/(np.log(epoch+2))
        lr = learning_rate_initial
        z_prd = integration_method(vdp, initial_condition, start_time, end_time, t_span, theta_prd)
        loss = J(z_prd, z_ref, t_span, mu)
        losses.append(loss)
        print(f'Epoch: {epoch}, Loss: {loss:.3f},  Mu:{mu:.3f}')
        terminal_condition = np.zeros((2, 2))
        terminal_condition[:, 0] = z_prd[:, -1]
        solution_and_adjoint_variable_at_t = integration_method(adjoint_model,
                                                terminal_condition.flatten(),
                                                end_time,
                                                start_time,
                                                np.flip(t_span),
                                                args=(*theta_prd, z_ref, t_span))
        solution_and_adjoint_variable_at_t = np.flip(solution_and_adjoint_variable_at_t.reshape((2, 2, steps)), axis=2)

        solution_variable_at_t = np.flip(solution_and_adjoint_variable_at_t[:, 0, :], axis=1)
        adjoint_variable_at_t = solution_and_adjoint_variable_at_t[:, 1, :]

        # Initial condition did not depend on mu
        d_initial_condition__d_mu = np.zeros((2, 1))

        del_f__del_theta__at_t = np.array([[np.zeros((steps))], [-(1-z_prd[0]**2)*z_prd[1]/m]])

        adjoint_variable_matmul_del_f__del_theta_at_t = np.einsum("iN,ijN->jN", adjoint_variable_at_t, del_f__del_theta__at_t)

        # The Derivative of g with respect to mu is equal to zero; thats why it is skipped here

        # d_J__d_theta__entire_trajectory = integrate.trapezoid(adjoint_variable_matmul_del_f__del_theta_at_t, t_span, axis=-1)
        d_J__d_theta__entire_trajectory = jnp.sum(adjoint_variable_matmul_del_f__del_theta_at_t, axis=-1)


        # mu = mu - lr * d_J__d_theta__entire_trajectory[0]
        theta_prd[0][1] = mu
        if epoch % 500 == 0 or epoch == epochs-1:
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
    fig.savefig(f'adjoint_mu_({theta[0][1]})_hand_heun_{steps}_steps.png')


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
    fig2.savefig(f'adjoint_mu_({theta[0][1]})_hand_heun_{steps}_steps_extra.png')
    