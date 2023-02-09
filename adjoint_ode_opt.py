import copy
import time

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, interpolate
from sklearn import metrics


def model(t, y, args):
    """
    The Model is a forced oscillator given by the second order scalar valued
    linear ODE y'' + c*y' + k*y = f * cos(w*t)

    where: y : Displacement y' : Velocity y'' : acceleration c : friction
    coefficient k : spring stiffness w : frequency of forcing t : time f :
    amplitude of forcing

    This scalar ODE of order 2 is transformed into a system of ODEs with order 1
    """
    c, k, f, w  = args

    return jnp.array([
        y[1],
        # f * jnp.sin(w*t) - k * (1-y[0]**2)*y[1] - c*y[0]
        - k * (1-y[0]**2)*y[1] - c*y[0]
        # f * jnp.cos(w*t) - k * y[0] - c*y[1]
    ])

def forward_sensitivity_model(t, s, args):
    """
    This solves a system of (P+1)*N differential equations (needs to solve the
    original ODE in conjunction)
    """
    y = s.reshape((2, 5))[:, 0]
    sensitivities = s.reshape((2, 5))[:, 1:]

    del_f__del_u = jax.jacobian(model, argnums=1)(t, y, args)
    del_f__del_theta = jnp.array(jax.jacobian(model, argnums=2)(t, y, args)).T
    
    
    original_rhs = model(t, y, args).reshape((-1, 1))
    sensitivity_rhs = del_f__del_u @ sensitivities + del_f__del_theta

    return jnp.concatenate((original_rhs, sensitivity_rhs), axis=1).flatten()

def adjoint_model_homogeneous(t, s, args):
    """
    This solves a system of 2*N differential equations in reverse, first the
    original ODE system (which is needed since we require the solution to
    evaluate the Jacobian) and then the adjoint system.

    THIS IS THE HOMOGENEOUS VERSION, that only works if the loss function is
    evaluated at the end of time horizon, because then the del_J__del_u can be
    used as a terminal condition and makes the ODE homogeneous
    """
    y = s.reshape((2, 2))[:, 0]
    adjoint_variable = s.reshape((2, 2))[:, 1]

    del_f__del_u = jax.jacobian(model, argnums=1)(t, y, args)

    original_rhs = model(t, y, args).reshape((-1, 1))
    adjoint_rhs = (- del_f__del_u.T @ adjoint_variable).reshape((-1, 1))

    return jnp.concatenate((original_rhs, adjoint_rhs), axis=1).flatten()

def adjoint_model_general(t, s, args, y_at_t_ref, t_discrete):
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
    # Unpack the state vector
    y = s.reshape((2, 2))[:, 0]
    adjoint_variable = s.reshape((2, 2))[:, 1]

    # Interpolate the reference solution
    y_at_current_t_ref = interpolate.interp1d(t_discrete, y_at_t_ref, axis=-1)(t)

    del_f__del_y = jax.jacobian(model, argnums=1)(t, y, args)
    del_J__del_y = (y - y_at_current_t_ref).T

    original_rhs = model(t, y, args).reshape((-1, 1))
    adjoint_rhs = (- del_f__del_y.T @ adjoint_variable - del_J__del_y.T).reshape((-1, 1))

    return jnp.concatenate((original_rhs, adjoint_rhs), axis=1).flatten()


if __name__ == '__main__':
    # Defining the integration horizon and the discretization
    integration_horizon = (0.0, 5.0)
    time_points_inbetween = 100
    t_discrete = np.linspace(integration_horizon[0], integration_horizon[1], time_points_inbetween)

    # The initial condition are not subject to parameters
    initital_condition = [1.0, 0.0]
    # initital_condition = [0.5, 0.0]

    #################
    ###### Creating a reference trajectory using "true" values
    #################
    c_true   = 1.0
    k_true   = 4.0
    f_true   = 0.0
    w_true   = 1.05
    parameters_true = [[
        c_true,
        k_true,
        f_true,
        w_true,
    ],]

    y_at_t_ref = integrate.solve_ivp(
        fun=model, 
        t_span=integration_horizon,
        y0=initital_condition,
        t_eval=t_discrete,
        args=parameters_true
    )["y"]

    ##############
    ##### Define Loss functions that rely on the reference solution
    ##############

    # Loss function that only consider the value at the very end of the
    # integration horizon. It uses a quadratic loss to contract the dimension
    def loss_function_at_end(y_at_t, theta):
        y_at_terminal_point = y_at_t[:, -1]
        y_at_terminal_point_ref = y_at_t_ref[:, -1]

        return 0.5 * (y_at_terminal_point - y_at_terminal_point_ref).T @ (y_at_terminal_point - y_at_terminal_point_ref)

    # Loss function that considers the quadratic loss over the entire trajectory
    def loss_function_entire_trajectory(y_at_t, theta):
        difference_at_t = y_at_t - y_at_t_ref
        quadratic_loss_at_t = 0.5 * np.einsum("iN,iN->N", difference_at_t, difference_at_t)

        return integrate.trapezoid(quadratic_loss_at_t, t_discrete, axis=-1)

    ##############
    ##### Now work based on parameter guesses
    ##############
    c_guess = 1.0
    k_guess = 1.0
    f_guess = 0.0
    w_guess = 1.05
    parameters_guess = [[
        c_guess,
        k_guess,
        f_guess,
        w_guess
    ],]

    y_at_t = integrate.solve_ivp(
        fun=model, 
        t_span=integration_horizon,
        y0=initital_condition,
        t_eval=t_discrete,
        args=parameters_guess
    )["y"]

    plt.figure()
    plt.subplot(121)
    plt.plot(t_discrete, y_at_t_ref[0, :], label="Reference solution")
    plt.plot(t_discrete, y_at_t[0, :], label="Guessed parameters")
    plt.legend()
    plt.grid()
    plt.subplot(122)
    plt.plot(t_discrete, y_at_t_ref[1, :], label="Reference solution")
    plt.plot(t_discrete, y_at_t[1, :], label="Guessed parameters")
    plt.legend()
    plt.grid()
    plt.show()

    lr = 0.05

    dynamic_sensitivity_jacobian = lambda t, y, params: jnp.array(jax.jacobian(model, argnums=2)(t, y, *params)).T
    # The jit is not really advantageous, because we are only calling the function once
    vectorized_dynamic_sensitivity_jacobian = jax.jit(jax.vmap(dynamic_sensitivity_jacobian, in_axes=(0, 1, None), out_axes=2))

    for epoch in range(100):

        # Solve the "classical" system, i.e. solve forward with the current guess
        # for the parameters. This would have to be solved anyways if wanted to
        # evaluate the prediction of the system or are interested in how well our
        # parameter guesses hold (i.e. evaluate the loss). Keep in mind that the
        # solution is queried at some intermediate points in order to then evaluate
        # the (potentially) integral-based loss over the entire time.
        time_classical_problem = time.time_ns()
        y_at_t = integrate.solve_ivp(
            fun=model, 
            t_span=integration_horizon,
            y0=initital_condition,
            t_eval=t_discrete,
            args=parameters_guess
        )["y"]
        time_classical_problem = time.time_ns() - time_classical_problem
        
        # J_at_end = loss_function_at_end(y_at_t, parameters_guess)
        J_entire_trajectory = loss_function_entire_trajectory(y_at_t, parameters_guess)

        print(f'Loss: {J_entire_trajectory}')

        #######
        # (1.2) Using an additional trajectory that runs backwards alongside with
        # the backwards ODE for loss over the entire trajectory
        #######

        terminal_condition_adjoint_sensitivities = jnp.zeros((2, 2))

        # The reverse running original ODE of course starts where the forward running one ended
        terminal_condition_adjoint_sensitivities = terminal_condition_adjoint_sensitivities.at[:, 0].set(y_at_t[:, -1])

        # In contrast to before, we now have the loss function valid over the entire
        # trajectory. Therefore, the adjoint ODE is inhomogeneous, but has a
        # homogenous (=zero) terminal condition. Since the corresponding array is
        # already initialized as zeros, nothing has to be done
        #
        # However, we have to provide the reference solution as well as its temporal
        # mesh such that it can be interpolated in the evaluation of the dynamics
        #
        # Running the ODE backwards does not seem to be a problem, once t_span is
        # set correctly and t_eval points are reversed (using np.flip)
        solution_and_adjoint_variable_at_t = np.flip(integrate.solve_ivp(
            fun=adjoint_model_general, 
            t_span=np.flip(integration_horizon),
            y0=terminal_condition_adjoint_sensitivities.flatten(),
            t_eval=np.flip(t_discrete),
            args=(*parameters_guess, y_at_t_ref, t_discrete),
        )["y"].reshape((2, 2, time_points_inbetween)), axis=2)

        y_at_t__backwards = np.flip(solution_and_adjoint_variable_at_t[:, 0, :], axis=1)
        adjoint_variable_at_t = solution_and_adjoint_variable_at_t[:, 1, :]

        # The initial condition was not dependent on the parameters
        d_u0__d_theta = jnp.zeros((2, 4))

        del_f__del_theta__at_t = vectorized_dynamic_sensitivity_jacobian(t_discrete, y_at_t, parameters_guess)
        
        adjoint_variable_matmul_del_f__del_theta_at_t = jnp.einsum("iN,ijN->jN", adjoint_variable_at_t, del_f__del_theta__at_t)

        d_J__d_theta__entire_trajectory__adjoint = (
            adjoint_variable_at_t[:, -1].T @ d_u0__d_theta
            + 
            jnp.zeros((1, 4))
            +
            integrate.trapezoid(adjoint_variable_matmul_del_f__del_theta_at_t, t_discrete, axis=-1)
        )

        parameters_guess = np.array(parameters_guess) - lr * d_J__d_theta__entire_trajectory__adjoint
        print(parameters_guess)
        parameters_guess = parameters_guess.tolist()
        

    ###########
    #### Post-Processing
    ###########
    # Plots

    plt.figure()
    plt.subplot(121)
    plt.plot(t_discrete, y_at_t_ref[0, :], label="Reference solution")
    plt.plot(t_discrete, y_at_t[0, :], label="Guessed parameters")
    plt.legend()
    plt.grid()
    plt.subplot(122)
    plt.plot(t_discrete, y_at_t_ref[1, :], label="Reference solution")
    plt.plot(t_discrete, y_at_t[1, :], label="Guessed parameters")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.title("Solution run reverse")
    plt.plot(t_discrete, y_at_t__backwards[0, :])
    plt.plot(t_discrete, y_at_t__backwards[1, :])

    # plt.figure()
    # plt.plot(t_discrete, y_at_t__backwards[0, :])
    # plt.plot(t_discrete, y_at_t__backwards[1, :])
    # plt.title("Stuff")

    plt.show()
