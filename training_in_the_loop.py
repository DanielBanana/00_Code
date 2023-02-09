import jax
from jax import grad
from jax import random
from jax import vmap
from jax import jit
from scipy import integrate, interpolate
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

from jax.nn import relu

def vdp(x, t, args):
    mu = args
    return np.array([x[1], damping(x, t, mu) + spring(x, t, mu)])

def spring(x, t, args):
    return -x[0]

def damping(x, t, args):
    mu = args
    return mu*(1-x[0]**2)*x[1]

def vdp_nn(x, t, args):
    params, mu = args
    return np.array([x[1], net(x, params, t, mu)+ spring(x, t, mu)])

def adjoint_model_general(state, t, args, target, t_discrete):
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
    z = state.reshape((2, 2))[:, 0]
    adjoint_variable = state.reshape((2, 2))[:, 1]

    # Interpolate the reference solution
    target_at_current_t = interpolate.interp1d(t_discrete, target, axis=0)(t)
    
    del_f__del_z = jax.jacobian(net, argnums=0)(z, t, *args)
    del_J__del_z = (z - target_at_current_t).T

    original_rhs = net(z, t, *args).reshape((-1, 1))
    adjoint_rhs = (- del_f__del_z.T @ adjoint_variable - del_J__del_z.T).reshape((-1, 1))

    return jnp.concatenate((original_rhs, adjoint_rhs), axis=1).flatten()


def euler(function, x0, start_time, stop_time, t_discrete, args):
    solution_vector = [x0]
    x = x0
    time = start_time
    for i in range(len(t_discrete)-1):
        time = t_discrete[i]
        dt = t_discrete[i+1]-t_discrete[i]
        x_new = x + dt*function(x, time, *args)
        solution_vector.append(x_new)
        x = x_new
        time += dt
    return np.array(solution_vector)

def net(x: jnp.ndarray, t, params: jnp.ndarray, mu) -> jnp.ndarray:
    hidden_layers = params[:-1]
    activation = x
    for w, b in hidden_layers:
        activation = jax.nn.relu(jnp.dot(w, activation) + b)
    w_last, b_last = params[-1]
    output = jnp.dot(w_last, activation) + b_last
    return output
batched_net = jax.vmap(net, in_axes=(0, None, None, None))

def net_init(layer_widths, parent_key, scale=1):

    if scale != 0.0:
        params = []
        keys = jax.random.split(parent_key, num=len(layer_widths)-1)
        for in_width, out_width, key in zip(layer_widths[:-1], layer_widths[1:], keys):
            weight_key, bias_key = jax.random.split(key)
            params.append([
                        scale*jax.random.normal(weight_key, shape=(out_width, in_width)),
                        scale*jax.random.normal(bias_key, shape=(out_width,))
                        ]
            )
    else:
        params = []
        keys = jax.random.split(parent_key, num=len(layer_widths)-1)
        for in_width, out_width, key in zip(layer_widths[:-1], layer_widths[1:], keys):
            weight_key, bias_key = jax.random.split(key)
            params.append([
                        jnp.zeros(shape=(out_width, in_width)),
                        jnp.zeros(shape=(out_width,))
                        ]
            )
    return params

def evaluate_net(x, t0, t1, t_discrete, params, args):
    args = [params, args]
    return euler(net, x, t0, t1, t_discrete, args)

def loss_function_at_end(prediction, target, theta):
    prediction_at_terminal_point = prediction[:, -1]
    target_at_terminal_point = target[:, -1]
    return 0.5 * (prediction_at_terminal_point - target_at_terminal_point).T @ (prediction_at_terminal_point - target_at_terminal_point)


def loss_function_entire_trajectory(prediciton, target, t_discrete, params):
    difference_at_t = prediciton - target
    quadratic_loss_at_t = 0.5 * np.einsum("iN,iN->N", difference_at_t, difference_at_t)
    return integrate.trapezoid(quadratic_loss_at_t, t_discrete, axis=-1)

# def ODEnet(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
#     return euler(vdp_nn, x, 0, 1, params)

# def f_and_a(adjoints, t):
#     x, a, d = adjoints




if __name__ == '__main__':
    # Solver Variables
    a = 0.0
    b = 10.0
    n_steps = 1000
    t_discrete = np.linspace(a, b, n_steps)
    ic = np.array([1, 0])
    mu = [2]

    # Get the Reference solution
    solution = euler(vdp, ic, a, b, t_discrete, mu)

    # Inititialize Net
    seed = 0
    key = jax.random.PRNGKey(seed)
    layer_sizes = [2, 10, 2]
    theta = net_init(layer_sizes, key, scale=0.01)
    prediction = evaluate_net(ic, a, b, t_discrete, theta, mu)


    plt.figure()
    plt.subplot(121)
    plt.plot(t_discrete, solution[:,0], label="Reference x")
    plt.plot(t_discrete, prediction[:,0], label="Prediction x")
    plt.legend()
    plt.grid()
    plt.subplot(122)
    plt.plot(t_discrete, solution[:,1], label="Reference v")
    plt.plot(t_discrete, prediction[:,1], label="Prediction v")
    plt.legend()
    plt.grid()
    plt.show()

    J_at_end = loss_function_at_end(prediction, solution, theta)
    J_entire_trajectory = loss_function_entire_trajectory(prediction, solution, t_discrete, theta)

    #######
    # (1.2) Using an additional trajectory that runs backwards alongside with
    # the backwards ODE for loss over the entire trajectory
    #######

    terminal_condition_adjoint_sensitivities = jnp.zeros((2, 2))

    # The reverse running original ODE of course starts where the forward running one ended
    terminal_condition_adjoint_sensitivities = terminal_condition_adjoint_sensitivities.at[:, 0].set(prediction[-1, :])

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
    solution_and_adjoint_variable_at_t = np.flip(euler(
        function=adjoint_model_general,
        x0=terminal_condition_adjoint_sensitivities.flatten(),
        start_time=b, 
        stop_time=a,
        t_discrete=np.flip(t_discrete),
        args=([theta, mu], solution, t_discrete)
    ).reshape((2, 2, n_steps)), axis=2)

    adjoint_variable_at_t = solution_and_adjoint_variable_at_t[:, 1, :]

    # The initial condition was not dependent on the parameters
    # So we generate a zeros with the shape of the parameters (i.e. in the form of the neural network)
    # with the variables of the ode as first dimension
    d_z0__d_theta = jnp.zeros((2, 4))
    d_z0__d_theta = net_init(layer_sizes, key, scale=0.0)

    # dynamic_sensitivity_jacobian = lambda z, t, theta, mu: jnp.array(jax.jacobian(net, argnums=2)(z, t, theta, mu)).T
    test_dynamic_sensitivity_jacobian = lambda z, t, theta, mu: jax.jacobian(net, argnums=2)(z, t, theta, mu)
    # The jit is not really advantageous, because we are only calling the function once
    # vectorized_dynamic_sensitivity_jacobian = jax.jit(jax.vmap(dynamic_sensitivity_jacobian, in_axes=(1, 0, None, None), out_axes=2))
    test_vectorized_dynamic_sensitivity_jacobian = jax.jit(jax.vmap(test_dynamic_sensitivity_jacobian, in_axes=(1, 0, None, None), out_axes=2))

    del_f__del_theta__at_t = test_vectorized_dynamic_sensitivity_jacobian(prediction.T, t_discrete, theta, mu)

    adjoint_variable_matmul_del_f__del_theta_at_t = []
    for i, layer_grad in enumerate(del_f__del_theta__at_t):
        weight_grad = layer_grad[0]
        bias_grad = layer_grad[1]


        # Multiply adjoints with the del_f__del_theta__at_t gradients       
        adjoint_variable_matmul_del_f__del_theta_at_t.append([jnp.einsum("iN,ijNk->jNk", adjoint_variable_at_t, weight_grad), jnp.einsum("iN,ijN->jN", adjoint_variable_at_t, bias_grad)])


        # weight_grads.append(layer_grad[0])
        # bias_grads.append(layer_grad[1])

    # adjoint_variable_matmul_del_f__del_theta_at_t_biases = []
    # adjoint_variable_matmul_del_f__del_theta_at_t_weights = []

    # for bias_grad in bias_grads:
    #     adjoint_variable_matmul_del_f__del_theta_at_t_biases.append(jnp.einsum("iN,ijN->jN", adjoint_variable_at_t, bias_grad))

    # for weight_grad in weight_grads:
    #     adjoint_variable_matmul_del_f__del_theta_at_t_weights.append(jnp.einsum("iN,ijNk->jNk", adjoint_variable_at_t, weight_grad))

    # adjoint_variable_matmul_del_f__del_theta_at_t = jnp.einsum("iN,ijN->jN", adjoint_variable_at_t, del_f__del_theta__at_t)

    pass

    # Integrate the last term over the time trajectory

    for i, layer, in enumerate(adjoint_variable_matmul_del_f__del_theta_at_t):
        weights = layer[0]
        biases = layer[1]

        current_weights = theta[i][0]
        current_biases = theta[i][1]

        # d_z0__d_theta_weights = d_z0__d_theta_[0]
        # d_z0__d_theta_biases = d_z0__d_theta_[1]
        # jnp.zeros_like(d_z0__d_theta_)

        d_J__d_theta_weight__entire_trajectory__adjoint = (
        
        # adjoint_variable_at_t[:, -1].T @ d_z0__d_theta_weights
        # + 
        # jnp.zeros_like(d_z0__d_theta_weights)
        # +
        integrate.trapezoid(weights, t_discrete, axis=1)
        )

        d_J__d_theta_biases__entire_trajectory__adjoint = (
        
        # adjoint_variable_at_t[:, -1].T @ d_z0__d_theta_weights
        # + 
        # jnp.zeros_like(d_z0__d_theta_weights)
        # +
        integrate.trapezoid(biases, t_discrete, axis=1)
        )

        lr = 0.001
        new_weights = current_weights + lr * d_J__d_theta_weight__entire_trajectory__adjoint
        new_biases = current_biases + lr * d_J__d_theta_biases__entire_trajectory__adjoint








    # d_J__d_theta__entire_trajectory__adjoint = (
    #     adjoint_variable_at_t[:, -1].T @ d_z0__d_theta
    #     + 
    #     jnp.zeros((1, 4))
    #     +
    #     integrate.trapezoid(adjoint_variable_matmul_del_f__del_theta_at_t, t_discrete, axis=-1)
    # )