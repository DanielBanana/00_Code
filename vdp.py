import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit
import numpy as np
from scipy import integrate, interpolate
from matplotlib import pyplot as plt
from euler import euler, euler_NN
import optax
from typing import Tuple, List

@jit
def model_true(t, y, args):
    m, mu, k, A, w = args

    return jnp.array([
        y[1],
        # A * jnp.sin(w*t) + (mu * (1-y[0]**2)*y[1] - k*y[0])/m
        (mu * (1-y[0]**2)*y[1] - k*y[0])/m
    ])

@jit
def model_hybrid(t, y, params, args):
    m, mu, k, A, w = args
    return jnp.array([
        y[1],
        (mu*net(y, params) - spring(t, y, args)[0])[0]
    ])

def spring(t, y, args):
    m, mu, k, A, w = args
    return jnp.array((k*y[0])/m, ndmin=2)

def euler_NN(y0, t_eval, params, args):
    step_size = t_eval[1] - t_eval[0]
    y = jnp.array(y0)
    sol = [y]
    for time in t_eval[:-1]:
        y_next = euler_NN_step(y, time, step_size, params, args)
        sol.append(y_next)
        y = y_next
    return jnp.array(sol).T

def euler_NN_step(y, time, step_size, params, args):
    return y + step_size * model_hybrid(time, y, params, args)


def loss(params, input, output, mu):
    return jnp.sum((output - mu*batched_net(input, params)[:-1,0])**2)
    # return jnp.sum(((y[1,1:]-y[1,:-1])/step_size - (((spring(t_discrete, y, parameters_guess).T + mu*batched_NN_predict(y.T, NN_params)))/m)[:-1])**2)

# def update(y, params, args, lr=0.001):
#     value = loss(y, params, args)
#     loss, grads = jax.value_and_grad(loss, argnums=[1])(y, params, args)
#     return loss, jax.tree_map(lambda p, g: p - lr*g, params, grads[0])

def get_model_data(args):
    integration_horizon, initial_condition, t_discrete, parameters_true = args
    x = euler(model_true, integration_horizon, initial_condition, t_discrete, parameters_true)
    m, mu, k, A, w = parameters_true
    output = ((x[1,1:]-x[1,:-1])/step_size).reshape(1, -1) - (((spring(t_discrete, x, parameters_true)))/m)[:,:-1]
    return x.T, output.T

def fit(params: optax.Params, optimizer: optax.GradientTransformation, INPUTS, OUTPUTS, epochs, args):
    opt_state = optimizer.init(params)
    integration_horizon, initial_condition, t_discrete, phys_params = args

    @jit
    def step(params, opt_state, input, output, args):
        loss_value, grads = jax.value_and_grad(loss)(params, input, output, args)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    losses = []
    n_batches = INPUTS.shape[2]
    for epoch in range(epochs):
        batch_losses = []
        for i in range(n_batches):
            input = INPUTS[:,:,i]
            output = OUTPUTS[:,:,i]
            params, opt_state, batch_loss = step(params, opt_state, input, output, phys_params[1])
            batch_losses.append(batch_loss)
        total_batch_loss = jnp.array(batch_losses).mean()
        losses.append(total_batch_loss)

        if epoch % 10 == 0:
            print(f'step {epoch}, loss: {total_batch_loss}')
        if epoch % 10000 == 0:
            plt.figure()
            plt.subplot(121)
            plt.plot(OUTPUTS[:,:,0], label="Resdiuals")
            plt.plot(batched_net(INPUTS[:,:,0], params), label="Predictions")
            plt.legend()
            plt.grid()
            plt.subplot(122)
            plt.plot(losses, label="Losses")
            plt.legend()
            plt.grid()
            plt.show()
    return params, losses

from jax.experimental.ode import odeint

def nn_dynamics(state, time, params):
    state_and_time = jnp.hstack([state, jnp.array(time)])
    return net(state_and_time, params)

def odenet(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    start_and_end_times = jnp.array([0.0, 1.0])
    init_state, final_state = odeint(nn_dynamics, x, start_and_end_times, params, atol=0.001, rtol=0.001)
    return final_state
batched_odenet = jax.vmap(odenet, in_axes=(0, None))

def odenet_loss(params, input, output, mu):
    return jnp.sum((output - mu*batched_odenet(input, params)[:-1,0])**2)

def odenet_fit(params: optax.Params, optimizer: optax.GradientTransformation, INPUTS, OUTPUTS, epochs, args):
    opt_state = optimizer.init(params)
    integration_horizon, initial_condition, t_discrete, phys_params = args

    # @jit
    def step(params, opt_state, input, output, args):
        odenet_loss(params, input, output, args)
        loss_value, grads = jax.value_and_grad(odenet_loss)(params, input, output, args)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    losses = []
    n_batches = INPUTS.shape[2]
    for epoch in range(epochs):
        batch_losses = []
        for i in range(n_batches):
            input = INPUTS[:,:,i]
            output = OUTPUTS[:,:,i]
            params, opt_state, batch_loss = step(params, opt_state, input, output, phys_params[1])
            batch_losses.append(batch_loss)
        total_batch_loss = jnp.array(batch_losses).mean()
        losses.append(total_batch_loss)

        if epoch % 10 == 0:
            print(f'step {epoch}, loss: {total_batch_loss}')
        if epoch % 1000 == 0:
            plt.figure()
            plt.subplot(121)
            plt.plot(OUTPUTS[:,:,0], label="Resdiuals")
            plt.plot(batched_net(INPUTS[:,:,0], params), label="Predictions")
            plt.legend()
            plt.grid()
            plt.subplot(122)
            plt.plot(losses, label="Losses")
            plt.legend()
            plt.grid()
            plt.show()
    return params, losses

@jit
def net(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    hidden_layers = params[:-1]
    activation = x
    for w, b in hidden_layers:
        activation = jax.nn.relu(jnp.dot(w, activation) + b)
    w_last, b_last = params[-1]
    output = jnp.dot(w_last, activation) + b_last
    return output
batched_net = jax.vmap(net, in_axes=(0, None))

def net_init(layer_widths, parent_key, scale=1):
    params = []
    keys = jax.random.split(parent_key, num=len(layer_widths)-1)
    for in_width, out_width, key in zip(layer_widths[:-1], layer_widths[1:], keys):
        weight_key, bias_key = jax.random.split(key)
        params.append([
                       scale*jax.random.normal(weight_key, shape=(out_width, in_width)),
                       scale*jax.random.normal(bias_key, shape=(out_width,))
                       ]
        )
    return params


if __name__ == '__main__':
    # test
    seed = 0
    key = jax.random.PRNGKey(seed)
    NN_params = net_init([2, 20, 20, 1], key)
    # print(jax.tree_map(lambda x: x.shape, NN_params))

    # Additional Dimensions for time
    odenet_params = net_init([3, 10, 1], key)

    # Defining the integration horizon and the discretization
    integration_horizon = (0.0, 1.0)
    time_points_inbetween = 101
    step_size = (integration_horizon[1] - integration_horizon[0])/time_points_inbetween
    t_discrete = np.linspace(integration_horizon[0], integration_horizon[1], time_points_inbetween)

    # The initial condition are not subject to parameters
    initial_condition = [1.0, 0.0]
    # initial_condition = [0.5, 0.0]

    #################
    ###### Creating a reference trajectory using "true" values
    #################
    m_true = 1.0
    mu_true = 8.53
    k_true = 1.0
    A_true = 1.2
    w_true = jnp.pi/5
    # c_true   = 0.5
    # k_true   = 2.0
    # f_true   = 0.5
    # w_true   = 1.05
    parameters_true = [[m_true, mu_true, k_true, A_true, w_true],]

    y_at_t_ref = integrate.solve_ivp(
        fun=model_true, 
        t_span=integration_horizon,
        y0=initial_condition,
        t_eval=t_discrete,
        args=parameters_true
    )["y"]

    m_guess = 1.0
    mu_guess = 8.53
    k_guess = 1.0
    A_guess = 1.2
    w_guess = jnp.pi/5
    parameters_guess = [m_guess, mu_guess, k_guess, A_guess, w_guess, NN_params]

    # y_at_t = integrate.solve_ivp(
    #     fun=model_hybrid, 
    #     t_span=integration_horizon,
    #     y0=initial_condition,
    #     t_eval=t_discrete,
    #     args=parameters_guess
    # )["y"]

    # y_euler = euler(model_true, integration_horizon, initial_condition, t_discrete, parameters_true)

    # plt.figure()
    # plt.subplot(121)
    # plt.plot(t_discrete, y_at_t_ref[0, :], label="Reference solution")
    # plt.plot(t_discrete, y_euler[0, :], label="Reference euler")
    # plt.legend()
    # plt.grid()
    # plt.subplot(122)
    # plt.plot(t_discrete, y_at_t_ref[1, :], label="Reference solution")
    # plt.plot(t_discrete, y_euler[1, :], label="Reference euler")
    # plt.legend()
    # plt.grid()
    # # plt.show()

    args = [integration_horizon, initial_condition, t_discrete, parameters_guess[:-1]]

    INPUTS, OUTPUTS = get_model_data(args)

    # Add an additional Dimension to input and output for multiple batches
    INPUTS = jnp.expand_dims(INPUTS, 2)
    OUTPUTS = jnp.expand_dims(OUTPUTS, 2)

    schedule = optax.linear_schedule(
        init_value = 1e-3,
        end_value = 1e-7,
        transition_steps=100000
        )

    optimizer = optax.adam(learning_rate=schedule)
    params, losses = fit(NN_params, optimizer, INPUTS, OUTPUTS, 100000, args)

    # odenet_params, odenet_losses = odenet_fit(odenet_params, optimizer, INPUTS, OUTPUTS, 10000, args)
    













    # num_epochs = 100000

    # losses = []

    # for epoch in range(num_epochs):
        
    #     alpha = 1/(100000*(epoch+1))

    #     loss, NN_params = update(y_euler, NN_params, args, lr=alpha)
        
    #     integration_horizon, initial_condition, t_discrete, parameters_guess = args


    #     # if epoch % 50 == 0:
    #     #     print(loss)
    #     print(f'Epoch {epoch}, loss: {loss}')
    #     losses.append(loss)
        

    #     if epoch % 33000 == 0 or epoch == num_epochs-1:
    #         prediction = euler_NN(
    #             y0=initial_condition,
    #             t_eval=t_discrete,
    #             params=NN_params,
    #             args=parameters_guess
    #         )
    #         plt.figure()
    #         plt.plot(losses)
    #         plt.figure()
    #         plt.subplot(121)
    #         plt.plot(t_discrete, y_euler[0, :], label="Reference euler")
    #         plt.plot(t_discrete, prediction[0], label="Prediction")
    #         plt.legend()
    #         plt.grid()
    #         plt.subplot(122)
    #         plt.plot(t_discrete, y_euler[1, :], label="Reference euler")
    #         plt.plot(t_discrete, prediction[1], label="Prediction")
    #         plt.legend()
    #         plt.grid()
    #         plt.show()

    #     pass
