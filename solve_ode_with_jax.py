import jax.numpy as np
from jax import grad
from jax import random
from jax import vmap
from jax import jit
import matplotlib.pyplot as plt

# Nonlinearity
def sigmoid(x):
    return 1./(1. + np.exp(-x))

# Forward function of the Neural Network
def f(params, x):
    w0 = params[:10]
    b0 = params[10:20]
    w1 = params[20:30]
    b1 = params[30]
    x = sigmoid(x*w0 + b0)
    x = np.sum(x*w1) + b1
    return x

# Randomly init the NN parameters
key = random.PRNGKey(0)
params = random.normal(key, shape=(31,))

# gradient of Network with respect to the inputs
dfdx = grad(f, 1)

# Test Second derivative
dffdxx = grad(grad(f, 1), 1)

# Inputs
inputs = np.linspace(0.0, 10., num=1001)

# Vectorize the forward function of the Network and get the gradient wrt to inputs
f_vect = vmap(f, (None, 0))
dfdx_vect = vmap(dfdx, (None, 0))
dffdxx_vect = vmap(dffdxx, (None, 0))




# The loss function which contains the differential equation
# y' = -2xy (expressed as y'+2xy = 0)
# with initial condition y(0) = 1 (expressed as y(0)-1 = 0)
# x represents the input values
# y is the function we are seeking in our case replaced by the NN
# y' is the derivative of the seeked function wrt to the inputs x 
#    in our case this is the derivative of the NN wrt to inputs x
@jit
def loss(params, inputs):
    # eq = dffdxx_vect(params, inputs) - 0.0*(1.0 - f_vect(params, inputs)**2)*dfdx_vect(params, inputs) + f_vect(params, inputs)
    eq = dffdxx_vect(params, inputs) + 0*dfdx_vect(params, inputs) + f_vect(params, inputs)
    ic = f(params, 0.) - 5.
    return np.mean(eq**2) + ic**2

# Differentiate the loss function with respect to the NN parameters (like always)
grad_loss = jit(grad(loss, 0))

# Training of the NN
epochs = 10000
learning_rate = 0.001
momentum = 0.9
velocity = 0.

for epoch in range(epochs):
    if epoch % 100 == 0:
        print('epoch: %3d loss: %.6f' % (epoch, loss(params, inputs)))
    # Gradient descent with Nesterov Accelerated Gradient (NAG)
    gradient = grad_loss(params + momentum*velocity, inputs)
    velocity = momentum*velocity - learning_rate*gradient
    params += velocity


# plt.plot(inputs, np.cos(inputs), label='exact')
plt.plot(inputs, f_vect(params, inputs), label='approx')
plt.legend()
plt.show()