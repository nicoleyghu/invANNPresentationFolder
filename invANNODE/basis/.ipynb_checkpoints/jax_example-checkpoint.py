# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: ML
#     language: python
#     name: ml
# ---

# +
import jax.numpy as np
from jax import random
from jax.experimental import stax
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax

# Use stax to set up network initialization and evaluation functions
net_init, net_apply = stax.serial(
    Conv(1, (3, 3), padding='SAME'), Relu,
    Conv(4, (3, 3), padding='SAME'), Relu,
    MaxPool((2, 2)), Flatten,
    Dense(32), Relu,
    Dense(10), LogSoftmax,
)

# Initialize parameters, not committing to a batch shape
rng = random.PRNGKey(0)
in_shape = (-1, 4, 4, 1)
out_shape, net_params = net_init(rng, in_shape)

# Apply network to dummy inputs
inputs = np.zeros((32, 4, 4, 1))
predictions = net_apply(net_params, inputs)

from jax.experimental import optimizers
from jax import jit, grad

# Define a simple squared-error loss
def loss(params, batch):
  inputs, targets = batch
  predictions = net_apply(params, inputs)
  return np.sum((predictions - targets)**2)

# Use optimizers to set optimizer initialization and update functions
opt_init, opt_update, get_params = optimizers.momentum(step_size=1e-3, mass=0.9)

# Define a compiled update step
@jit
def step(i, opt_state, batch):
  params = get_params(opt_state)
  print(np.shape(params))
  g = grad(loss)(params, batch)
  print(np.shape(g))
  return opt_update(i, g, opt_state)

# Dummy input data stream
data_generator = ((np.zeros((32, 4, 4, 1)), np.zeros((32, 10)))
                  for _ in range(10))

# Optimize parameters in a loop
opt_state = opt_init(net_params)
for i in range(10):
  print(i)
  opt_state = step(i, opt_state, next(data_generator))
net_params = get_params(opt_state)
# -


