#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:45:50 2019

@author: ggusmao3
"""

# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
import pickle as pk
import jax.numpy as np
from numpy.random import choice
from jax import grad, jit, vmap, jacobian, jacfwd, jacrev
from jax import random
from jax.scipy.special import logsumexp
from jax.experimental import optimizers
from jax.config import config
from jax.tree_util import tree_map
config.update("jax_debug_nans", True)
config.update('jax_enable_x64', True)
import time
from IPython.display import clear_output
from matplotlib import pyplot as plt
import itertools

"""### Hyperparameters
Let's get a few bookkeeping items out of the way.
"""

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_M_params(m, key, scale=1e-2):
    #w_key, b_key = random.split(key)
    #print(tuple(scale * random.normal(key, (m, 1))))
    return (scale * random.normal(key, (m,)))#, scale * random.normal(b_key, (n,))


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key, scale):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k, scale) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def init_M_params(size, key, scale):
 key = random.split(key,2)[-1] 
 return random_M_params(size, key, scale)


@jit
def transfer_fun(x):
    #return np.maximum(0, x)
    #return np.nan_to_num(x / (1.0 + np.exp(-x)))
    #return x / (1.0 + np.exp(-x))
    #return np.tanh(x)
    #return np.exp(-x**2)    
    #return x
    return np.nan_to_num(np.true_divide(2.,(1.+np.exp(-2.*x)))-1)
    #return 0.5*np.tanh(x) + 0.5*x / (1.0 + np.exp(-x))
    #return(x)

@jit
def state(params, t):
    # per-example stateions
    activations = t
    for w, b in params[:-1]:
        outputs = np.dot(w, activations) + b
        activations = transfer_fun(outputs)
    
    final_w, final_b = params[-1]
    y = (np.dot(final_w, activations) + final_b)
    #y = y / y.sum()
    return y


# Make a batched version of the `state` function
batched_state = vmap(state, in_axes=(None,0))#, in_axes=(None, 0))


@jit
def diff_state(params,t):
        i = np.arange(len(t))
        #return (jacobian(batched_state,argnums=1)(params,t)[i,:,i,0])
        return np.nan_to_num(jacfwd(lambda t : batched_state(params,t))(t)[i,:,i,0])

# ## Polynomial Roots

# +
layer_sizes_poly = [1, 200, 2000, 1]
#model_params = model_params0.copy()
p_total = 100.
alpha = 0.01
nn_scale = .1
model_scale = 1.
num_epochs = 100
num_eras = 10000
latent_variables = 4

batch_size = 99#n_points-1
tf = 20.


key = random.PRNGKey(0)

nn_params_poly = init_network_params(layer_sizes_poly, key, nn_scale)

order = 3
t = np.linspace(-1,1,10).reshape(-1,1)

@jit
def fun(x):
        return x**2

#model_params = init_M_params(model_size, key, model_scale)


# -

@jit
def get_errors(nn_params_poly,batch):
    x, t = batch
    pred_x = (batched_state(nn_params_poly, t))
    #err_M = (diff_state(batched_state(nn_params_chain, pred_x)))**2
    err_data    =    (x-pred_x)#+(diff_state(nn_params_int,t)-pred_x[:,:-latent_variables])**2
    return err_data#, 0.*err_M


@jit
def loss(nn_params_poly, batch):
    #print('pred[0]: {}'.format(pred_x[0]))
    #print('sum pred[0] - bc: {}'.format(sum((pred_x[0]-bc)**2))) 
    return sum([_.var() for _ in get_errors(nn_params_poly,batch)])


# +
# #%prun
opt_init_poly, opt_update_poly, get_params_poly = optimizers.adam(1e-3, b1=0.9, b2=0.9999, eps=1e-5)#, b1=0.1, b2=0.999, eps=1e-10)
    
@jit
def step(i, opt_state_poly, batch):
    nn_params_poly = get_params_poly(opt_state_poly)
    grads_poly = grad(loss,argnums=0)(nn_params_poly, batch)
    return [opt_update_poly(i, grads_poly, opt_state_poly)]
 
itercount    = itertools.count()        

opt_state_poly = opt_init_poly(nn_params_poly)

for j in range(num_eras):
        #err, _ = get_errors(nn_params_poly,nn_params_chain,t)
        #err_dist = err.mean(axis=1)#+err_L.mean(axbis=1)#+err_B.mean(axis=1)#+err_M.mean(axis=1)
        #err_dist = err_dist/err_dist.sum()     
        #sel = np.concatenate((np.array([0]),choice(np.arange(len(data[0])),batch_size,False,err_dist.flatten())))
        batch = tuple([fun(t),t])
        #batch = data
        for i in range(int(num_epochs)):
            it = next(itercount)
            opt_state_poly, = step(it, opt_state_poly, batch)
            nn_params_poly = get_params_poly(opt_state_poly)
            loss_it_sample = loss(nn_params_poly, batch)    
            #loss_it_data = loss(nn_params_poly, nn_params_chain, data)
            #print('Iteration: {:4d}, Loss Batch: {:.7e}, Loss ata: {:.7e}'.format(i,loss_it_sample,loss_it_data))
            print('Iteration: {:4d}, Loss Batch: {:.7e}, '.format(i,loss_it_sample)) 
            clear_output(wait=True)
            #print('--.--')                                
nn_params_poly = get_params_nn(opt_state_poly)
# -

plt.plot(t,fun(t))
plt.plot(t,batched_state(nn_params_poly,t),'.-')


