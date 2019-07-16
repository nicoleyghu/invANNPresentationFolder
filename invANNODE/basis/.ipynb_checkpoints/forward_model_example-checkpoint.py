#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:45:50 2019

@author: ggusmao3
"""

# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
import jax.numpy as np
from jax import grad, jit, vmap, jacobian, jacfwd, jacrev
from jax import random
from jax.scipy.special import logsumexp
from jax.experimental import optimizers
from jax.config import config
config.update("jax_debug_nans", True)
config.update('jax_enable_x64', True)
import time
import itertools

"""### Hyperparameters
Let's get a few bookkeeping items out of the way.
"""

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
    ##print(key)
    w_key, b_key = random.split(key)
    ##print(np.shape(w_key))
    ##print(np.shape(b_key))
    #raise(Exception)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

@jit
def transfer_fun(x):
    #return np.maximum(0, x)
    #return np.nan_to_num(x / (1.0 + np.exp(-x)))
    return x / (1.0 + np.exp(-x))
    #return np.tanh(x)
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
    y = (np.dot(final_w, activations) + final_b)**2
    #y = y / y.sum()
    return y


# Make a batched version of the `state` function
batched_state = vmap(state, in_axes=(None,0))#, in_axes=(None, 0))


# + {"active": ""}
# def accuracy(params, t, targets):
#    target_class = np.argm25ax(targets, axi25s=1)
#    stateed_class = np.argmax(batched_state(params, t), axis=1)
#    return np.mean(stateed_class == target_class)

# + {"active": ""}
# def accuracy(params, t, target_x):
#    pred_x = batched_state(params,t)
#    y    = pred_x-np.mean(pred_x) 
#    y_ = target_x-np.mean(target_x)
#    ##print(np.shape(y),np.shape(y_))
#    return np.mean(np.dot(y.T,y_)**2/(np.dot(y.T,y)*np.dot(y_.T,y_)))*np.exp(-np.mean((pred_x-target_x)**2).mean())
# -

@jit
def diff_state(params,t):
        i = np.arange(len(t))
        return (jacobian(batched_state,argnums=1)(params,t)[i,:,i,0])

# +
@jit
def model(params_model,x):
        #print('model x: {}'.format(x))
        #x = np.abs(np.nan_to_num(x))
        return np.array([[-params_model[0]*(x[0]**2)+params_model[3]*x[3]**2+params_model[3]*x[2]**2],
                                         [params_model[0]*(x[0]**2) - params_model[1]*((x[1])**1.5)],
                                         [params_model[1]*((x[1])**1.5)-params_model[3]*x[2]**2],
                                         [params_model[2]*(x[1]*x[0])-params_model[3]*x[3]**2]])

batched_model = vmap(model, in_axes=(None,0))#, in_axes=(None, 0))
# -

params_model = np.array([1.,1.,0.5,.1])
bc=np.array([1.0,0.,0.,0.])

@jit
def loss(params, t):
    pred_x = (batched_state(params, t))
    err = (((diff_state(params,t))-(batched_model(params_model,pred_x)[:,:,0]))**2).mean()
    #print('pred[0]: {}'.format(pred_x[0]))
    #print('sum pred[0] - bc: {}'.format(sum((pred_x[0]-bc)**2)))
    return err+((pred_x[0]-bc)**2).mean()#+((pred_x.sum(axis=1)-1.)**2).mean()

# +
# %%time
#@jit
#def update(params, t):
#    grads = jacobian(loss,argnums=0)(params, t)
#    #grads = [tuple(i.sum(axis=0) for i in j) for j in grads]
#    return [(w - step_size * dw, b - step_size * db)
#                    for (w, b), (dw, db) in zip(params, grads)]
# Use optimizers to set optimizer initialization and update functions
r=1
layer_sizes = [1, 12, 4]
param_scale = .1
num_epochs = 1000
num_eras = 20
batch_size = 50
params = init_network_params(layer_sizes, random.PRNGKey(0))

opt_init, opt_update, get_params = optimizers.adam(1e-3,eps=1e-50)#sgd(1e-2)#(1e-2, gamma=0.9, eps=1e-6)#(1e-4)#, b1=0.01, b2=0.9, eps=1e-12)#, b1=0.1, b2=0.999, eps=1e-10)

x = np.logspace(0,np.log10(51),300).reshape((-1,1))-1.

@jit
def step(i, opt_state, t):
    params = get_params(opt_state)
    grads = jacobian(loss,argnums=0)(params, t)    
    return opt_update(i, grads, opt_state)

couter = itertools.count()
opt_state = opt_init(params)
for j in range(int(num_eras)):
        batch = np.concatenate((np.array([x[0]]),x[random.shuffle(random.PRNGKey(j),np.arange(len(x)))[:batch_size],:]))
        for i in range(int(num_epochs)):
            #print(i)
            opt_state = step(next(couter), opt_state, batch)
            #print('--.--')    
params = get_params(opt_state)
#print(np.shape(opt_state.packed_state[3][0]))
batched_state(params,x),x

# +
# %%time
from scipy.integrate import solve_ivp
def ode(t, C):
        Ca, Cb, Cc, Cd = C
        dCadt = -k1 * Ca**2 + k4*Cd**2 + k4*Cc**2
        dCbdt = k1 * Ca**2 - k2 * Cb**1.5
        dCcdt = k2 * Cb**1.5 - k4*Cc**2
        dCddt = k3 * Ca * Cb - k4*Cd**2
        return [dCadt, dCbdt, dCcdt, dCddt]

C0 = [1.0, 0.0, 0.0, 0.0]
k1 = 1
k2 = 1
k3 = 0.5
k4 = .1
sol = solve_ivp(ode, (0, 50), C0, t_eval = x.flatten())
# -

import matplotlib.pyplot as plt
#print((batched_state(params,x)).sum(axis=1))
#print((sol.y).sum(axis=0))
plt.plot(sol.t, sol.y.T)
plt.plot(x.flatten(),batched_state(params,x),'-o',lw=0.2,ms=2)
plt.legend(['A', 'B', 'C', 'D'])
plt.xlabel('Time')
plt.ylabel('C');


