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
from IPython.display import clear_output
from matplotlib import pyplot as plt

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
def random_model_params(m, key, scale=1e-2):
    #w_key, b_key = random.split(key)
    #print(tuple(scale * random.normal(key, (m, 1))))
    return np.abs(scale * random.normal(key, (m,)))#, scale * random.normal(b_key, (n,))


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key, scale):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k, scale) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def init_model_params(size, key, scale):
 key = random.split(key,2)[-1] 
 return random_model_params(size, key, scale)


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
    y = np.dot(final_w, activations) + final_b
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
def model(model_params,x):
        #print('model x: {}'.format(x))
        #x = np.abs(np.nan_to_num(x))
        mp = model_params
        return np.array([[-mp[0]*(x[0])],
                                         [mp[0]*(x[0]) - mp[1]*((x[1]))],
                                         [mp[1]*((x[1]))]])
        
        """
        return np.array([[-mp[0]*(x[0]**2)+mp[3]*x[3]**2+mp[3]*x[2]**2],
                                         [mp[0]*(x[0]**2) - mp[1]*((x[1])**1.5)],
                                         [mp[1]*((x[1])**1.5)-mp[3]*x[2]**2],
                                         [mp[2]*(x[1]*x[0])-mp[3]*x[3]**2]])
        """

batched_model = vmap(model, in_axes=(None,0))#, in_axes=(None, 0))
# -

#model_params = np.array([1.,1.,0.5,1.])
model_params0 = np.array([1.,1.])
#bc=np.array([1.0,0.,0.,0.])
bc0=np.array([1.0,0.,0.])

# +
# %%time

tmax = 10
t = np.logspace(0,np.log10(tmax+1),60).reshape((-1,1))-1.
from scipy.integrate import solve_ivp

def ode(t,C):
        return model(model_params0,C).flatten()

n_points = 50

t_eval = np.logspace(0,np.log10(tmax+1),n_points)-1.
#t_eval = np.linspace(0,20,n_points)
sol = solve_ivp(ode, (0, tmax), bc0, t_eval = t_eval)

t = sol.t
x0 = sol.y.T
x = x0#+random.normal(random.PRNGKey(0),sol.y.T.shape)*0.025
plt.plot(sol.t, x0)
plt.plot(sol.t, x,'.-',ms=5,lw=0.5)
#plt.legend(['A', 'B', 'C', 'D'])
plt.legend(['A', 'B', 'C'])
plt.xlabel('Time')
plt.ylabel('C');

t = t.reshape([-1,1])

batch = (x,t)


# -

@jit
def loss(nn_params, model_params, batch):
    x, t = batch
    pred_x = (batched_state(nn_params, t))
    err_model = ((diff_state(nn_params,t).reshape([-1,1])-batched_model(model_params_,pred_x)[:,:,0].reshape([-1,1]))**2)
    err_data    =    ((x-pred_x)**2)#((x-pred_x)**2).sum(axis=0).mean()
    #print('pred[0]: {}'.format(pred_x[0]))
    #print('sum pred[0] - bc: {}'.format(sum((pred_x[0]-bc)**2)))
    return ((1.+err_data.var()))*err_data.mean()#+err_model.mean()#+np.sum([-np.nan_to_num(np.log(-i)) for i in model_params])#((pred_x[0]-bc)**2).mean()#+((pred_x.sum(axis=1)-1.)**2).mean()

# +
# %%time
#@jit
#def update(params, t):
#    grads = jacobian(loss,argnums=0)(params, t)
#    #grads = [tuple(i.sum(axis=0) for i in j) for j in grads]
#    return [(w - step_size * dw, b - step_size * db)
#                    for (w, b), (dw, db) in zip(params, grads)]
# Use optimizers to set optimizer initialization and update functions

layer_sizes = [1, 15, 3]
model_size = 2
param_scale = .01
model_scale = 1.5
num_epochs = 1000

key = random.PRNGKey(0)
nn_params = init_network_params(layer_sizes, key, param_scale)
model_params = init_model_params(model_size, key, model_scale)

opt_init_nn, opt_update_nn, get_params_nn = optimizers.adam(1e-3, b1=0.9, b2=0.999, eps=1e-90)#, b1=0.1, b2=0.999, eps=1e-10)
opt_init_model, opt_update_model, get_params_model = optimizers.adam(1e-2, b1=0.9, b2=0.999, eps=1e-90)

for i in range(10000):
        
        @jit
        def step_nn(i, opt_state, batch):
            nn_params = get_params_nn(opt_state_nn)
            model_params = get_params_model(opt_state_model)
            grads_nn = grad(loss,argnums=0)(nn_params, model_params, batch)    
            return opt_update_nn(i, grads_nn, opt_state_nn)

        #"""
        @jit
        def step_model(i, opt_state, batch):
            nn_params = get_params_nn(opt_state_nn)
            model_params = get_params_model(opt_state_model)
            model_params_ = model_params
            grads_model = grad(loss,argnums=1)(nn_params, model_params, batch)    
            return opt_update_model(i, grads_model, opt_state_model)
        #"""

        opt_state_nn = opt_init_nn(nn_params)
        opt_state_model = opt_init_model(model_params)
        model_params_ = model_params
        for i in range(int(num_epochs)):
            nn_params = get_params_nn(opt_state_nn)
            model_params = get_params_model(opt_state_model)
            print('Iteration: {}, Loss: {}'.format(i,loss(nn_params, model_params, batch)))
            opt_state_nn        = step_nn(i, opt_state_nn, batch)
            #opt_state_model = step_model(i, opt_state_model, batch)
            clear_output(wait=True)
            #print('--.--')    
        print('Iteration: {}, Loss: {}'.format(i,loss(nn_params, model_params, batch)))
nn_params = get_params_nn(opt_state_nn)
params_model = get_params_model(opt_state_model)
#print(np.shape(opt_state.packed_state[3][0]))
# -

import matplotlib.pyplot as plt
#print((batched_state(nn_params,model_params,x)).sum(axis=1))
#print((sol.y).sum(axis=0))
plt.plot(t, x)
plt.plot(t.flatten(),batched_state(nn_params, t),'-o',lw=0.2,ms=2)
plt.legend(['A', 'B', 'C', 'D'])
plt.xlabel('Time')
plt.ylabel('C');

model_params




