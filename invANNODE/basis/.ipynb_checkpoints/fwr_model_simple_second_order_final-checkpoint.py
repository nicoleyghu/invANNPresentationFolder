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
def random_params(m, key, scale=1e-2):
    #w_key, b_key = random.split(key)
    #print(tuple(scale * random.normal(key, (m, 1))))
    return (scale * random.normal(key, (m,)))#, scale * random.normal(b_key, (n,))


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key, scale):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k, scale) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def init_params(size, key, scale):
 key = random.split(key,2)[-1] 
 return random_params(size, key, scale)


@jit
def transfer_fun(x):
    #return np.maximum(0, x)
    #return np.sin(x)
    #return np.log(np.cosh(x))
    return 2./(1.+np.exp(-2.*x))-1.
    # return 
    #return np.nan_to_num(x / (1.0 + np.exp(-x)))
    #return x / (1.0 + np.exp(-x))
    #return np.exp(-x**2)
    #return 2./(1.+np.exp(-2.*x))-1.    
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
def diff_state(params,t,pos):
    i = np.arange(len(t))
    #return (jacobian(batched_state,argnums=1)(params,t)[i,:,i,0])
    return (jacfwd(lambda t : batched_state(params,t))(t)[i,:,i,pos])

# ### Forward NN-based PDE
# To solve forward problems, in general random or error-based sampling does not change.
#
# #### MAYBE WE CAN USE RELU WITH A DENSE MATRIX AND A VANISHING NONLINEAR OPERATOR TO FIND BEST COLOCATTION MATRICES@

# + {"active": ""}
# @jit
# def source(source_params,x):
#         #print('source x: {}'.format(x))
#         #x = np.abs(np.nan_to_num(x))
#         k = source_params
#         return np.array([
#                         [-k[0]*x[0]*x[6]+k[1]*x[3]],
#                         [-k[2]*x[1]*x[6]+k[3]*x[4]],
#                         [ k[6]*x[5]-k[7]*x[2]*x[6]],
#                         [ k[0]*x[0]*x[6]-k[1]*x[3]-k[4]*x[3]*x[4]+k[5]*x[5]*x[6]],
#                         [ k[2]*x[1]*x[6]-k[3]*x[4]-k[4]*x[3]*x[4]+k[5]*x[5]*x[6]],
#                         [ k[4]*x[3]*x[4]-k[5]*x[5]*x[6]-k[6]*x[5]+k[7]*x[2]*x[6]],
#                         [-k[0]*x[0]*x[6]+k[1]*x[3]-k[2]*x[1]*x[6]+k[3]*x[4]\
#                         +k[4]*x[3]*x[4]-k[5]*x[5]*x[6]+k[6]*x[5]-k[7]*x[2]*x[6]]
#                         ])
#         
#         """
#         return np.array([[-mp[0]*(x[0])],0
#                          [mp[0]*(x[0]) - mp[1]*((x[1]))],
#                          [mp[1]*((x[1]))]])
#         """
# batched = vmap(source,in_axes=(None,0))#, in_axes=(None, 0)

# + {"active": ""}
# source_params0 = np.array([1.,2.,1.5,1.3,1.2*100.,2.*100.,3.,1.5])
# bc0=np.array([.5,.2,.3,0.,0.,0.,1.])
# -

@jit
def source(source_params,x):
    #print('source x: {}'.format(x))
    x = np.abs(np.nan_to_num(x))
    k = np.abs(source_params)
    return -k[0]*x
    
    """
    return np.array([[-mp[0]*(x[0])],0
                     [mp[0]*(x[0]) - mp[1]*((x[1]))],
                     [mp[1]*((x[1]))]])
    """
batched = vmap(source,in_axes=(None,0,None))#, in_axes=(None, 0)

source_params = np.array([3.])
bc0=np.array([1.])
bcf=np.array([0.])

layer_sizes = [1, 8, 1]
layer_sizes2 = [1, 8, 1]
source_size = len(source_params)
nn_scale = .01
D = 0.01
num_epochs = 100
num_eras = 100

# + {"active": ""}
# np.save('nn_p_1o',nn_p,allow_pickle=True)

# + {"active": ""}
# key = random.PRNGKey(0)
# try:
#         nn_p = [tuple(np.array(_,np.float64) for _ in __) for __ in np.load('nn_p_1o.npy',allow_pickle=True).tolist()]
# except:
#         nn_p = init_network_params(layer_sizes, key, nn_scale)
# -

key = random.PRNGKey(0)
nn_p = init_network_params(layer_sizes, key, nn_scale)
nn_p2 = init_network_params(layer_sizes2, key, nn_scale)

n_points = 20
batch_size = 20#n_points-1
tf = 1
#src = np.logspace(0,np.log10(tf+1),n_points).reshape(-1,1)-1.
src = np.linspace(-tf,tf,n_points).reshape(-1,1)
data = src


@jit
def get_errors_f(nn_p, nn_p2, batch):
    s = batch
    x_s = batched_state(nn_p, s)
    dx_s_ = batched_state(nn_p2, s)
    dx_s = diff_state(nn_p, s, 0)
    d2x_s_ = diff_state(nn_p2, s, 0)
    err = 1e5*(((d2x_s_-s**5))**2).sum(axis=1)#((s+1)**4/(x_s+1)**2)*
    err_coupling = (((dx_s_-dx_s))**2).sum(axis=1)#+((dx_s[0]-1)**2).sum(axis=1)#(((((s+1)**2)**2/(x_s+1)**2)*(d2x_s-1))**2).sum(axis=1)#+((dx_s[0]-1)**2).sum(axis=1)((s+1)**2/(x_s+1)**2)*
    err_bc =     (((batched_state(nn_p, np.array([[0.]])))**2).sum()+((batched_state(nn_p, np.array([[1.]])))**2).sum(axis=1))
    return err, err_coupling, err_bc 


@jit
def loss_f(nn_p, nn_p2, batch):
    __ = get_errors_f(nn_p, nn_p2, batch)     
    return sum([_.mean()    for _ in __])


# +
opt_init, opt_update, get_params = optimizers.adam(1e-3, b1=0.9, b2=0.9999, eps=1e-100)
opt_init2, opt_update2, get_params2 = optimizers.adam(1e-3, b1=0.9, b2=0.9999, eps=1e-100)
    
@jit
def step(i, opt_state, opt_state2, batch):
    nn_p = get_params(opt_state)    
    nn_p2 = get_params(opt_state2)    
    grads = grad(loss_f,argnums=0)(nn_p, nn_p2, batch)
    grads2 = grad(loss_f,argnums=1)(nn_p, nn_p2, batch)
    opt = opt_update(i, grads, opt_state)
    opt2 = opt_update2(i, grads2, opt_state2)
    return [opt, opt2]


# +
itercount    = itertools.count()        

opt_state = opt_init(nn_p)
opt_state2 = opt_init(nn_p2)

for j in range(num_eras):
    sel = random.shuffle(random.PRNGKey(j),np.arange(n_points))     
    #err, _, _ = get_errors_f(nn_p_space, nn_p_space_sub, data)
    #err_dist = err#+err_L.mean(axbis=1)#+err_B.mean(axis=1)#+err.mean(axis=1)
    #err_dist = err_dist/err_dist.sum()        
    #sel = np.concatenate((np.array([0]),choice(np.arange(len(data)),batch_size,False,err_dist.flatten())))
    batch = data[sel[:batch_size],:]
    #batch = data
    loss_it_data0 = np.inf
    loss_it_sample0 = np.inf
    for i in range(int(num_epochs)):
        #err, err_L, err_B, err_BC = get_errors_f(nn_p,data)
        print('Iteration: {:4d}, Loss Batch: {:.7e}, Loss Data: {:.7e}'.format(i,loss_it_sample0,loss_it_data0))
        opt_state, opt_state2 = step(next(itercount), opt_state, opt_state2, batch)
        nn_p = get_params(opt_state)
        nn_p2 = get_params(opt_state2)
        loss_it_sample = loss_f(nn_p, nn_p2, batch)
        loss_it_data = loss_f(nn_p, nn_p2, data)
        loss_it_sample0 = loss_it_sample
        loss_it_data0 = loss_it_data    
        clear_output(wait=True)
    #
nn_p = get_params(opt_state)
# -

get_errors_f(nn_p, nn_p2, batch)

plt.figure(figsize=[10,10*2./3],dpi=100)
tf = 1
src2 = np.linspace(-tf,tf,1000).reshape(-1,1)
plt.plot(src2, batched_state(nn_p,src2),'-',label='NN')
plt.plot(src, src**7/30.-1/30.*src,'o',label='function')
#plt.gca().set_yscale('log')
#plt.plot(src, diff_state(nn_p,src,0),'.',label='f')


