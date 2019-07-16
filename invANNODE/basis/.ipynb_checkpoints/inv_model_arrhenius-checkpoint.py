#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:45:50 2019

@author: ggusmao3
"""

# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
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
    return (scale * random.normal(key, (m,)))#, scale * random.normal(b_key, (n,))


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key, scale):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k, scale) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def init_model_params(size, key, scale):
 key = random.split(key,2)[-1] 
 return [random_model_params(s, key, scale) for s in size]


@jit
def transfer_fun(x):
    #return np.maximum(0, x)
    #return np.nan_to_num(x / (1.0 + np.exp(-x)))
    return x / (1.0 + np.exp(-x))
    #return 2./(1.+np.exp(-2.*x))-1.
    #return 0.5*np.tanh(x) + 0.5*x / (1.0 + np.exp(-x))
    #return(x)

# +
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


# -

@jit
def diff_state(params,t):
        i = np.arange(len(t))
        #return (jacobian(batched_state,argnums=1)(params,t)[i,:,i,0])
        return np.nan_to_num(jacfwd(lambda t : batched_state(params,t))(t)[i,:,i,0])

dh = [1,-1,2,-2,-2,2]


# +
@jit
def model(model_params,batch):
    #print('model x: {}'.format(x))
    #print('model params: {}'.format(model_params))
    x, t = batch
    c = np.abs(np.nan_to_num(x[:-1]))
    T = np.abs(np.nan_to_num(x[-1]))
    k0, ear = model_params
    k = np.nan_to_num(np.abs(k0)*np.exp(-np.abs(ear)/T))
    return np.array([[-2*k[0]*x[0]**2+2*k[1]*x[1]*x[2]],
                     [k[0]*x[0]**2-k[1]*x[1]*x[2]-k[2]*x[1]+k[3]*x[3]**2],
                     [k[0]*x[0]**2-k[1]*x[1]*x[2]-k[4]*x[3]*x[2]+k[5]*x[4]],
                     [2*k[2]*x[1]-2*k[3]*x[3]**2-k[4]*x[2]*x[3]+k[5]*x[4]],
                     [k[4]*x[2]*x[3]-k[5]*x[4]],
                     [(k[0]*x[0]**2)*dh[0]+\
                      (k[1]*x[1]*x[2])*dh[1]+\
                      (k[2]*x[1])*dh[2]+\
                      (k[3]*x[3]**2)*dh[3]+\
                      (k[4]*x[2]*x[3])*dh[4]+\
                      (k[5]*x[4])*dh[5]]])

batched_model = vmap(model,in_axes=(None,0))#, in_axes=(None, 0)
# -

model_params0 = np.array([[.9,.1,2.,.4,2.,0.5],[.3,.4,.5,.1,.7,.8]])
#model_params0 = np.array([1.,1.])
bc0=np.array([0.1,0.2,0.3,0.35,0.05,.5])
#bc0=np.array([1.0,0.,0.])

# +
 %%time

tmax = 30.
t = np.logspace(0,np.log10(tmax+1),12).reshape((-1,1))-1.
from scipy.integrate import solve_ivp

def ode(t,C):
    return model(model_params0,[C,[t]]).flatten()

n_points = int(2*tmax)

t_eval = np.logspace(0,np.log10(tmax),n_points)-1.
#t_eval = np.linspace(0,tmax,n_points)
sol = solve_ivp(ode, (0, tmax), bc0, t_eval = t_eval)

plt.figure(figsize=[10,10*2./3])
t = sol.t
x0 = sol.y.T
x = x0#+random.normal(random.PRNGKey(0),sol.y.T.shape)*0.025
lines = plt.plot(sol.t, x0[:,:-1])
plt.plot(sol.t, x[:,:-1],'.-',ms=5,lw=0.5)
#plt.legend(['A', 'B', 'C', 'D','E','T'])
plt.xlabel('Time')
plt.ylabel('C');
#plt.gca().set_xscale('log')
#plt.gca().set_yscale('log')
plt.legend()
plt.twinx()
lines += plt.plot(sol.t, x0[:,-1],c='black')
plt.plot(sol.t, x[:,-1],'.-',ms=5,lw=0.5,color='black')
plt.ylabel('T');
#plt.gca().set_xscale('log')
#plt.gca().set_yscale('log')

plt.legend(iter(lines), ['A', 'B', 'C','D','E','T'])

t = t.reshape([-1,1])
t_scale = t.max()
data = (x,t/t_scale)
t_err = np.linspace(data[1].min(),data[1].max()*1.01,len(t)*2).reshape(-1,1)


# -

@jit
def get_errors(nn_params,model_params,batch):
    x, t = batch
    pred_x = (batched_state(nn_params, t))
    pred_x_err = (batched_state(nn_params, t_err))
    err_model = 1e-2*((diff_state(nn_params,t_err)[:,:]-batched_model(model_params,[pred_x_err,t_err*t_scale])[:,:,0]))**2
    err_data  = (x-pred_x)**2#+(np.log(1.+np.abs(x))-np.log(1.+np.abs(pred_x)))**2#((x-pred_x)**2).sum(axis=0).mean()
    return err_data, err_model        


@jit
def loss(nn_params, model_params, batch):
    #print('pred[0]: {}'.format(pred_x[0]))
    #print('sum pred[0] - bc: {}'.format(sum((pred_x[0]-bc)**2))) 
    
    return np.array([_.mean() for _ in get_errors(nn_params,model_params,batch)]).sum()


# + {"active": ""}
# layer_sizes = [1, 10, 6]
# model_size = [6,6]
# nn_scale = .01
# model_scale = .00000000001
#
# key = random.PRNGKey(0)
#
# nn_params = init_network_params(layer_sizes, key, nn_scale)
# model_params = init_model_params(model_size, key, model_scale)
# -

num_epochs = 100
num_eras = int(1e20)
batch_size = n_points

# +
opt_init_nn, opt_update_nn, get_params_nn = optimizers.adam(1e-3, b1=0.9, b2=0.9999, eps=1e-100)#, b1=0.1, b2=0.999, eps=1e-10)
opt_init_model, opt_update_model, get_params_model = optimizers.adam(1e-3, b1=0.9, b2=0.9999, eps=1e-100)
    
"""
@jit
def step_nn(i, opt_state_nn, opt_state_model, batch):
    nn_params = get_params_nn(opt_state_nn)
    model_params = get_params_model(opt_state_model)
    grads_nn = grad(loss,argnums=0)(nn_params, model_params, batch)    
    return opt_update_nn(i, grads_nn, opt_state_nn)

@jit
def step_model(i, opt_state_nn, opt_state_model, batch):
    nn_params = get_params_nn(opt_state_nn)
    model_params = get_params_model(opt_state_model)
    grads_model = grad(loss,argnums=1)(nn_params, model_params, batch)    
    return opt_update_model(i, grads_model, opt_state_model)
"""
@jit
def step(i, opt_state_nn, opt_state_model, batch):
    nn_params = get_params_nn(opt_state_nn)
    model_params = get_params_model(opt_state_model)
    grads_model = grad(loss,argnums=1)(nn_params, model_params, batch)  
    grads_nn = grad(loss,argnums=0)(nn_params, model_params, batch)  
    return [opt_update_nn(i, grads_nn, opt_state_nn), opt_update_model(i, grads_model, opt_state_model)]
 
itercount    = itertools.count()        

opt_state_nn = opt_init_nn(nn_params)
opt_state_model = opt_init_model(model_params)

for j in range(num_eras):
    sel = random.shuffle(random.PRNGKey(j),np.arange(n_points))    
    #err_data, err_model = get_errors(nn_params,model_params,data)
    #err_dist = err_data.mean(axis=1)+err_model.mean(axis=1)
    #err_dist = err_dist/err_dist.sum()     
    #sel = choice(np.arange(len(data[1])),batch_size,False,err_dist.flatten())
    batch = tuple(_[sel[:batch_size],:] for _ in data)
    #batch = data
    for i in range(int(num_epochs)):
        nn_params = get_params_nn(opt_state_nn)
        model_params = get_params_model(opt_state_model)
        it = next(itercount)
        opt_state_nn, opt_state_model    = step(it, opt_state_nn, opt_state_model, batch) 
        #print('--.--')
    nn_params = get_params_nn(opt_state_nn)
    model_params = get_params_model(opt_state_model)   
    loss_it_sample = loss(nn_params, model_params, batch)    
    loss_it_batch = loss(nn_params, model_params, data)
    err_data, err_model = [_.mean() for _ in get_errors(nn_params,model_params,data)]
    print('Iteration: {:4d}, Loss Batch: {:.5e}, Loss Data: {:.5e}, Fit Data: {:.5e}, Fit Model: {:.5e}'.format(i,loss_it_sample,loss_it_batch,err_data,err_model))
    clear_output(wait=True)    


# +
#print((batched_state(nn_params,model_params,x)).sum(axis=1))
#print((sol.y).sum(axis=0))
def ode2(t,C):
        return model([np.abs(_) for _ in model_params],[C,[t]]).flatten()

sol2 = solve_ivp(ode2, (0, tmax), bc0, t_eval = t_eval)

plt.figure(figsize=[10,10*2./3])
lines = plt.plot(t, x[:,:-1])
plt.plot(t.flatten(),batched_state(nn_params, t/t_scale)[:,:-1],'-o',lw=0.2,ms=2)
plt.plot(sol2.t, (sol2.y.T)[:,:-1],'-.',lw=0.75,ms=2)
plt.legend(['A', 'B', 'C', 'D'])
plt.xlabel('Time')
plt.ylabel('C');
plt.legend(iter(lines), ['A', 'B', 'C','D','E'])
plt.figure(figsize=[10,10*2./3])
plt.plot(sol.t, x[:,-1],c='black')
plt.plot(t.flatten(),batched_state(nn_params, t/t_scale)[:,-1],'-o',lw=0.2,ms=2)
plt.plot(sol2.t, (sol2.y.T)[:,-1],'-.',lw=0.75,c='black')
plt.ylabel('T');
#plt.gca().set_yscale('log')
#plt.gca().set_xscale('log')



plt.figure(figsize=[10,10*2./3])
plt.plot(t.flatten(),diff_state(nn_params, t/t_scale)[:,:-1],'-o',lw=0.2,ms=2)
plt.twinx()
plt.plot(t.flatten(),diff_state(nn_params, t/t_scale)[:,-1],'-o',lw=0.2,ms=2)


display([(_) for _ in model_params])
display([(_) for _ in model_params0])
#display(dict(zip(model_params0[0],model_params0[0])))
#display(dict(zip(model_params0[1],model_params[1])))
#plt.gca().set_xscale('log')
#plt.gca().set_yscale('log')
# -
err_data, err_model = get_errors(nn_params,model_params,data)

        err_data, err_model = get_errors(nn_params,model_params,data)
        err_dist = err_data.mean(axis=1)#+err_model.mean(axis=1)
        err_dist = err_dist/err_dist.sum()     

err_data.mean(axis=1)

err_model.mean(axis=1)

err_dist

f=model_params0[0]
(f)

model_params0[0]
