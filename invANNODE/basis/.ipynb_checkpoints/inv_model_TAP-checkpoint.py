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
import os
import pandas as pd
from IPython.display import Javascript

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
 return np.abs(random_M_params(size, key, scale))


@jit
def transfer_fun(x):
    #return np.maximum(0, x)
    #return np.nan_to_num(x / (1.0 + np.exp(-x)))
    #return x / (1.0 + np.exp(-x))
    #return np.tanh(x)
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

# ### Inverse TAP Problem
#
# To solve forward problems, in general random or error-based sampling does not change.
#
# #### MAYBE WE CAN USE RELU WITH A DENSE MATRIX AND A VANISHING NONLINEAR OPERATOR TO FIND BEST COLOCATTION MATRICES

# #### Import Data

# +
path = os.path.abspath('../../tap_data/lang_folder')
labels_g = ['CO','CO2']
labels_s = ['CO*','O*','*']
species = dict(zip(range(len(labels_g+labels_s)),labels_g+labels_s))
data_thin = []
data_out = []
df = pd.read_csv(path+'/input_file.csv',dtype= object,index_col= 0,header=None)
total_time = np.float64(df.loc[df[1] == 'Pulse Duration'][2].values[0])
total_steps = np.float64(df.loc[df[1] == 'Time Steps'][2].values[0])
T = np.float64(df.loc[df[1] == 'Reactor Temperature'][2].values[0]) # K
bc_s = np.float64(df.loc[df[1] == 'Initial Surface Composition'][2].values[0].split(',')) # nnmol/cm**3
pulse_fracs = np.float64(df.loc[df[1] == 'Pulse Ratio'][2].values[0].split(','))[:-1] # nnmol/cm**3
mw_g = np.float64(df.loc[df[1] == 'Mass List'][2].values[0].split(','))[:-1] # g/mol
pulse =    np.float64(df.loc[df[1] == 'Reference Pulse Size'][2].values[0]) # cm
length =    np.float64(df.loc[df[1] == 'Reactor Length'][2].values[0]) # cm
radius =    np.float64(df.loc[df[1] == 'Reactor Radius'][2].values[0]) # cm
alpha    =    np.float64(df.loc[df[1] == 'Catalyst Fraction'][2].values[0]) # fraction
area = np.pi*radius**2
cat_vol = length*area*alpha # cm**3
tot_sites = cat_vol*bc_s.sum()*1e-9 # mol
bc_s = bc_s/bc_s.sum() # frac
for _ in labels_g:
        try:
                data_thin += [pd.read_csv(path+'/thin_data/'+_+'.csv').values]
                data_out += [pd.read_csv(path+'/flux_data/'+_+'.csv').values]
        except:
                print('Error: {}.'.format(_))
                raise(Exception)

mw_s = np.array([mw_g[0], mw_g[1]-mw_g[0], 0])
R = 8.31446261815324 # m**3.Pa/K.mol
no_model_params = 3
#data = [_*R*T*1e-8 for _ in data] # bar
data_thin = [_*cat_vol for _ in data_thin] # nmol
pulse_n = (pulse*mw_g*pulse_fracs).sum()
# -

tspan = np.linspace(0,total_time,total_steps)
plt.figure(dpi=100)
plt.plot(tspan,(np.array(data_thin).std(axis=2)/np.array(data_thin).mean(axis=2)).T)
plt.gca().set_yscale('log')
plt.xlabel('time (s)')
plt.ylabel('relative deviation')
plt.title('Relative Deviation Across Thin Zone')
data_thin = pulse_n*np.array(data_thin).mean(axis=2).T
data_out = pulse_n*np.array(data_out).mean(axis=2).T
scale_thin = data_thin.max()
scale_out = data_out.max()
scale_t = tspan.max()
data0 = [np.log(data_thin/scale_thin+1.)]+\
                [np.log(data_out/scale_out+1.)]+\
                [tspan.reshape(-1,1)/scale_t]


# +
nn_thin_size = [1, 5, 5, len(species)]
nn_outlet_size = [1, 5, 5, len(labels_g)]
model_size = no_model_params
#model_params = model_params0.copy()
p_total = 100.
alpha = 0.1
nn_scale = .0001
model_scale = 1e1
key = random.PRNGKey(0)


key = random.PRNGKey(0)

load_nn = False
load_model = False

loader = lambda _ : [[tuple(np.array(_,np.float64) for _ in __) \
                for __ in k] for k in np.load(_,allow_pickle=True).tolist()]

if load_nn:
        try:
                nn_params_thin, nn_params_outlet = [[tuple(np.array(_,np.float64) for _ in __) \
                                for __ in k] for k in np.load('data_nn.npy',allow_pickle=True).tolist()]
                print('NN Load success!')
                failed = False
        except:
                failed = True
                print('Load failed. Generating new networks.')
else:
        load_nn = False
        

if load_model:
        try:
                model_params = np.float64(np.load('data_model.npy',allow_pickle=True).tolist()[0])
                print('MODEL load success!')
                failed = False
        except:
                failed = True
                print('Load failed. 1Generating new networks.')
else:
        load_model = False

if not load_nn:
        nn_params_thin         = init_network_params(nn_thin_size, key, nn_scale)
        nn_params_outlet     = init_network_params(nn_outlet_size,key,nn_scale)
        print('New networks generated.')        
if not load_model:
        model_params             = init_M_params(model_size, key, model_scale)
        print('New model inital guess generated.')        

# +
num_epochs = 300
num_eras = 10000
latent_variables = 3

frac = .99
step = 5
data = tuple([_[:int(len(data0[0])*frac)] for _ in data0])
samples = np.concatenate((np.arange(0,len(data[0]),step),np.array([len(data[0])-1])))
data = tuple([_[samples,:] for _ in data])
batch_size = int(len(data[0])*0.75)


# -

@jit
def model(model_params,x):
        #print('model x: {}'.format(x))
        x = np.nan_to_num(x)
        k = model_params
        return np.array([
                                        [-k[0]*x[0]*x[4]+k[1]*x[2]],
                                        [k[2]*x[2]*x[3]],
                                        [k[0]*x[0]*x[4]-k[1]*x[2]-k[2]*x[2]*x[3]],
                                        [-k[2]*x[2]*x[3]],
                                        [-k[0]*x[0]*x[4]+k[1]*x[2]+2*k[2]*x[2]*x[3]]
                                        ])
        
        """
        return np.array([[-mp[0]*(x[0])],0
                                         [mp[0]*(x[0]) - mp[1]*((x[1]))],
                                         [mp[1]*((x[1]))]])
        """
batched_M = vmap(model,in_axes=(None,0))#, in_axes=(None, 0)


@jit
def get_errors(nn_params_thin, nn_params_outlet, model_params, batch):
    # need to check come up with outlet integral
    x, x_out, t     = batch
    eff = 1e-4
    eff_model = eff*10.
    pred_x_ = batched_state(nn_params_thin, t)
    pred_x_out_ = batched_state(nn_params_outlet, t)
    err_data    =    (x-pred_x_[:,:-latent_variables])**2
    err_data_int    =    (x_out-diff_state(nn_params_outlet,t))**2
    pred_x = np.exp(pred_x_)-1.
    pred_x_out = np.exp(pred_x_out_)-1. 
    diff_M = batched_M(model_params,pred_x)[:,:,0]
    diff_nn = diff_state(nn_params_thin,t)
    err_M_g    = eff_model*((diff_nn[:,:-latent_variables]-diff_M[:,:-latent_variables]))**2
    err_M_s    = eff_model*((diff_nn[:,latent_variables-1:])-(diff_M[:,latent_variables-1:]))**2
    err_B        =    (pred_x[:,latent_variables-1:].sum(axis=1)-1.)**2
    err_MB         =    eff*((pulse_n+(scale_thin*tot_sites*pred_x[0,latent_variables-1:]*mw_s).sum(axis=1)\
                                -(scale_thin*tot_sites*pred_x[1,latent_variables-1:]*mw_s).sum(axis=1)\
                            +(scale_out*pred_x_out[1,:]*mw_g).sum(axis=1))**2).sum()\
                            +eff*batched_state(nn_params_outlet, np.array([[0.]]))**2
                            #+((pred_x[0,latent_variables-1:]-bc_s)**2).sum()
    barrier = 1e-3
    return err_data, err_data_int, err_M_g, err_M_s, err_MB, err_B\
                                 +barrier**2*np.exp((1-model_params[0]/barrier)).mean()\
                                 +barrier**2*np.exp((1.-pred_x/barrier)).mean()
                                #+barrier*np.maximum(0.,-model_params).mean()\
                                #+barrier*np.maximum(0.,-pred_x).mean()\
                                #+barrier*np.maximum(0.,-pred_x_out).mean()\
                                #+barrier*np.log(model_params**2).mean()\
                                #+barrier*np.log(pred_x**2).mean()\
                                #+barrier*np.log(pred_x_out**2).mean()


@jit
def loss(nn_params_thin, nn_params_outlet, model_params, batch):
    #print('pred[0]: {}'.format(pred_x[0]))
    #print('sum pred[0] - bc: {}'.format(sum((pred_x[0]-bc)**2))) 
    e = get_errors(nn_params_thin, nn_params_outlet, model_params, batch)
    return np.array([_.sum() for _ in e]).sum()/(1-np.array([_.var() for _ in e]).var())**2


# +
step = 1e-3
b1_ = 0.9
b2_ = 0.999
eps_ = 1e-100
opt_init_thin, opt_update_thin, get_params_thin = optimizers.adam(step, b1=b1_, b2=b2_, eps=eps_)#, b1=0.1, b2=0.999, eps=1e-10)
opt_init_outlet, opt_update_outlet, get_params_outlet = optimizers.adam(step, b1=b1_, b2=b2_, eps=eps_)
opt_init_M, opt_update_M, get_params_M = optimizers.adam(step,b1=b1_, b2=b2_, eps=eps_)
    
@jit
def step(i, opt_state_thin, opt_state_outlet, opt_state_M, batch):
    nn_params_thin = get_params_thin(opt_state_thin)
    nn_params_outlet = get_params_outlet(opt_state_outlet)
    model_params = get_params_M(opt_state_M)
    grads_thin = grad(loss,argnums=0)(nn_params_thin, nn_params_outlet, model_params, batch)
    grads_outlet = grad(loss,argnums=1)(nn_params_thin, nn_params_outlet, model_params, batch)
    grads_M = grad(loss,argnums=2)(nn_params_thin, nn_params_outlet, model_params, batch)
    return [opt_update_thin(i, grads_thin, opt_state_thin),\
                    opt_update_outlet(i, grads_outlet, opt_state_outlet),
                    opt_update_M(i, grads_M, opt_state_M)]

opt_state_thin = opt_init_thin(nn_params_thin)
opt_state_outlet = opt_init_outlet(nn_params_outlet)
opt_state_M = opt_init_M(model_params)

# +
# #%prun

itercount    = itertools.count()        

for j in range(num_eras):
        #err, _, _, _,_ = get_errors(nn_params_thin, nn_params_outlet, model_params, data)
        #err_dist = err.mean(axis=1)#+err_L.mean(axbis=1)#+err_B.mean(axis=1)#+err_M.mean(axis=1)
        #err_dist = err_dist/err_dist.sum()     
        #sel = np.concatenate((np.array([0,len(data[0])-1]),choice(np.arange(len(data[0])),batch_size,False,err_dist.flatten())))
        sel = random.shuffle(key,np.arange(len(data[0])))
        sel = np.concatenate((np.array([0,len(data[0])-1]),sel))
        batch = tuple([_[sel[:batch_size],:] for _ in data])
        #batch = data
        for i in range(int(num_epochs)):
            it = next(itercount)
            opt_state_thin, opt_state_outlet, opt_state_M = step(it, opt_state_thin, opt_state_outlet, opt_state_M, batch)
            nn_params_thin = get_params_thin(opt_state_thin)
            nn_params_outlet = get_params_outlet(opt_state_outlet)
            model_params = get_params_M(opt_state_M)
            loss_it_sample = loss(nn_params_thin, nn_params_outlet, model_params, batch)
            loss_it_data = loss(nn_params_thin, nn_params_outlet, model_params, data)
            #print('Iteration: {:4d}, Loss Batch: {:.7e}, Loss Data: {:.7e}'.format(i,loss_it_sample,loss_it_data))
            print('Iteration: {:4d}, Loss Batch: {:.7e}'.format(i,loss_it_sample))
            clear_output(wait=True)
            #print('--.--')                                
nn_params = get_params_nn(opt_state_nn)
model_params = get_params_M(opt_state_M)

# +
#print((batched_state(nn_params,model_params,x)).sum(axis=1))
#print((sol.y).sum(axis=0))
#def ode2(t,C):
#        return model(np.abs(model_params),C).flatten()

#sol2 = solve_ivp(ode2, (0, tmax), bc0, t_eval = t_eval)
x, x_out, t     = data

plt.figure(figsize=[10,10*2./3])
plt.plot(t, x)
plt.plot(t,(batched_state(nn_params_thin, t)[:,:-latent_variables]),'-o',lw=0.2,ms=2)
#plt.plot(sol2.t, sol2.y.T,'-.',lw=0.75,ms=2)
plt.legend(labels_g+labels_g)
plt.xlabel('Time')
plt.ylabel('C');

plt.figure(figsize=[10,10*2./3])
plt.plot(t,(batched_state(nn_params_thin, t)[:,latent_variables-1:]),'-o',lw=0.2,ms=2)
#plt.plot(sol2.t, sol2.y.T,'-.',lw=0.75,ms=2)
plt.legend(labels_s)
plt.xlabel('Time')
plt.ylabel('theta');

plt.figure(figsize=[10,10*2./3])
plt.plot(t, x_out)
plt.plot(t,(diff_state(nn_params_outlet, t)),'-o',lw=0.2,ms=2)
#plt.plot(sol2.t, sol2.y.T,'-.',lw=0.75,ms=2)
plt.legend(labels_g+labels_g)
plt.xlabel('Time')
plt.ylabel('C');

display(model_params)
#display(dict(zip(model_params0,model_params)))
#plt.gca().set_xscale('log')
#plt.gca().set_yscale('log')

err_data, err_data_int, err_M_g, err_M_s, err_MB, err_B = get_errors(nn_params_thin, nn_params_outlet, model_params, batch)
print('ERROR\n')
print('Data: {:.3e}'.format(err_data.mean()))
print('Data Integral: {:.3e}'.format(err_data_int.mean()))
print('M_g: {:.3e}'.format(err_M_g.mean()))
print('M_s: {:.3e}'.format(err_M_s.mean()))
print('MB: {:.3e}'.format(err_MB.mean()))
print('B: {:.3e}'.format(err_B.mean()))
# -
np.save('data_nn.npy',[nn_params_thin,nn_params_outlet],allow_pickle=True)
np.save('data_model.npy',[model_params],allow_pickle=True)


