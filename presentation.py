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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# # Inverse Artificial Neural Network (ANN) ODE Solver
#
# ### Summer 2019 CX4240 - Project
#
# Dr. Mahdi Roozbahani
#
# #### [**Project Proposal Document**](./proposal/proposal.pdf)
#
# ### Group:
#
#    - Gabriel Sabenca Gusmao **[GSG]**
#    - Nicole Yuge Hu **[NYH]**
#    - Zhenzi Yu  **[ZY]**

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### 1. Introduction:
#
# **a. Background: Chemical Reactions:** each state variable and time are features that can be synthetically created.
#
# $$A\overset{k_1}{\to} B\overset{k_2}{\to} C\notag $$
#
# Set of coupled ordinary differential equations (ODEs):
# $$\frac{dC_A}{dt} = -k_{1} C_A$$
# $$\frac{dC_B}{dt} = k_1 C_A-k_2 C_B$$
# $$\frac{dC_C}{dt} = k_2 C_B$$   
#
# **b. Goal: solve Ordinary Differential Equations (ODEs).**
#   - Traditional solution: discrete, iterative, problem dependent.
#   - Neural Network: flexible, continuous, parallelizable.    
#   
# **c. Validation: Can a neural network learn (interpolate) the ODE numerical solution?**
#
# <img src="./invANNODE/basis/intro.gif" style="width: 700px;"/>
#
# **d. Forward problem:** get the solution only using k values (model parameters), the governing equation and boundary conditions.
#
# **f. Inverse Problem:** infer k values given data and governing equations.
#
#    - Model Selection and Parameter Fitting
#

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### 2. Model: Inverse ANN solution for ODE's for model selection and parameter regression.
#
# #### Project Diagram / Workflow
# <p></p>
# <div>
#   <img src="./imgs/fig3.png" alt="overview" style="width: 900px;"/> 
# </div>

# + {"slideshow": {"slide_type": "slide"}, "cell_type": "markdown"}
# ### 3. Result & Discussion
# **All ANN have a single hidden layer (8 to 10 neurons) and use hyperbolic tangent activation function (in exponential form).**
#
# #### 3.1 Forward problem
# NN ODE forward solution and derivatives of model and ANN
#
# <div>
#   <img src="./invANNODE/basis/fwd.gif" alt="overview" style="width: 700px;"/> 
# </div>
#
# Usign [ADAM optimizer](https://arxiv.org/abs/1412.6980v9) (arXiv:1412.6980 [cs.LG])
# - ANN can readily be applied to solve forward ODE problems by interpolating between data. 
# - ANN method has gradient converge to model after about 80 epochs of 100 iterations each. 
#
# #### 3.2 Inverse problem
# NN ODE inverse solution and derivatives of model and ANN
#
# <div>
#     <img src="./invANNODE/basis/non_noise.gif" alt="overview" style="width: 900px;"/> 
# </div>
#
# Model parameter comparison at epochs = 200
#
# | True $k_{1}$ |Regressed $ k_{1}$ | True $k_{2}$ |Regressed $ k_{2}$ |
# |--------------|-------------------|--------------|-------------------|
# | 1.00000      | 0.98567           |  1.00000     | 0.98181           |
#
#
# - NN can also be applied to solve inverse ODE problems and extract model parameters. 
#
# #### 3.3 Noisy data
# Added noise to data generated to model actual experimental errors.
#
# ANN ODE noisy data solution and derivatives of model and NN
#
# <div>
#     <img src="./invANNODE/basis/noisy.gif" alt="overview" style="width: 900px;"/> 
# </div>
#
# - NN model adapt well to data with significant noises.
# - One hidden layer and small number of nodes were used in ANN to avoid over-fitting.
#
#
# ### Discussion
#
# - NN has the advantages of being continuous, differentiable, closed-form solution in ANN meaning that it requires less memory, but at the same time, it requires more time to train on the dataset. 
# - NN can solve kinetics ODE in forward and inverse fashion. 
# - NN also works with noisy dataset that mimics actual experiment measurements. 
#
# ### Future work:
#
# - Compatibility of ANN with stiff problems
# - Optimize discretization of domain like ivp_solver. 
# -

# ### 4. Similar projects
#
# [**Neural Ordinary Differential Equations**](https://arxiv.org/abs/1806.07366v4) (https://arxiv.org/abs/1806.07366)   
#
# &ensp; [2] Chen, R. T. Q., Rubanova, Y., Bettencourt, J. & Duvenaud, D. Neural Ordinary Differential Equations. (2018).
#
#    - Extensive study on the forward problem automatic differentiation capability.
#    - Does not explore inverse problems.
#
# [**Hidden Physics Models: Machine Learning of Nonlinear Partial Differential Equations**](https://arxiv.org/abs/1708.00588) (https://arxiv.org/abs/1708.00588)   
#
# &ensp; [3] Raissi, M. & Karniadakis, G. E. Hidden Physics Models: Machine Learning of Nonlinear Partial Differential Equations. (2017).
#    
#    - Non-linear ordinary and partial differential equation identification.
#    - Uses Gaussian Processes as scaffold for learning.
#    - Assumes model is known apriori.
#    
# [**Solving coupled ODEs with a neural network and autograd**](http://kitchingroup.cheme.cmu.edu/blog/category/ode/ ) (Kitchin's Group)
#
#    - Solves the forward ODE for kinetic systems using ANN.
#    - Show example of automatic differentiation with *autograd*.
#    - Kitchin's work on the forward problem will be used as starting point for the inverse problem.
#    
# [**Multistep Neural Networks for Data-driven Discovery of Nonlinear Dynamical Systems**](https://arxiv.org/abs/1801.01236) (https://arxiv.org/abs/1801.01236)   
#
#
# &ensp; [4] Raissi, M., Perdikaris, P. & Karniadakis, G. E. Multistep Neural Networks for Data-driven Discovery of Nonlinear Dynamical Systems. (2018).
#
#    - Solves multiple kinds of PDE's using ANN.
#    - Explore different time-stepping formulas.
#    - Discusses whole of noise and regularization
#    - Does not entail parameter fitting.
#    

# ### 6. Library Dependencies
#
#
# - Automatic differentiation: [**JAX**](https://github.com/google/jax) (https://github.com/google/jax)
# - Array-operation: [**numpy**](https://www.numpy.org/)
# - Solution for initial-value problems, IVPs (forward problem): [**dasslc2py**](https://github.com/asanet/dasslc2py) (https://github.com/asanet/dasslc2py) or [**scipy**](https://www.scipy.org/)
# - <s>Non-linear minimization (optimization): [**scipy**](https://www.scipy.org/) ... for now.]</s> [**JAX**]
# - <s>Artificial Neural Networks (ANN): [**TensorFlow**](https://www.tensorflow.org/) (https://www.tensorflow.org/)</s> [**JAX**]
