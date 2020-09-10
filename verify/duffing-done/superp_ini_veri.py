import torch
import torch.nn as nn
import acti
import numpy as np


############################################
# set default data type to double; for GPU
# training use float
############################################
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.FloatTensor)


VERBOSE = 1 # set to 1 to display epoch and batch losses in the training process
VISUAL = 1 # plot figure or not

FINE_TUNE = 1 # set to 1 for fine-tuning a pre-trained model
FIX_CTRL = 0
FIX_BARR = 0


############################################
# set the system dimension
############################################
DIM_S = 2 # dimension of system
DIM_C = 1 # dimension of controller input

############################################
# set the network architecture
############################################
N_H_B = 1 # the number of hidden layers for the barrier
D_H_B = 10 # the number of neurons of each hidden layer for the barrier

N_H_C = 1 # the number of hidden layers for the controller
D_H_C = 5 # the number of neurons of each hidden layer for the controller

############################################
# for activation function definition
############################################
BENT_DEG = 0.0001

BARR_ACT = acti.my_act(BENT_DEG)
CTRL_ACT = nn.ReLU()

BARR_OUT_BOUND = 1e16 # set the output bound of the barrier NN
CTRL_OUT_BOUND = 1e16 # set the output bound of the controller NN: for bounded controller


############################################
# set loss function definition
############################################
TOL_INIT = 0.0
TOL_SAFE = 0.0

TOL_LIE = 0.0
TOL_NORM_LIE = 0.0
TOL_BOUNDARY = 0.05 # initial boundary 0.01

WEIGHT_LIE = 1.0
WEIGHT_NORM_LIE = 0

HEIGHT_ASYMP = 0.1 # set the norm lower bound outside a neighborhood of the asymptotic stability point with radius 
RADIUS_ASYMP = 0.1 # set the radius of the neighborhood around the asymptotic stability point
ZERO_ASYMP = 0.01 # set the norm upper bound at the asymptotic stability point

WEIGHT_ASYMP_DOMAIN = 1
WEIGHT_ASYMP_POINT = 1

DECAY_LIE = 1 # decay of lie weight 0.1 works, 1 does not work
DECAY_INIT = 1
DECAY_UNSAFE = 1
DECAY_ASYMP = 0 # set the weight of the asymptotic stability loss


############################################
# number of training epochs
############################################
EPOCHS = 100

############################################
# my own scheduling policy: 
############################################
ALPHA = 0.01
BETA = 0.1
GAMMA = 5

############################################
# training termination flags
############################################
LOSS_OPT_FLAG = 1e-16
TOL_MAX_GRAD = 5
GRAD_CTRL_FACTOR = 1.4

############################################
# for training set generation
############################################
TOL_DATA_GEN = 1e-16

DATA_EXP_I = np.array([7, 7]) 
    # for sampling from initial; length = prob.DIM
DATA_LEN_I = np.power(2, DATA_EXP_I) 
    # the number of samples for each dimension of domain
BLOCK_EXP_I = np.array([5, 5]) 
    # 0 <= BATCH_EXP <= DATA_EXP
BLOCK_LEN_I = np.power(2, BLOCK_EXP_I) 
    # number of batches for each dimension

DATA_EXP_U = np.array([8, 8]) # for sampling from initial; length = prob.DIM
DATA_LEN_U = np.power(2, DATA_EXP_U) # the number of samples for each dimension of domain
BLOCK_EXP_U = np.array([6, 6]) # 0 <= BATCH_EXP <= DATA_EXP
BLOCK_LEN_U = np.power(2, BLOCK_EXP_U) # number of batches for each dimension

DATA_EXP_D = np.array([8, 8]) # for sampling from initial; length = prob.DIM
DATA_LEN_D = np.power(2, DATA_EXP_D) # the number of samples for each dimension of domain
BLOCK_EXP_D = np.array([6, 6]) # 0 <= BATCH_EXP <= DATA_EXP
BLOCK_LEN_D = np.power(2, BLOCK_EXP_D) # number of batches for each dimension


############################################
# for plotting
############################################
PLOT_EXP_B = np.array([8, 8]) # sampling from domain for plotting the boundary of barrier using contour plot
PLOT_LEN_B = np.power(2, PLOT_EXP_B) # the number of samples for each dimension of domain, usually larger than superp.DATA_LEN_D

PLOT_EXP_V = np.array([6, 6]) # sampling from domain for plotting the vector field
PLOT_LEN_V = np.power(2, PLOT_EXP_V) # the number of samples for each dimension of domain, usually equal to PLOT_LEN_P

PLOT_EXP_P = np.array([6, 6]) # sampling from domain for plotting the scattering sampling points, could be equal to PLOT_EXP_V
PLOT_LEN_P = np.power(2, PLOT_EXP_P) # the number of samples for each dimension of domain

PLOT_VEC_SCALE = None