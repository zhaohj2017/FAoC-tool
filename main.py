import torch
import torch.nn as nn
import superp
import ann
import data
import train
import time
import redlog
import plot # comment this line if matplotlib, mayavi, or PyQt5 was not successfully installed
import plot3d

# generating training model
barr_nn = ann.gen_nn(superp.N_H_B, superp.D_H_B, superp.DIM_S, 1, superp.BARR_ACT, superp.BARR_OUT_BOUND) # generate the nn model for the barrier
ctrl_nn = ann.gen_nn(superp.N_H_C, superp.D_H_C, superp.DIM_S, superp.DIM_C, superp.CTRL_ACT, superp.CTRL_OUT_BOUND) # generate the nn model for the controller

# loading pre-trained model
if superp.FINE_TUNE == 1:
    barr_nn.load_state_dict(torch.load('pre_trained_barr.pt'), strict=True)
    ctrl_nn.load_state_dict(torch.load('pre_trained_ctrl.pt'), strict=True)


###########################################
# # check counter example
###########################################
# counter_ex = [0.313456207410, -0.35799516543633, -0.059641790995704782]
# train.test_lie(barr_nn, ctrl_nn, counter_ex)
# plot3d.plot_sys_3d(barr_nn, ctrl_nn, counter_ex) 


# generate training data
time_start_data = time.time()
batches_init, batches_unsafe, batches_domain, batches_asymp = data.gen_batch_data()
time_end_data = time.time()

############################################
# number of mini_batches
############################################
BATCHES_I = len(batches_init)
BATCHES_U = len(batches_unsafe)
BATCHES_D = len(batches_domain)
BATCHES_A = len(batches_asymp)
BATCHES = max(BATCHES_I, BATCHES_U, BATCHES_D, BATCHES_A)
NUM_BATCHES = [BATCHES_I, BATCHES_U, BATCHES_D, BATCHES_A, BATCHES]

# train and return the learned model
time_start_train = time.time()
res = train.itr_train(barr_nn, ctrl_nn, batches_init, batches_unsafe, batches_domain, batches_asymp, NUM_BATCHES) 
time_end_train = time.time()

print("\nData generation totally costs:", time_end_data - time_start_data)
print("Training totally costs:", time_end_train - time_start_train)
print("-------------------------------------------------------------------------")

if res == True:
    # save model for fine tuning
    torch.save(barr_nn.state_dict(), 'pre_trained_barr.pt')
    torch.save(ctrl_nn.state_dict(), 'pre_trained_ctrl.pt')
    # generate script for verification in redlog
    redlog.script_gen(barr_nn, ctrl_nn)
    # comment this line if matplotlib, mayavi, or PyQt5 was not successfully installed
    if superp.VISUAL == 1:
        plot.plot_sys(barr_nn, ctrl_nn) 
else:
    print("Synthesis failed!")