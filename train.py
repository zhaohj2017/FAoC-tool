import torch
import torch.nn as nn
import numpy as np
import superp # parameters
import ann # generating model
import data # generating data
import loss # computing loss
import opt
import lrate
import prob


#################################################
# iterative training: the most important function
# it relies on three assistant functions:
#################################################


# used to output learned model parameters
def print_nn(model):
    for p in model.parameters():
        print(p.data)

def print_nn_matlab(model):
    layer = 0
    for p in model.parameters():
        layer = layer + 1
        arr = p.detach().numpy()
        if arr.ndim == 2:
            print( "w" + str((layer + 1) // 2) + " = [", end="")
            print('; '.join([', '.join(str(curr_int) for curr_int in curr_arr) for curr_arr in arr]), end="];\n")
        elif arr.ndim == 1:
            print( "b" + str(layer // 2) + " = [", end="")
            if layer == 2:
                print(', '.join(str(i) for i in arr), end="]';\n")
            else:
                print(', '.join(str(i) for i in arr), end="];\n")
        else:
            print("Transform error!")

# used for initialization and restart
def initialize_nn(model, stored_name, num_batches, fixed):    
    print("Initialize nn parameters!")

    ## random initialize or load saved
    if superp.FINE_TUNE == 0:
        for p in model.parameters():
            nn.init.normal_(p) #standard Gaussian distribution
    else:
        model.load_state_dict(torch.load(stored_name), strict=True)

    ## fix parameters
    if fixed == True:
        for p in model.parameters():
            p.requires_grad = False

    optimizer = opt.set_optimizer(model)
    scheduler = lrate.set_scheduler(optimizer, num_batches)

    return optimizer, scheduler

# to prevent generating a nn with large gradient, it works only for nn model with a single hidden layer
# do we need scale_ctrl? regularization？
def scale_nn(model, scale_factor): 
    with torch.no_grad():
        print("Scale nn parameters!")
        i = 0
        for p in model.parameters(): # i = 1, 3, 5, 7: weight matrix; i = 2, 4, 6, 8: bias
            i = i + 1
            if i % 2 == 0:
                p.data = p.data / torch.pow(torch.tensor(scale_factor), i // 2)
            else:
                p.data = p.data / scale_factor

def itr_train(barr_nn, ctrl_nn, batches_init, batches_unsafe, batches_domain, batches_asymp, NUM_BATCHES):
    # set the number of restart times
    num_restart = -1

    ############################## the main training loop ##################################################################
    while num_restart < 4:
        num_restart += 1
        
        # initialize nn models and optimizers and schedulers
        optimizer_barr, scheduler_barr = initialize_nn(barr_nn, "pre_trained_barr.pt", NUM_BATCHES[4], superp.FIX_BARR)
        optimizer_ctrl, scheduler_ctrl = initialize_nn(ctrl_nn, "pre_trained_ctrl.pt", NUM_BATCHES[4], superp.FIX_CTRL)

        # check counter example by isat3
        # loss.test_lie(barr_nn, ctrl_nn, [-0.316, 0.0477, 0.0367])

        init_list = np.arange(NUM_BATCHES[4]) % NUM_BATCHES[0]  # generate batch indices    # I
        unsafe_list = np.arange(NUM_BATCHES[4]) % NUM_BATCHES[1]                            # U
        domain_list = np.arange(NUM_BATCHES[4]) % NUM_BATCHES[2]                            # D
        asymp_list = np.arange(NUM_BATCHES[4]) % NUM_BATCHES[3]                             # A

        for epoch in range(superp.EPOCHS): # train for a number of epochs
            # initialize epoch
            epoch_loss = 0 # scalar
            epoch_gradient_flag = True # gradient is within range
            superp.CURR_MAX_GRAD = 0

            # mini-batches shuffle by shuffling batch indices
            np.random.shuffle(init_list) 
            np.random.shuffle(unsafe_list)
            np.random.shuffle(domain_list)
            np.random.shuffle(asymp_list)

            # train mini-batches
            for batch_index in range(NUM_BATCHES[4]):
                # batch data selection
                batch_init = batches_init[init_list[batch_index]]
                batch_unsafe = batches_unsafe[unsafe_list[batch_index]]
                batch_domain = batches_domain[domain_list[batch_index]]
                batch_asymp = batches_asymp[asymp_list[batch_index]]

                ############################## mini-batch training ################################################
                optimizer_barr.zero_grad() # clear gradient of parameters
                optimizer_ctrl.zero_grad()
                curr_batch_loss, curr_batch_gradient = loss.calc_loss(barr_nn, ctrl_nn, batch_init, batch_unsafe, batch_domain, batch_asymp, epoch, batch_index) 
                    # batch_loss is a tensor, batch_gradient is a scalar
                if curr_batch_loss.item() > 0:
                    curr_batch_loss.backward() # compute gradient using backward()
                    # update weight and bias
                    optimizer_barr.step() # gradient descent once
                    optimizer_ctrl.step()

                # learning rate scheduling for each mini batch
                scheduler_barr.step() # re-schedule learning rate once
                scheduler_ctrl.step()


                # update epoch loss
                epoch_loss += curr_batch_loss.item()
                # update epoch gradient flag
                curr_batch_gradient_flag = curr_batch_gradient < superp.TOL_MAX_GRAD
                epoch_gradient_flag = epoch_gradient_flag and curr_batch_gradient_flag

                if curr_batch_gradient > superp.CURR_MAX_GRAD:
                    superp.CURR_MAX_GRAD = curr_batch_gradient

                if superp.VERBOSE == 1:
                    print("restart: %-2s" % num_restart, "epoch: %-3s" % epoch, "batch: %-5s" % batch_index, "batch_loss: %-25s" % curr_batch_loss.item(), \
                        "boundary_gradient: %-25s" % superp.CURR_MAX_GRAD, "epoch_loss: %-25s" % epoch_loss)

                # gradient control by scale parameters
                if not curr_batch_gradient_flag:
                    scale_nn(barr_nn, superp.GRAD_CTRL_FACTOR)

            if (epoch_loss < superp.LOSS_OPT_FLAG) and epoch_gradient_flag:
                print("The last epoch:", epoch, "of restart:", num_restart)
                if superp.VERBOSE == 1:
                    print("\nSuccess! The nn barrier is:")
                    print_nn_matlab(barr_nn) # output the learned model
                    print("\nThe nn controller is:")
                    print_nn_matlab(ctrl_nn)

                return True # epoch success: end of epoch training

    return False

#------------------------------------------------------------------------------------------------------------------------

## test counter example
def test_lie(barr_nn, ctrl_nn, test_point):
    test_point = torch.tensor([test_point])
    test_point.requires_grad = True
    test_output = barr_nn(test_point)
    gradient_test = torch.autograd.grad(
            torch.sum(test_output),
            test_point,
            grad_outputs=None,
            create_graph=True,
            only_inputs=True,
            allow_unused=True)[0]
    test_point.requires_grad = False
    vector_test = prob.vector_field(test_point, ctrl_nn)
    test_lie = torch.sum(gradient_test * vector_test, dim=1) # sum the columns of lie_domain
    norm_vector_test = torch.norm(vector_test, dim=1)
    norm_gradient_test = torch.norm(gradient_test, dim=1)
    norm_lie_test = test_lie / norm_vector_test / norm_gradient_test
    ctrl = ctrl_nn(test_point)
    print("The nn output:", test_output)
    print("The gradient:", gradient_test)
    print("The vector field:", vector_test)
    print("The lie derivative:", test_lie)
    print("The normed lie derivate:", norm_lie_test)
    print("The control:", ctrl)
    # print("The barr nn:")
    # print_nn(barr_nn)
    # print("The ctrl nn:")
    # print_nn(ctrl_nn)
    # weight_list_barr_nn = [p.data for p in barr_nn.parameters()]
    # barr_hidden_input = torch.mm(weight_list_barr_nn[0], test_point.t()) + weight_list_barr_nn[1].reshape(-1,1)
    # barr_hidden_output = superp.BARR_ACT(barr_hidden_input)
    # print("barr nn hidden input:", barr_hidden_input)
    # print("barr nn hidden output:", barr_hidden_output)
    # print("barr nn deri hidden output:", 0.25 * barr_hidden_input / (barr_hidden_output - 0.5 * barr_hidden_input) + 0.5)
    input()