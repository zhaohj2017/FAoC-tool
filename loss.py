import torch
import torch.nn as nn
import superp
import prob


############################################
# constraints for barrier certificate B:
# (1) init ==> B <= 0
# (2) unsafe ==> B > 0 <==> B >= eps <==> eps - B <= 0 (positive eps)
# (3) domain ==> lie <= 0  (lie + lambda * barrier <= 0, where lambda >= 0)
# (4) domain /\ B = 0 ==> lie < 0 (alternatively)
############################################


############################################
# given the training data, compute the loss
############################################
def calc_loss(barr_nn, ctrl_nn, input_init, input_unsafe, input_domain, input_asymp, epoch, batch_index):
    # compute loss of init    
    output_init = barr_nn(input_init)
    loss_init = torch.relu(output_init + superp.TOL_INIT) #tolerance

    # compute loss of unsafe
    output_unsafe = barr_nn(input_unsafe)
    loss_unsafe = torch.relu((- output_unsafe) + superp.TOL_SAFE) #tolerance

    ##--------------------------------------------------------------------------------------------------------------
    # select boundary points
    with torch.no_grad():
        output_domain = barr_nn(input_domain)
        boundary_index = ((output_domain[:,0] >= -superp.TOL_BOUNDARY) & (output_domain[:,0] <= superp.TOL_BOUNDARY)).nonzero()
        input_boundary = torch.index_select(input_domain, 0, boundary_index[:, 0])

    if len(input_boundary) > 0 and superp.DECAY_LIE > 0:
        # compute the gradient of nn on boundary
        input_boundary.requires_grad = True # temporarily enable gradient
        output_boundary = barr_nn(input_boundary)
        gradient_boundary = torch.autograd.grad(
                torch.sum(output_boundary),
                input_boundary,
                grad_outputs=None,
                create_graph=True,
                only_inputs=True,
                allow_unused=True)[0]
        input_boundary.requires_grad = False # temporarily disable gradient

        # compute the maximum gradient norm on boundary
        with torch.no_grad():   
            norm_gradient_boundary = torch.norm(gradient_boundary, dim=1)
            max_gradient = (torch.max(norm_gradient_boundary)).item() # computing max norm of gradient for controlling boundary sampling

        # compute the vector field on boundary
        vector_boundary = prob.vector_field(input_boundary, ctrl_nn) # with torch.no_grad():
        # compute the lie derivative on boundary
        lie_boundary = torch.sum(gradient_boundary * vector_boundary, dim=1) # sum the columns of lie_domain

        # compute the normalized lie derivative on the boundary   
        if superp.WEIGHT_NORM_LIE > 0:
            # nn.functional.normalize(*) # compute the normalized vector of the input vector in Pytorch
            # torch.nn.functional.normalize(input, p=2, dim=1, eps=1e-12, out=None)
            func_lie = nn.CosineSimilarity(eps=1e-16) # compute the cosine similarity of two vectors in Pytorch
            norm_lie_boundary = func_lie(gradient_boundary, vector_boundary)
        else:
            norm_lie_boundary = torch.tensor([0.0])

        # compute loss of lie
        loss_lie = superp.WEIGHT_LIE * torch.relu(lie_boundary + superp.TOL_LIE) + \
                        superp.WEIGHT_NORM_LIE * torch.relu(norm_lie_boundary + superp.TOL_NORM_LIE)
    
    else:
        loss_lie = torch.tensor([0.0])
        max_gradient = 0.0

    ##------------------------------------------------------------------------------------------------------------------------
    # calculate the cost of asymptotic stability
    if superp.DECAY_ASYMP > 0: # usually loss_asymp is not counted for pre-training
        vector_asymp = prob.vector_field(input_asymp, ctrl_nn) # with torch.no_grad():
        vector_asymp_norm = torch.norm(vector_asymp, dim=1)
        loss_asymp_domain = torch.relu(-vector_asymp_norm + superp.HEIGHT_ASYMP)
        
        ## loss_asymp_point = torch.relu(torch.norm(prob.vector_field(torch.tensor([prob.ASYMP]), ctrl_nn)) - superp.ZERO_ASYMP)
        loss_asymp_point = torch.relu(torch.norm(ctrl_nn(torch.tensor([prob.ASYMP]))) - superp.ZERO_ASYMP) # the loss of control input at the asymp point
        loss_asymp = superp.WEIGHT_ASYMP_DOMAIN * loss_asymp_domain + (batch_index <= 0) * superp.WEIGHT_ASYMP_POINT * loss_asymp_point

    else:
        loss_asymp = torch.tensor([0.0])

    ##-----------------------------------------------------------------------------------------------------
    # compute total loss
    total_loss = superp.DECAY_INIT * torch.sum(loss_init) + superp.DECAY_UNSAFE * torch.sum(loss_unsafe) \
                    + superp.DECAY_LIE * torch.sum(loss_lie) + superp.DECAY_ASYMP * torch.sum(loss_asymp) # the loss for asymptotic stability
                # torch.mean() for average

    # return total_loss is a tensor, max_gradient is a scalar
    return total_loss, max_gradient 
