import torch
import superp
import math



############################################
# set the super-rectangle range
############################################
# set the initial in super-rectangle
INIT = [[-math.pi / 30, math.pi / 30], \
            [-math.pi / 30, math.pi / 30], \
                [-math.pi / 30, math.pi / 30]
        ]
INIT_SHAPE = 1 # 2 for circle, 1 for rectangle

SUB_INIT = []
SUB_INIT_SHAPE = []

# the the unsafe in super-rectangle
UNSAFE = [[-math.pi / 2.5, math.pi / 2.5], \
            [-math.pi / 2.5 , math.pi / 2.5], \
                [-math.pi / 2.5, math.pi / 2.5]
        ]
UNSAFE_SHAPE = 1 # 2 for circle, 1 for rectangle

SUB_UNSAFE = [ [[-math.pi / 2.5, math.pi / 2.5], [-math.pi / 2.5, math.pi / 2.5], [-math.pi / 2.5, math.pi / 2.5]], \
    [[-math.pi / 3, math.pi / 3], [-math.pi / 3, math.pi / 3], [-math.pi / 3, math.pi / 3]] ]
SUB_UNSAFE_SHAPE = [1, 1]

# the the domain in super-rectangle
DOMAIN = [[-math.pi / 2.5, math.pi / 2.5], \
            [-math.pi / 2.5 , math.pi / 2.5], \
                [-math.pi / 2.5, math.pi / 2.5]
        ]
DOMAIN_SHAPE = 1 # 1 for rectangle

ASYMP = [0.0, 0.0, 0.0]

############################################
# set the range constraints
############################################
# accept a two-dimensional tensor and return a 
# tensor of bool with the same number of columns
def cons_init(x): 
    return x[:, 0] == x[:, 0] # equivalent to True

def cons_unsafe(x):
    inner_box = (x[:, 0] > -math.pi / 3 + superp.TOL_DATA_GEN) & (x[:, 0] < math.pi / 3 - superp.TOL_DATA_GEN) \
        & (x[:, 1] > -math.pi / 3 + superp.TOL_DATA_GEN) & (x[:, 1] < math.pi / 3 - superp.TOL_DATA_GEN) \
        & (x[:, 2] > -math.pi / 3 + superp.TOL_DATA_GEN) & (x[:, 2] < math.pi / 3 - superp.TOL_DATA_GEN)   
    return ~inner_box # x[:, 0] stands for x1 and x[:, 1] stands for x2
 
def cons_domain(x):
    return x[:, 0] == x[:, 0] # equivalent to True

def cons_asymp(x):
    return torch.norm(x - torch.tensor(ASYMP), dim=1) >= superp.RADIUS_ASYMP


############################################
# set the vector field
############################################
# this function accepts a tensor input and returns the vector field of the same size
def vector_field(x, ctrl_nn):
    # the vector of functions
    def f(i, x):
        if i == 1:
            return x[:, 1] # x[:, 1] stands for x2
        elif i == 2:
            return 30 * (torch.sin(x[:, 0]) + 0.5 * (ctrl_nn(x))[:, 0] * torch.cos(x[:, 0]))
            # 30 * (sin(x1) + 0.5*cos(x1)*u)
        elif i == 3:
            return torch.cos(x[:, 2]) * ((ctrl_nn(x))[:, 0] * torch.cos(x[:, 2]) - 20.0 * torch.sin(x[:, 2]))
            # u = u'*cos(x3)*cos(x3) - 20*cos(x3)*sin(x3)
        else:
            print("Vector function error!")
            exit()

    vf = torch.stack([f(i + 1, x) for i in range(superp.DIM_S)], dim=1)
    return vf