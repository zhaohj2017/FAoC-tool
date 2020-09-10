import torch
import superp
import math

############################################
# set default data type to double
############################################
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float32)
# torch.set_default_tensor_type(torch.FloatTensor)

############################################
# set the super-rectangle range
############################################
# set the initial in super-rectangle
INIT = [[-20 / 180 * math.pi, 20 / 180 * math.pi], \
            [- 20 / 180 * math.pi, 20 / 180 * math.pi]
        ]
INIT_SHAPE = 1 # 2 for circle, 1 for rectangle

SUB_INIT = []
SUB_INIT_SHAPE = []

# the the unsafe in super-rectangle
UNSAFE = [[-90 / 180 * math.pi, 90 / 180 * math.pi], \
            [-90 / 180 * math.pi, 90 / 180 * math.pi]
        ]
UNSAFE_SHAPE = 1 # 2 for circle, 1 for rectangle

SUB_UNSAFE = [ [[-90 / 180 * math.pi, -30 / 180 * math.pi], [-90 / 180 * math.pi, 90 / 180 * math.pi]], \
                    [[30 / 180 * math.pi, 90 / 180 * math.pi], [-90 / 180 * math.pi, 90 / 180 * math.pi]], \
                        [[-30 / 180 * math.pi, 30 / 180 * math.pi], [-90 / 180 * math.pi, -30 / 180 * math.pi]], \
                            [[-30 / 180 * math.pi, 30 / 180 * math.pi], [30 / 180 * math.pi, 90 / 180 * math.pi]]
]
SUB_UNSAFE_SHAPE = [1, 1, 1, 1]

# the the domain in super-rectangle
DOMAIN = [[-90 / 180 * math.pi, 90 / 180 * math.pi], \
            [-90 / 180 * math.pi, 90 / 180 * math.pi]
        ]
DOMAIN_SHAPE = 1 # 1 for rectangle

ASYMP = [0.0, 0.0]


############################################
# set the range constraints
############################################
# accept a two-dimensional tensor and return a 
# tensor of bool with the same number of columns
def cons_init(x): 
    return x[:, 0] == x[:, 0] # equivalent to True

def cons_unsafe(x):
    unsafe1 = (x[:, 0] >= -90 / 180 * math.pi - superp.TOL_DATA_GEN) & (x[:, 0] <= -30 / 180 * math.pi + superp.TOL_DATA_GEN) & (x[:, 1] >= -90 / 180 * math.pi - superp.TOL_DATA_GEN) \
        & (x[:, 1] <= 90 / 180 * math.pi + superp.TOL_DATA_GEN)
    unsafe2 = (x[:, 0] >= 30 / 180 * math.pi - superp.TOL_DATA_GEN) & (x[:, 0] <= 90 / 180 * math.pi + superp.TOL_DATA_GEN) & (x[:, 1] >= -90 / 180 * math.pi - superp.TOL_DATA_GEN) \
        & (x[:, 1] <= 90 / 180 * math.pi + superp.TOL_DATA_GEN)
    unsafe3 = (x[:, 0] >= -30 / 180 * math.pi - superp.TOL_DATA_GEN) & (x[:, 0] <= 30 / 180 * math.pi + superp.TOL_DATA_GEN) & (x[:, 1] >= -90 / 180 * math.pi - superp.TOL_DATA_GEN) \
        & (x[:, 1] <= -30 / 180 * math.pi + superp.TOL_DATA_GEN)
    unsafe4 = (x[:, 0] >= -30 / 180 * math.pi - superp.TOL_DATA_GEN) & (x[:, 0] <= 30 / 180 * math.pi + superp.TOL_DATA_GEN) & (x[:, 1] >= 30 / 180 * math.pi - superp.TOL_DATA_GEN) \
        & (x[:, 1] <= 90 / 180 * math.pi + superp.TOL_DATA_GEN)
    return unsafe1 | unsafe2 | unsafe3 | unsafe4 # x[:, 0] stands for x1 and x[:, 1] stands for x2
 
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
            return 9.8 * (x[:, 0] - torch.pow(x[:, 0], 3) / 6.0) + (ctrl_nn(x))[:, 0]
        else:
            print("Vector function error!")
            exit()

    vf = torch.stack([f(i + 1, x) for i in range(superp.DIM_S)], dim=1)
    return vf