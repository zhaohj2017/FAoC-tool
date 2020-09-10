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
INIT_BOUND = 0.05

# state = (x, theta, dot_x, dot_theta)
INIT = [[-INIT_BOUND, INIT_BOUND], \
            [-INIT_BOUND, INIT_BOUND], \
                [-INIT_BOUND, INIT_BOUND], \
                    [-INIT_BOUND, INIT_BOUND]
        ]
INIT_SHAPE = 1 # 2 for circle, 1 for rectangle

SUB_INIT = []
SUB_INIT_SHAPE = []

# the the unsafe in super-rectangle
UNSAFE = [[-2, 2], \
            [-1, 1], \
                [-2, 2], \
                    [-2, 2]
        ]   # the bound for theta is approximately 60 degree
UNSAFE_SHAPE = 1 # 2 for circle, 1 for rectangle

SUB_UNSAFE = []
SUB_UNSAFE_SHAPE = []

# UNSAFE_BOUNDARY = [[-0.9, 0.9], \
#             [-0.4, 0.4], \
#                 [-0.9, 0.9], \
#                     [-0.9, 0.9]
#         ]

# the the domain in super-rectangle
DOMAIN = [[-2, 2], \
            [-1, 1], \
                [-2, 2], \
                    [-2, 2]
        ]
DOMAIN_SHAPE = 1 # 1 for rectangle

ASYMP = [0.0, 0.0, 0.0, 0.0]

############################################
# set the range constraints
############################################
# accept a two-dimensional tensor and return a 
# tensor of bool with the same number of columns
def cons_init(x): 
    return x[:, 0] == x[:, 0] # equivalent to True

def cons_unsafe(x):
    inner_box = (x[:, 0] > -2 + superp.TOL_DATA_GEN) & (x[:, 0] < 2 - superp.TOL_DATA_GEN) \
        & (x[:, 1] > -1 + superp.TOL_DATA_GEN) & (x[:, 1] < 1 - superp.TOL_DATA_GEN) \
        & (x[:, 2] > -2 + superp.TOL_DATA_GEN) & (x[:, 2] < 2 - superp.TOL_DATA_GEN) \
        & (x[:, 3] > -2 + superp.TOL_DATA_GEN) & (x[:, 3] < 2 - superp.TOL_DATA_GEN)
    return ~inner_box # x[:, 0] stands for x1 and x[:, 1] stands for x2

# def cons_unsafe_boundary(x):
#     inner_box = (x[:, 0] >= -0.9 + superp.TOL_DATA_GEN) & (x[:, 0] <= 0.9 - superp.TOL_DATA_GEN) \
#         & (x[:, 1] >= -0.4 + superp.TOL_DATA_GEN) & (x[:, 1] <= 0.4 - superp.TOL_DATA_GEN) \
#         & (x[:, 2] >= -0.9 + superp.TOL_DATA_GEN) & (x[:, 2] <= 0.9 - superp.TOL_DATA_GEN) \
#         & (x[:, 3] >= -0.9 + superp.TOL_DATA_GEN) & (x[:, 3] <= 0.9 - superp.TOL_DATA_GEN)
#     return ~inner_box # x[:, 0] stands for x1 and x[:, 1] stands for x2

def cons_domain(x):
    return x[:, 0] == x[:, 0] # equivalent to True

def cons_asymp(x):
    return torch.norm(x - torch.tensor(ASYMP), dim=1) >= superp.RADIUS_ASYMP

############################################
# set the vector field
############################################
# this function accepts a tensor input and returns the vector field of the same size
def vector_field(x, ctrl_nn):
    gravity = 9.8
    # m_cart = 1.0
    # m_pole = 0.1
    # m_total = m_cart + m_pole
    l_pole = 1.0

    normed_force = (ctrl_nn(x))[:, 0]

    # # state: (x, theta, dox_x, dot_theta)
    dd_x = normed_force
    dd_theta = (gravity * torch.sin(x[:, 1]) - normed_force * torch.cos(x[:, 1])) / l_pole

    # normed_force = (force + m * sin(theta) * (l * d_theta * d_theta - g * cos(theta))) \
    #                   / (M + m * sin(theta) * sin(theta))

    f = [x[:, 2], x[:, 3], dd_x, dd_theta] # (dot_x, dot_theta, dot_dot_x, dot_dot_theta)
    
    # reinforcement learning model
    # gravity = 9.8
    # m_cart = 1.0
    # m_pole = 0.1
    # m_total = m_cart + m_pole
    # l_pole = 0.5
    # ml_pole = m_pole * l_pole

    # force = (ctrl_nn(x))[:, 0]
    # temp = (force + ml_pole * x[:, 3] * x[:, 3] * torch.sin(x[:, 1])) / m_total

    # dd_theta = (gravity * torch.sin(x[:, 1]) - torch.cos(x[:, 1]) * temp) / l_pole \
    #                 / (4.0 / 3.0 - m_pole * torch.cos(x[:, 1]) * torch.cos(x[:, 1]) / m_total)
    # dd_s = temp - ml_pole * dd_theta * torch.cos(x[:, 1]) / m_total

    # f = [x[:, 2], x[:, 3], dd_s, dd_theta] # (dot_x, dot_theta, dot_dot_x, dot_dot_theta)


    vf = torch.stack([f[i] for i in range(superp.DIM_S)], dim=1)
    return vf