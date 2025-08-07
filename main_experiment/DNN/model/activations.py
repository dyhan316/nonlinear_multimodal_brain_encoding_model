#for taking in activation list str and returning a list of activation functions
import torch.nn as nn

#TODO : create custom acitvation functions if needed and add to "ACTIVATIONS" dict


#dict of all activations 
ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
    "softplus": nn.Softplus,
    "softsign": nn.Softsign,
    "elu": nn.ELU,
    "none": None,
    "softmax": nn.Softmax,
    "log_softmax": nn.LogSoftmax,
    "hardshrink": nn.Hardshrink,
    "hardtanh": nn.Hardtanh,
    "softshrink": nn.Softshrink,
    "prelu": nn.PReLU,
    "rrelu": nn.RReLU,
    "celu": nn.CELU,
    "glu": nn.GLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "identity": nn.Identity,
}



