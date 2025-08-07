import argparse
import datetime
import re
from torchinfo import summary
import torch 
from torch import nn
import sys
import os
import numpy as np 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import config
from utils_resp import get_PCA_mat

def weighted_mse_loss(input, target, weight):
        return ((weight * (input - target) ** 2)/weight.mean()).mean()

class PCA_weighted_MSE(nn.Module):
    def __init__(self, subject):
        super(PCA_weighted_MSE, self).__init__()
        self.subject = subject
        self.weight = self._load_weight()

    def forward(self, input, target):
        return weighted_mse_loss(input, target, self.weight)

    def _load_weight(self):
        weight = np.load(file=f"{config.R_SETTINGS['PCA_MODEL']}/{self.subject}_PCA_8192.npz",
                         allow_pickle=True)['explained_variance_ratio']
        weight = weight[:config.R_SETTINGS['PCA_N_COMP']]
        weight = torch.from_numpy(weight).type(torch.float32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return weight

# def PCA_weighted_MSE_loss(input, target, subject):
#     """ 
#     MSE loss but weighted by PCA weights
#     """
#     weight = np.load(file=f"{config.R_SETTINGS['PCA_MODEL']}/{subject}_PCA_8192.npz", 
#                       allow_pickle=True)['explained_variance_ratio']
#     weight = weight[:config.R_SETTINGS['PCA_N_COMP']]
#     weight = torch.from_numpy(weight).type(input.dtype).to(input.device)  # Convert numpy array to PyTorch tensor
    
#     return weighted_mse_loss(input, target, weight)

#! TODO : DO BELOW BUT FOR nn.MOdule class recontrsucttion thing 
class PCA_recon_after_model(nn.Module):
    def __init__(self, subject, net ):
        super(PCA_recon_after_model, self).__init__()
        self.subject = subject
        self.PCA_mat_reduced = torch.from_numpy(get_PCA_mat(subject)['pca_components']).type(torch.float32)[:config.E_SETTINGS['recon_n_comp'],:].to('cuda') # device fixed to cuda
        self.net = net

    def forward(self, input):
        net_output = self.net(input)
        recon_from_pca_embedding = torch.matmul(net_output, self.PCA_mat_reduced) # direct matrix multiplication for reconstruction 
        return recon_from_pca_embedding
    

"""
* weighted MSE => done
* PCA reconstructed MSE calculation needed! Should get from loss function itself... Don't change --resp setting for this!!
* Prepare for testing 
"""


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def str2_int_list(v):
    v_list = v.split(",")
    return [int(i) for i in v_list]
    
def str2_float_list(string):
    # Split the string by '/' to get individual values
    values = string.split(',')
    # Convert each value to float
    floats = [float(value) for value in values]
    return floats

def str2_str_list(string):
    # Split the string by '/' to get individual values
    values = string.split(',')
    # Convert each value to float
    return values

    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default="ours", type=str) # experiment name
    parser.add_argument('--device_ids', default="0", type=str) # gpu ids
    parser.add_argument('--seed', default=2023, type=int) # global seed
    parser.add_argument("--datetime", default=datetime, type=str) # datetime to log
    #parser.add_argument('--root', default='/path/to/project', type=str) # root directory

    # important 
    parser.add_argument('--task', default='MNIST', type=str) # task #MNIST, FMNIST, CIFAR10,
    parser.add_argument('--debug', default=False, type=bool) # debug mode
    
    
    ######HYPERHYPERPARAMETERS########
    #model options 
    parser.add_argument('--use_pretrain_weight', default=True, type=str2bool, help="Do not use pretrained model")
    parser.add_argument('--model', default='MLP', type=str)  #MLP, dend_Bilinear_MLP, ..., "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "wide_resnet50_2", "wide_resnet101_2"
    parser.add_argument('--batch_size', default=128, type=int) # batc
    parser.add_argument('--width', default="16,16", type=str2_int_list) 
    parser.add_argument('--modify_first_block', default="None", type=str) #None, Linear #TODO : rest : not implemented yet! (don't plan on doing)
    parser.add_argument('--last_layer', default="Linear", type=str) #Linear or Bilinear or Linear_Bilinear ##TODO put in more options (linear+bilinear and so on)
    parser.add_argument('--block_size', default=2, type=int) #TODO maybe make it possible to be taken in as list? (i.e. diff block size for each layer)
    parser.add_argument('--activation', default="sigmoid", type=str) # DBNN may diverge easily, so sigmoid might be better
    parser.add_argument('--batchnorm', default = "batchnorm", type=str) #True(batchnorm), batchnorm, layernorm, False
    # parser.add_argument('--batchnorm', default=True, type=str2bool) 

    #training options
    parser.add_argument('--optimizer', default='SGD', type=str) #SGD, Adam, RMSprop, Adagrad, Adadelta, AdamW, SparseAdam, Adamax, ASGD, LBFGS, Rprop
    parser.add_argument('--epochs', default=10, type=int) 
    
    ######HYPERPARAMETERS########
    parser.add_argument('--do_optuna', default=True, type=bool) # do optuna
    # training options (i.e. hyperparameters)
    parser.add_argument('--dropout', default="0.1", type=str2_float_list) 
    parser.add_argument('--lr', default='0.01', type=str2_float_list)
    parser.add_argument('--weight_decay', default='5e-4', type=str2_float_list)
    parser.add_argument('--momentum', default='0.9', type=str2_float_list)
    #TODO maybe put in beta? (for adam)
    
    
    
    return parser


#count number of parameters in a model
#from https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
def get_n_params(net):
    num_params = sum(p.numel() for p in net.parameters())
    return num_params #units of millions

#print mean ,std of weight, bias values of BiLinear and Linear layers
def print_mean_std_shape(net):
    for name, param in net.named_parameters():
        if "weight" in name or "bias" in name:
            print(f"{name} Mean: {param.data.mean().item():.2e}, Std: {param.data.std().item():.2e}")

def get_model_summary(net, input_size):
    """ 
    returns summary of model, with low verbosity and high verbosity, AND model parameter size
    """
    verbose_model_stats = summary(net, input_size=input_size, verbose = 2,
            col_names = ["input_size",
                    "output_size",
                    "num_params",
                    "params_percent",
                    "kernel_size",
                    "mult_adds"])
    simple_model_stats = summary(net, input_size=input_size, verbose = 1,col_names = ["output_size",
                    "num_params",
                    "params_percent",
                    "mult_adds"])
    model_param_count = get_n_params(net)
    return verbose_model_stats, simple_model_stats, model_param_count
    
    
    #summary(net, input_size=input_size, verbose = verbose, col_names = ["input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"])
    summary(net, input_size=input_size, verbose = verbose, col_names = ["output_size", "num_params", "params_percent", "mult_adds"])
    
def generate_study_name(params_dict):
    study_name = "|".join([f"{key}__{value}" for key, value in params_dict.items()])
    return study_name

def parse_study_name(study_name):
    params_dict = {}
    pairs = study_name.split("|")
    for pair in pairs:
        key, value = pair.split("__")
        params_dict[key] = value
    return params_dict