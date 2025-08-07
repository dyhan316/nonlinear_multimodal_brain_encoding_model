#MLP for classifiying the MNIST dataset, model part (given the width and depth, makes a generic DNN model)

#currently only implementd for LINEAR, but can be extended to CNN, RNN, etc.
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import ACTIVATIONS
from . import custom_nn

##overal structure : took hints from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
__all__ = ["MLP"]


class BasicBlock(nn.Module):
    def __init__(self, input_size, output_size, bn=False, activation="relu", dropRate=0.0, **kwargs):
        """
        BasicBlock class represents a basic building block of a neural network.
        
        Args:
            input_size (int): The number of input features.
            output_size (int): The number of output features.
            bn (bool, optional): Whether to apply batch normalization. Defaults to False.
            activation (str, optional): The activation function to use. Defaults to "relu".
            dropRate (float, optional): The dropout rate. Defaults to 0.0.
        """
        super(BasicBlock, self).__init__()
        self.activation = ACTIVATIONS[activation]()
        self.Linear = nn.Linear(input_size, output_size)
        if bn == "batchnorm" or bn == "True":
            self.bn = nn.BatchNorm1d(output_size)
        elif bn == "layernorm":
            self.bn = nn.LayerNorm(output_size)
        elif bn == "False":
            self.bn = None
        # self.bn = nn.BatchNorm1d(output_size) if bn else None
        self.droprate = dropRate

    def forward(self, x):
        out = self.Linear(x)
        if self.bn is not None:
            out = self.bn(out)
        out = self.activation(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return out
        # return torch.cat([x, out], 1) # since not doing residual connection, just return out
        
        
class BBasicBlock(nn.Module):
    r"""
    BBasicBlock class represents a basic building block of a bilinear neural network.
    Block diagram of a BBasicBlock is as follows:
    x -> Bilinear -> BN -> activation -> dropout -> out
    (Bilinear part could have Linear, could also be block diagonal Bilinear, ... )
    
    Args:
        input_size (int): The number of input features.
        output_size (int): The number of output features.
        bn (bool, optional): Whether to apply batch normalization. Defaults to False.
        activation (str, optional): The activation function to use. Defaults to "relu".
        dropRate (float, optional): The dropout rate. Defaults to 0.0.
        bilinear_type (str, optional): The type of bilinear layer to use. options are : "dend_Bilinear", "regular_Bilinear", "dend_Linear_Bilinear", "regular_Linear_Bilinear". Defaults to "regular_Bilinear".
    
    
    """
    def __init__(self, input_size, output_size, bn=False, activation="sigmoid", dropRate=0.0, 
                 bilinear_type = "regular_Bilinear", **kwargs):
                 #add_linear=True, block_diag=False, **kwargs):
        super(BBasicBlock, self).__init__()
        self.activation = ACTIVATIONS[activation]()
        #TODO CHANGING THIS PART!
        if bilinear_type == "dend_Bilinear":
            self.Bilinear = custom_nn.dend_Bilinear(input_size, output_size, **kwargs)
        elif bilinear_type == "regular_Bilinear":
            self.Bilinear = custom_nn.regular_Bilinear(input_size, output_size, **kwargs)
        elif bilinear_type == "dend_Linear_Bilinear":
            self.Bilinear = custom_nn.dend_Linear_Bilinear(input_size, output_size, **kwargs)
        elif bilinear_type == "regular_Linear_Bilinear":
            self.Bilinear = custom_nn.regular_Linear_Bilinear(input_size, output_size, **kwargs)
        elif bilinear_type == "regular_Linear_Bilinear_sqrt":
            self.Bilinear = custom_nn.regular_Linear_Bilinear_sqrt(input_size, output_size, **kwargs)
        elif bilinear_type == "regular_Linear_Bilinear_block_activation":
            self.Bilinear = custom_nn.regular_Linear_Bilinear_block_activation(input_size, output_size, **kwargs)
        elif bilinear_type == "regular_Linear_Bilinear_sqrt_residual_1":
            self.Bilinear = custom_nn.regular_Linear_Bilinear_sqrt_residual_1(input_size, output_size, **kwargs)
        else:
            raise NotImplementedError(f"bilinear_type {bilinear_type} is not implemented yet")
        #self.Bilinear = custom_nn.self_Bilinear(input_size, input_size, output_size) if not block_diag else custom_nn.dend_Bilinear(input_size, output_size, **kwargs)
        #self.Linear = nn.Linear(input_size, output_size, bias=False) if add_linear else None
        # self.add_linear = add_linear
        # self.block_diag = block_diag
        #self.Bilinear = custom_nn.regular_Bilinear(input_size, output_size) if not block_diag else custom_nn.dend_Bilinear(input_size, output_size, **kwargs)
        #=====MODIFHY ABOVE=====#TODO (get "bilinear_type" as input or sth, then, set self.Bilinear accordingly (and remove self.Linear... linear_bilinear thing is already done in custom_nn level)) 
        #TODO modify the add_linear part too! (no need now!)
        
        if bn == "batchnorm" or bn == "True":
            self.bn = nn.BatchNorm1d(output_size)
        elif bn == "layernorm":
            self.bn = nn.LayerNorm(output_size)
        elif bn == "False":
            self.bn = None
        # self.bn = nn.BatchNorm1d(output_size) if bn else None
        self.droprate = dropRate
            
    def forward(self,x):
        #factor = 1 #! TODO : TEST HOW FACTOR CHANGES THINGS! #may need to scale initializations of weights? (n^2개를 더하니 등등... (make it so that all same value std or sth))
        #out = factor*self.Bilinear(x)
        #out += self.Linear(x) if self.add_linear else 0
        out = self.Bilinear(x)
        if self.bn is not None:
            out = self.bn(out)
        out = self.activation(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return out
        # return torch.cat([x, out], 1) # since not doing residual connection, just return out

class _MLP(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        input_size: int = 28,
        width: Union[int, List[int]] = 1,
        activation: Union[str, List[str]] = "relu",
        batchnorm: Union[bool, List[bool]] = False,
        dropout: Union[float, List[float]] = 0.0,
        modify_first_block: str = "None", #first_layer : None or Linear or Bilinear or Linear_Bilinear
        last_layer: str = "Linear", #last_layer : Linear or Bilinear or Linear_Bilinear
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.width = width if isinstance(width, list) else [width]
        self.depth = len(self.width)
        self.activation = activation if isinstance(activation, list) else [activation] * self.depth
        self.batchnorm = batchnorm if isinstance(batchnorm, list) else [batchnorm] * self.depth
        self.dropout = dropout if isinstance(dropout, list) else [dropout] * self.depth
        self.modify_first_block = modify_first_block
        self.last_layer = last_layer
        self.kwargs = kwargs
        
        assert len(self.width) == len(self.activation) == len(self.batchnorm) == len(self.dropout), "width, activation, batchnorm, dropout must be of the same length"
        
        #building model
        self.layers = nn.ModuleList()
        self.build_model()
        
    def get_layers(self, block_type):        
        for i, width_i in enumerate(self.width):
            if width_i == 0:
                continue 
            elif i == 0:
                self.layers.append(block_type(self.input_size, width_i, self.batchnorm[i], self.activation[i], self.dropout[i], **self.kwargs))
            else:
                self.layers.append(block_type(self.width[i-1], width_i, self.batchnorm[i], self.activation[i], self.dropout[i], **self.kwargs))
        
        ##adding last layer! (cls layer)
        last_func_to_add = nn.Linear if self.last_layer == "Linear" else custom_nn.regular_Bilinear if self.last_layer == "Bilinear" else custom_nn.regular_Linear_Bilinear if self.last_layer == "Linear_Bilinear"  else None
        assert last_func_to_add is not None, f"last_layer {self.last_layer} is not implemented yet"
        
        if self.width == [0] : 
            self.layers.append(last_func_to_add(self.input_size, self.num_classes))
        else : 
            self.layers.append(last_func_to_add(self.width[-1], self.num_classes))
        self.layers.append(nn.Softmax(dim=1))     
        
        ##modify first block if specified by argparse
        if self.modify_first_block == "None" :
            pass
        elif self.modify_first_block == "Linear" :
            self.layers[0] = BasicBlock(self.input_size, self.width[0], self.batchnorm[0], self.activation[0], self.dropout[0], **self.kwargs)
        else : 
            raise NotImplementedError(f"modify_first_block {self.modify_first_block} is not implemented yet")
                
    def build_model(self):
        raise ValueError("build_model method must be implemented in derived classes (ex : BMLP, MLP))")
        
    def forward(self, x):
        x = x.view(x.shape[0], -1) #flatten
        for layer in self.layers:
            x = layer(x)
        return x

# def get_nn_Module(layer_name):
#     if layer_name == "Linear" : 
#         return nn.Linear
#     elif layer_name == "Bilinear" :
#         return custom_nn.regular_Bilinear
#     elif layer_name == "Linear_Bilinear" :
#         return custom_nn.regular_Linear_Bilinear
#     else :
#         raise NotImplementedError(f"layer_name {layer_name} is not implemented yet")    

class MLP(_MLP):
    def build_model(self):
        self.get_layers(BasicBlock)

class dend_Bilinear_MLP(_MLP):
    def build_model(self):
        self.get_layers(partial(BBasicBlock, bilinear_type="dend_Bilinear"))
class regular_Bilinear_MLP(_MLP):
    def build_model(self):
        self.get_layers(partial(BBasicBlock, bilinear_type="regular_Bilinear"))
class dend_Linear_Bilinear_MLP(_MLP):
    def build_model(self):
        self.get_layers(partial(BBasicBlock, bilinear_type="dend_Linear_Bilinear"))
class regular_Linear_Bilinear_MLP(_MLP):
    def build_model(self):
        self.get_layers(partial(BBasicBlock, bilinear_type="regular_Linear_Bilinear"))
        
class regular_Linear_Bilinear_sqrt_MLP(_MLP):
    def build_model(self):
        self.get_layers(partial(BBasicBlock, bilinear_type="regular_Linear_Bilinear_sqrt"))
        
        
class regular_Linear_Bilinear_block_activation_MLP(_MLP):
    def build_model(self):
        self.get_layers(partial(BBasicBlock, bilinear_type="regular_Linear_Bilinear_block_activation", block_activation=torch.tanh))

class regular_Linear_Bilinear_sqrt_residual_1_MLP(_MLP):
    def build_model(self):
        self.get_layers(partial(BBasicBlock, bilinear_type="regular_Linear_Bilinear_sqrt_residual_1"))
        
#TODO : implement other activation functions or combining activation functions or parameterized activation functions (maxout, etc.)
#actually, make it s.t. it takes in a list of activation functions, and then it will apply it to each layer (each neuron-wise activation? possible?)

#TODO implement dropout prob masking (especially for BNN)
