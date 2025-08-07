import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Callable, List, Optional, Type, Union, Dict

import torch.nn as nn

from .activations import ACTIVATIONS

class MEyeMLP(nn.Module):
    r""" 
    MLP from "Mind's Eye" paper.
    """
    def __init__(self, out_dim=257*768, in_dim=15724, clip_size=768, h=4096):
        super().__init__()
        # in_dim corresponds to the subject-specific
        # number of voxels in the "midsageneral" brain region.
        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h, bias=False),
            nn.LayerNorm(h),
            nn.GELU(inplace=True),
            nn.Dropout(0.5))
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h, bias=False),
                nn.LayerNorm(h),
                nn.GELU(inplace=True),
                nn.Dropout(0.15)
            ) for _ in range(4)])
        self.lin1 = nn.Linear(h, out_dim, bias=True)
        self.proj = nn.Sequential(
            nn.LayerNorm(clip_size),
            nn.GELU(inplace=True),
            nn.Linear(clip_size, 2048, bias=False),
            nn.LayerNorm(2048),
            nn.GELU(inplace=True),
            nn.Linear(2048, 2048, bias=False),
            nn.LayerNorm(2048),
            nn.GELU(inplace=True),
            nn.Linear(2048, clip_size, bias=True))
        self.clip_size = clip_size

    def forward(self, x):
        x = self.lin0(x)
        residual = x
        for res_block in range(len(self.mlp)):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        diffusion_prior_input = self.lin1(x).reshape(len(x), -1, self.clip_size)
        disjointed_clip_fmri = self.proj(diffusion_prior_input)
        return diffusion_prior_input, disjointed_clip_fmri

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
        self.activation = ACTIVATIONS[activation]() #TODO : modify later
        self.Linear = nn.Linear(input_size, output_size)
        if bn == "batchnorm" or bn == "True":
            self.bn = nn.BatchNorm1d(output_size)
        elif bn == "layernorm":
            self.bn = nn.LayerNorm(output_size)
        elif bn == "False" or bn == False:
            self.bn = None
        self.bn = nn.BatchNorm1d(output_size) if self.bn is not None else None
        self.dropout = nn.Dropout(p=dropRate) if dropRate > 0 else None
        self.droprate = dropRate

    def forward(self, x):
        out = self.Linear(x)
        if self.bn is not None:
            out = self.bn(out)
        out = self.activation(out)
        if self.droprate > 0:
            out = self.dropout(out)
        return out
        # return torch.cat([x, out], 1) # since not doing residual connection, just return out
        
class _MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        width: Union[int, List[int]] = 1,
        activation: Union[str, List[str]] = "relu",
        batchnorm: Union[bool, List[bool]] = False,
        dropout: Union[float, List[float]] = 0.0,
        # modify_first_block: str = "None", #first_layer : None or Linear or Bilinear or Linear_Bilinear
        # last_layer: str = "Linear", #last_layer : Linear or Bilinear or Linear_Bilinear
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.width = width if isinstance(width, list) else [width]
        self.depth = len(self.width)
        self.activation = activation if isinstance(activation, list) else [activation] * self.depth
        self.batchnorm = batchnorm if isinstance(batchnorm, list) else [batchnorm] * self.depth
        self.dropout = dropout if isinstance(dropout, list) else [dropout] * self.depth
        # self.modify_first_block = modify_first_block
        # self.last_layer = last_layer
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
                self.layers.append(block_type(self.input_size, width_i, self.batchnorm[i], self.activation[i], 0.0, **self.kwargs))
            else:
                self.layers.append(block_type(self.width[i-1], width_i, self.batchnorm[i], self.activation[i], self.dropout[i], **self.kwargs))
        
        ##adding last layer! (cls layer)
        last_func_to_add = nn.Linear 
        #TODO : CHANGE BELOW!!
        if self.width == [0] : 
            self.layers.append(last_func_to_add(self.input_size, self.output_size))
        else : 
            self.layers.append(last_func_to_add(self.width[-1], self.output_size))
        # self.layers.append(nn.Softmax(dim=1))   #NOT CLS! so not softmax
                
    def build_model(self):
        raise ValueError("build_model method must be implemented in derived classes (ex : BMLP, MLP))")
        
    def forward(self, x):
        x = x.view(x.shape[0], -1) #flatten
        for layer in self.layers:
            x = layer(x)
        return x

class MLP(_MLP):
    def build_model(self):
        self.get_layers(BasicBlock)

class MultimodalMLP(nn.Module):
    def __init__(self, input_sizes: List[int], output_size: int, hiddens: List[int], 
                 activation: str, batchnorm: bool, dropout: float = 1, separating_layer: int = 0):
        """
        Initializes a multimodal MLP which handles multiple input branches.
        Concatenation occurs after the 'separating_layer' number of layers in each branch.

        Args:
        input_sizes (List[int]): List of input sizes for each modality.
        output_size (int): The number of output features after all modalities are merged.
        hiddens (List[int]): List of hidden layer sizes for the network.
        activation (str): Activation function to use across all branches.
        batchnorm (bool): Indicates whether to apply batch normalization across all branches.
        dropout (float): Dropout rate to apply across all branches.
        separating_layer (int): The layer number after which concatenation of branch outputs occurs.
        """
        super(MultimodalMLP, self).__init__()
        
        # Initial layers before concatenation
        self.branches = nn.ModuleList([
            nn.Sequential(*[
                BasicBlock(input_size if i == 0 else hiddens[i-1], hiddens[i], bn=batchnorm, activation=activation, dropRate=dropout)
                for i in range(separating_layer)
            ]) for input_size in input_sizes
        ])

        # Combined input size calculation based on the last layer of branches before concatenation
        combined_input_size = hiddens[separating_layer-1] * len(input_sizes)

        # Layers after concatenation
        additional_layers = [
            BasicBlock(combined_input_size if i == separating_layer else hiddens[i-1], hiddens[i], bn=batchnorm, activation=activation, dropRate=dropout)
            for i in range(separating_layer, len(hiddens))
        ]
        self.post_combination_layers = nn.Sequential(*additional_layers)

        # Final output layer
        if len(hiddens) == 1:
            self.final = nn.Linear(combined_input_size, output_size)
        else : 
            self.final = nn.Linear(hiddens[-1], output_size)

    def forward(self, x_list : List[torch.Tensor]):
        # Process each modality through its respective branch up to the separating layer
        outputs = [branch(x) for x, branch in zip(x_list, self.branches)]
        
        # Concatenate the outputs from all branches
        combined = torch.cat(outputs, dim=1)
        
        # Process the combined output through the subsequent layers
        combined = self.post_combination_layers(combined)
        
        # Generate final output
        return self.final(combined)

    #previoius : branch하면서 복잡하게 함.. 걍 list로 순서대로 하기로 함 나는 
    # def forward(self, x_dict: Dict[str, torch.Tensor]):
    #     # Process each modality through its respective branch up to the separating layer
    #     import pdb ; pdb.set_trace()
    #     outputs = [branch(x_dict[key]) for key, branch in zip(x_dict, self.branches)]
        
    #     # Concatenate the outputs from all branches
    #     combined = torch.cat(outputs, dim=1)
        
    #     # Process the combined output through the subsequent layers
    #     combined = self.post_combination_layers(combined)
        
    #     # Generate final output
    #     return self.final(combined)
