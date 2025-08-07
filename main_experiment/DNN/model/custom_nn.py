import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules.module import Module
from torch import Tensor
import math
import warnings


def block_Bilinear(x,W_stack,b = None, block_b = None, sum_b = None,
                   block_weight = None, sum_weight = None,
                   input_activation = None, block_activation = None, sum_activation = None):
    """ 
    Perform a block-wise self bilinear operation on the input tensor. (주의 : assume block diagonal is square with uniform size)

    Args:
        x (torch.Tensor): The input tensor of shape (B, X), where B is the batch size and X is the input dimension.
        W_stack (torch.Tensor): The stacked diagonal blocks of the weight tensor, of shape (D, M, K, K), where D is the output dim, M is the number of blocks, and K is the (square) block size
        b (torch.Tensor): The bias tensor of shape (D,), where D is output dimension.
        block_b (torch.Tensor) : The bias tensor of shape (D,M), where D is output dimension and M is the number of blocks.
        sum_b (torch.Tensor) : The bias tensor of shape (D,), where D is output dimension. 
        block_weight (torch.Tensor) : The weight tensor of shape (D,M), where D is output dimension and M is the number of blocks.
        sum_weight (torch.Tensor) : The weight tensor of shape (D,), where D is output dimension. 
        
        
    Returns:
        torch.Tensor: The result of the block-wise bilinear operation, of shape (B, D).
        equation : sub_activation(sum(block_activation(xWx^T + block_b))+sum_b) + b

    Raises:
        None
    """
    #get B,X, ...
    B,X = x.shape
    D,M,K,_ = W_stack.shape
    
    
    #reshape x to block-sized (i.e B,X -> B,M,K)
    x_stack = x.reshape(B,M,K)

    #inject extra dim to W_stack and x_stack (injected dim is the block dim)
    W_stack = W_stack.unsqueeze(1) #D,1,M,K,K #1에 x 의 batch dim이 들어갈 것임 
    x_stack = x_stack.unsqueeze(-2) #B,M,1,K #1에 W의 batch dim이 들어갈 것임

    #apply input activation if provided
    if input_activation is not None :
        x_stack = input_activation(x_stack)
    
    #actual computation
    result = torch.matmul(x_stack,torch.matmul(W_stack,x_stack.transpose(-1,-2)))

    #squeeze the last two dims
    result = result.squeeze((-1,-2)) #D,B,M,1,1 -> D,B,M
    
    #add block bias if provided
    if block_b is not None :
        result += block_b.unsqueeze(1) #unsuqeeze to create batch dim to broadcast to 

    #apply block wise activation 
    if block_activation is not None : 
        result = block_activation(result)
        
    if block_weight is not None :
        result = result*block_weight.unsqueeze(1)    
    
    #sum up all the blocks (M)
    #apply block wise activation
    result = result.sum(dim=-1) 
        
    #transpose finally 
    result =  result.T #B,D
    
    #apply sum_weight if provided #! check if correct!
    if sum_weight is not None :
        result = result*sum_weight.unsqueeze(0)
    
    #add sum_b if provided
    if sum_b is not None :
        result += sum_b
                
    #apply total activation 
    if sum_activation is not None : 
        result = sum_activation(result)
    
    #add bias if provided
    result += b if b is not None else 0
    
    return result



class _dend_Bilinear(Module):
    r"""Applies a bilinear transformation to the incoming data: :math:`y = x_1^T A x_2 + b`.
    Args:
        in_features: size of each first input sample
        out_features: size of each output sample
        block_size : size of each block 
        padding_mode : padding mode to use. Can be one of {"zero", "replicate", "reflect", "none"} or None.
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :#TODO 
        - Output: #TODO 

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{out\_features//kernel\_size},\text{in\_features}, \text{in\_features})`.

    Examples::

        >>> m = SBilinear(in_features=20, out_features=30, block_size=4)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        #torch.Size([128, 30])
    """

    __constants__ = ['in_features', 'out_features','block_size'] #Modified (added block_size)
    in_features: int
    out_features: int
    block_size : int #Modified (added block_size)
    weight: Tensor

    def __init__(self, in_features: int,  out_features: int, block_size : int, padding_mode : str = "zero",
                 block_weight : bool = False, sum_weight : bool = False, #TODO MAJOR CHANGE NEEDED! (current implementation : will bring chaos)
                 bias: bool = True, block_bias : bool = False, sum_bias : bool = False,
                 block_activation = None, sum_activation = None, input_activation = None,
                  device=None, dtype=None, initial_wiring = None) -> None:
        #TODO modify the initial_wiring part (i.e. how to initialize the rewire_indx)
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.padding_mode = padding_mode
        
        self.factory_kwargs = factory_kwargs 
        if in_features % block_size != 0 :
            warnings.warn(f"input size not divisible by block_size, so padding_mode {padding_mode} was used") 
            self.pad_func = self.get_pad_func(in_features, block_size)
            self.num_blocks = in_features//block_size + 1 if padding_mode != "none" else in_features//block_size #one extra block to take in padded x if not "none"
        else :
            self.pad_func = lambda x : x
            self.num_blocks = in_features//block_size
            
        self.block_size = block_size
        
        ##weight part
        self.weight = Parameter(torch.empty((out_features, self.num_blocks, block_size, block_size), **self.factory_kwargs))
                
        if block_weight :
            self.block_weight = Parameter(torch.empty(out_features, self.num_blocks, **self.factory_kwargs))
        else :
            self.register_parameter('block_weight', None)
        if sum_weight :
            self.sum_weight = Parameter(torch.empty(out_features, **self.factory_kwargs))
        else :
            self.register_parameter('sum_weight', None)
            
        ##bias part
        if bias:
            self.bias = Parameter(torch.empty(out_features, **self.factory_kwargs))
        else:
            self.register_parameter('bias', None)

        if block_bias :
            self.block_bias = Parameter(torch.empty(out_features, self.num_blocks, **self.factory_kwargs))
        else :
            self.register_parameter('block_bias', None)
            
        if sum_bias :
            self.sum_bias = Parameter(torch.empty(out_features, **self.factory_kwargs))
        else :
            self.register_parameter('sum_bias', None)
            
        ##activation part
        self.input_activation = input_activation
        self.block_activation = block_activation
        self.sum_activation = sum_activation            
            
        self.reset_parameters()
        self.reset_parameters_other()
        
        #TODO DO BELOW!
        # self.rewire_indx = self.rewire(initial_wiring) #TODO : initialize difficult if input is not known... but in_features있으니 괜찮을듯
        #rewiring index (i.e. permutation index) of input features (will be used in forward))
        #TODO rewire might be complicated by the fact that padding might have been used (i.e. in_features might be different from the original in_features)
         

    def reset_parameters(self) -> None:
        #TODO : find better initialization! (kaiming for Biliear?)
        bound = 1 / math.sqrt(self.weight.size(1))
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)
    
    def reset_parameters_other(self) -> None:
        #TODO change this for better, for now just regular kaiming 
        #initialize for block_weight, sum_weight, block_bias, sum_bias
        if self.block_weight is not None:
            init.kaiming_uniform_(self.block_weight, a=math.sqrt(5))
        if self.sum_weight is not None:
            init.kaiming_uniform_(self.sum_weight, a=math.sqrt(5))
        if self.block_bias is not None:
            init.kaiming_uniform_(self.block_bias, a=math.sqrt(5))
        if self.sum_bias is not None:
            init.kaiming_uniform_(self.sum_bias, a=math.sqrt(5))

    def forward(self, input: Tensor) -> Tensor:
        input = self.pad_func(input)
        #return block_Bilinear(input,self.weight,self.bias)
        return block_Bilinear(input,self.weight,self.bias, self.block_bias, self.sum_bias,
                              self.block_weight, self.sum_weight,
                              self.input_activation, self.block_activation, self.sum_activation)

        
    def get_pad_func(self, in_features, block_size):
        padding_modes = {
            "zero": ("zero", lambda x: F.pad(x, (0, block_size - in_features % block_size), mode='constant', value=0)),
            "replicate": ("replicate", lambda x: F.pad(x, (0, block_size - in_features % block_size), mode='replicate')),
            "reflect": ("reflect", lambda x: F.pad(x, (0, block_size - in_features % block_size), mode='reflect')),
            "none": ("none", lambda x: x[:, :-(in_features % block_size)])
        }

        if self.padding_mode in padding_modes:
            _, padding_func = padding_modes[self.padding_mode]
            return padding_func
        else:
            raise NotImplementedError
        
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, block_size={}, num_blocks={}, bias={}'.format(
            self.in_features, self.out_features, self.block_size, self.num_blocks ,self.bias is not None
        )
    
    #rewire intra-neuron(dendrite level) AND inter-neuron(connectivity level) AND inter-neuronal clusters(connectivity of clusters level) #TODO : organize into this and rename?
    # def rewire(self, method : str, **kwargs) : #TODO 
    #     # reorder_batched_tnsr(self., method, **kwargs)
    #     raise NotImplementedError("rewiring not implemented yet")
            
    #rewire : permute (reorder(?)) + initialization => divide rewire into two things?
    
    
    
    #def rewire(self, input : Tensor, method : str) : 
        
    # def shuffle_pairing(self, shuffle_order):
    #     return NotImplemented, "Reordering the weight pairing thing, need to be done (위에 markdown에 things to implement에 적은것들 해보기..) "
    
    #ADD linear option 
    
    #TODO : initial rewiring option AND rewire optoin 두가지... 
    #TODO : add option to use non uniform block size => then for loop with compile or toch.sparse or zero padding of W or grouping into 2,4,8 sized and so on, then concatting them then doing regular thing we did  or sth (the torch community response보기)
    #TODO : define stochastic pairing (use shuffle pariing or sth)
    #TODO : maybe? add abiliity to count how many times a certain thing wa used or sth.. or EWMA (for counting uses)
    #TODO : however, EWMA를 직접 만들기보다는, 이 class 를 inherit하는 다른아이에서 method만 재정의해줘도될듯
    #TODO : implement prunign (def prune), (i.e. set either bottom 10% of weights to zero, or like below a threshold zero or stb)
    #TODO : we want to test if there is a rewiring rule that enables block_Bilinear to become line convolution (i.e. learning by itself to become like convolution(looking at nearby blocks))
    #TODO : maybe that "self learnt" rule can overpower even the default CNN?
    
    #TODO : hebbian block rewiring rule, dnedritic block rewiring rule, 등등 만들기
    #TODO : LINEAR REWIRING시 : (two rewrigin options : completely rewiring all connections (BILINEAR AND LINEAR), or only rewiring the interaction(bilinear) terms (reassigining the different block)
    
    #DONE, NEEDS TESTING
    #TODO : linear version inside bilinear (also rewiring based on that)
    #TODO : do this instead of manually adding x!! 
    
    #DONE 
    #implement not divisible case (pad or sth)
    
class dend_Bilinear(_dend_Bilinear):
    r"""
    #just a wrapper for _dend_Bilinear  (another name for _dend_Bilinear)
    #_dend_Bilinear is just bilinear, so same as this
    """
    def __init__(self, in_features: int, out_features: int, block_size: int, padding_mode: str = "zero",
                 bias: bool = True, device=None, dtype=None, **kwargs) -> None:
        super().__init__(in_features, out_features, block_size, padding_mode = padding_mode, bias = bias, device = device, dtype = dtype)

class regular_Bilinear(_dend_Bilinear):
    r""" 
    * no blocks, just Bilinear (i.e. regular bilinear, as in nn.Bilinear)
    #did this cuz later, could add rewiring or sth, so a unified interface would be nice
    """
    def __init__(self, in_features: int, out_features: int, padding_mode: str = "zero",
                 bias: bool = True, device=None, dtype=None, **kwargs) -> None:
        super().__init__(in_features, out_features, in_features, padding_mode = padding_mode, bias = bias, device = device, dtype = dtype) 
        #i.e. block_size = in_features (i.e. no blocks)
    
class dend_Linear_Bilinear(_dend_Bilinear):
    r""" 
    #partly ported from nn.Linear source code
    
    use this if you want to also add linear to the bilinear part. (i.e. bilinear + linear)
    inherits from dend_Bilinear
    
    * adds "linear_factor" parameter (default 1.0) to control the relative strength of the linear part (i.e. linear_factor * linear + bilinear)
    """
    #TODO : below does not care about blocks within linear... think about whetehr and how to do this
    def __init__(self, in_features: int, out_features: int, block_size: int, linear_factor : float = 1.0,  padding_mode: str = "zero",
                 block_weight : bool = False, sum_weight : bool = False,
                 bias: bool = True, block_bias : bool = False, sum_bias : bool = False,
                 block_activation = None, sum_activation = None, input_activation = None,
                 device=None, dtype=None, initial_wiring = None) -> None:
                #  bias: bool = True, device=None, dtype=None) -> None:
        self.linear_factor = linear_factor
        super().__init__(in_features, out_features, block_size, padding_mode, 
                         block_weight, sum_weight,
                         bias, block_bias, sum_bias, block_activation, sum_activation, input_activation,
                         device, dtype, initial_wiring)
        
        self.weight_linear = Parameter(torch.empty((out_features, in_features), **self.factory_kwargs))
        self.reset_linear_parameters() #reset the linear parameters 
    
    #partly from nn.Linear code
    def reset_linear_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight_linear, a=math.sqrt(5))
        #removed bias initialization (as it is already done in the bilinear part)
    
    def forward(self, input: Tensor) -> Tensor:
        #returns bilinear + linear (bilinear can possibly be affine if bias = True)
        input = self.pad_func(input)
        return block_Bilinear(input,self.weight,self.bias, self.block_bias, self.sum_bias,
                                self.block_weight, self.sum_weight,
                                self.input_activation, self.block_activation, self.sum_activation) + self.linear_factor*F.linear(input, self.weight_linear, None)

class dend_Linear_Bilinear_block_activation(dend_Linear_Bilinear):
    r""" 
    general class to use, for doing block activation (i.e. activation applied to each block), and adding them and so on and so on 
    """
    def __init__(self, in_features: int, out_features: int, block_size: int, linear_factor : float = 1.0,  padding_mode: str = "zero",
                 bias = True, block_bias = True, block_activation = torch.tanh, block_weight = True, **kwargs) -> None:
        self.linear_factor = linear_factor
        super().__init__(in_features, out_features, block_size = block_size, linear_factor=linear_factor,padding_mode = padding_mode, 
                         bias = bias, block_bias = block_bias, block_activation = block_activation, block_weight=block_weight, **kwargs)
    

class regular_Linear_Bilinear(dend_Linear_Bilinear):
    r""" 
    * just regular linear + Bilinear, without any blocks (inherits from dend_Linear_Bilinear)
    """
    def __init__(self, in_features: int, out_features: int, linear_factor : float = 1.0, padding_mode: str = "zero",
                 bias: bool = True, device=None, dtype=None, **kwargs) -> None:
        super().__init__(in_features, out_features, in_features, linear_factor, padding_mode, bias = bias, device = device, dtype = dtype) 
        #i.e. block_size = in_features (i.e. no blocks)

class regular_Linear_Bilinear_block_activation(dend_Linear_Bilinear_block_activation):
    r""" 
    * just regular linear + Bilinear, without any blocks 
    * but with tanh applied to the bilinear term
    """
    def __init__(self, in_features: int, out_features: int, linear_factor = 1.0, padding_mode = "zero",
                 bias: bool = True, block_activation = torch.tanh,
                 block_weight = True, block_bias = True,
                 device=None, dtype=None, **kwargs) -> None:
        super().__init__(in_features, out_features, in_features, linear_factor, padding_mode, bias = bias, 
                         device = device, dtype = dtype, block_weight=block_weight, block_bias=block_bias, block_activation=block_activation)
        #i.e. block_size = in_features (i.e. no blocks)  

#REDO THIS!
class regular_Linear_Bilinear_sqrt(dend_Linear_Bilinear): 
    r""" 
    * just regular linear + Bilinear, without any blocks 
    * but with sqrt(abs(x))*sign(x) activation applied to the input for initilalization
    """
    def __init__(self, in_features: int, out_features: int, linear_factor : float = 1.0, padding_mode: str = "zero",
                 bias: bool = True, input_activation = lambda x : torch.sqrt(torch.abs(x)+1e-8)*torch.sign(x),
                 device=None, dtype=None, **kwargs) -> None:
        super().__init__(in_features, out_features, in_features, linear_factor=linear_factor, padding_mode=padding_mode, bias = bias, 
                         device = device, dtype = dtype, input_activation=input_activation)
        #i.e. block_size = in_features (i.e. no blocks)  

class regular_Linear_Bilinear_sqrt_residual_1(dend_Linear_Bilinear): 
    r""" 
    * just regular linear + Bilinear, without any blocks 
    * but with sqrt(abs(x))*sign(x) activation applied to the input for initilalization
    * and residual connection (i.e. x + bilinear + linear)
    """
    def __init__(self, in_features: int, out_features: int, linear_factor : float = 1.0, padding_mode: str = "zero",
                 bias: bool = True, input_activation = lambda x : torch.sqrt(torch.abs(x)+1e-8)*torch.sign(x),
                 device=None, dtype=None, **kwargs) -> None:
        super().__init__(in_features, out_features, in_features, linear_factor=linear_factor, padding_mode=padding_mode, bias = bias, 
                         device = device, dtype = dtype, input_activation=input_activation)
        #i.e. block_size = in_features (i.e. no blocks)  
        
    def forward(self, input: Tensor) -> Tensor:
        #returns bilinear + input
        input = self.pad_func(input)
        return block_Bilinear(input,self.weight,self.bias, self.block_bias, self.sum_bias,
                                self.block_weight, self.sum_weight,
                                self.input_activation, self.block_activation, self.sum_activation) + input

##TODO 
# class dend_Linear_Bilinear_sqrt(_dend_Bilinear):
# class regular_Linear_Bilinear_sqrt_scaled(_dend_Bilinear):



#block activation 은 regular일떄는 안해도될듯?
# class dend_Linear_Bilinear_block_acti(dend_Linear_Bilinear): 
#     r""" 
#     for taking 
#     """

# class regular_Linear_Bilinear_sqrt(dend_Linear_Bilinear):
#     r""" 
#     * just regular linear + Bilinear, without any blocks (inherits from dend_Linear_Bilinear)
#     * but with sqrt(abs(x))*sign(x) activation applied to the input for initilalization
#     """
#     def __init__(self, in_features: int, out_features: int, linear_factor : float = 1.0, padding_mode: str = "zero",
#                  bias: bool = True, device=None, dtype=None, **kwargs) -> None:
#         super().__init__(in_features, out_features, in_features, linear_factor, padding_mode, bias, device, dtype) 
#         #i.e. block_size = in_features (i.e. no blocks)
    
#     def forward(self, input: Tensor) -> Tensor:
#         #returns bilinear + linear (bilinear can possibly be affine if bias = True)
#         input = self.pad_func(input)
#         return block_Bilinear(torch.sqrt(torch.abs(input))*torch.sign(input),self.weight,self.bias) + self.linear_factor*F.linear(input, self.weight_linear, None)
    
    
    
# class regular_Linear_Bilinear_sqrt_scaled(dend_Linear_Bilinear):
#     r""" 
#     * just regular linear + Bilinear, without any blocks (inherits from dend_Linear_Bilinear)
#     * but with sqrt(abs(x))*sign(x) activation applied to the input for initilalization
#     """
#     def __init__(self, in_features: int, out_features: int, linear_factor : float = 1.0, padding_mode: str = "zero",
#                  bias: bool = True, device=None, dtype=None, **kwargs) -> None:
#         super().__init__(in_features, out_features, in_features, linear_factor, padding_mode, bias, device, dtype) 
#         #i.e. block_size = in_features (i.e. no blocks)
    
#     def forward(self, input: Tensor) -> Tensor:
#         #returns bilinear + linear (bilinear can possibly be affine if bias = True)
#         input = self.pad_func(input)
#         return block_Bilinear(torch.sqrt(torch.abs(input))*torch.sign(input),self.weight,self.bias) + self.linear_factor*F.linear(input, self.weight_linear, None)
    
# #TODO : also done one with sqrt divided by sqrt of block size (i.e. sqrt(abs(x))/sqrt(block_size)*sign(x) activation applied to the input for initilalization)

