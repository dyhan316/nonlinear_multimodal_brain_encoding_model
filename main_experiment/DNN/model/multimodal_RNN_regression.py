import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import torch
import torch.nn as nn
from typing import List, Optional

class MultimodalRNNVecPred(nn.Module):
    r""" 
    RNN (RNN, LSTM, GR) version of MultimodalTFVecPred
    #https://dacon.io/en/competitions/official/236129/codeshare/8827 여기보고 LSTM + layernorm 힌트 얻음! 
    """
    def __init__(self, input_sizes = [1024, 1024], d_input = 512, d_hidden = 128, num_layers = 1, 
                 model_type = "RNN", dropout = 0.1, bidirectional = True, activation = 'tanh', norm_type = None , 
                 linear_first = [True, True], output_size = 1024,  #output_size = False로 하면 identity로 바
                 head_type = 'linear', token_selection_strategy = 'last', fixed_window = True, 
                 delays = [1], modality_embedding = False, separate_modalities = False, *args, **kwargs):
        r""" 
        head_type : 'linear' or 'mlp' or 'none' (None)
        #! 주의 : 일단 위의 token_selection_strategy는 single modality일때만 가능! (i.e. 여러개의 modality일때는 아직 생각안함.. 어떤 modality의 first를 택할지 등등)=> 일단은 무시!
        """
        super().__init__(*args, **kwargs)
        self._validate_lengths(input_sizes, linear_first, delays)
        self.input_sizes = input_sizes
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        self.output_size = output_size
        self.delays = delays
        self.final_out_dim = d_hidden * 2 if bidirectional else d_hidden  #num layers not used since output을 볼것이니
        assert head_type in ['linear', 'mlp', None, 'none'], "head_type must be either 'linear' or 'mlp'"
        
        if token_selection_strategy == 'concat' : 
            assert fixed_window, "fixed_window must be True for token_selection_strategy 'concat' (cuz linear is trained)"
            assert head_type == 'linear', "head_type must be 'linear' for token_selection_strategy 'concat' (if None, then it can't be matched as easily))"
            
        self.input_embeddings = nn.ModuleList([
            nn.Linear(input_size, d_input) if linear_first[i] else nn.Identity() 
            for i, input_size in enumerate(input_sizes)
        ])        
        
        if head_type == 'linear':
            if token_selection_strategy == 'concat' :
                self.head = None #no head for now, will be initialized in forward
                self.head_initialized = False
            else :
                self.head = nn.Linear(self.final_out_dim, self.output_size) #no dropout or activation cuz usually just linear is used
            
            if token_selection_strategy == 'weighted_pool' :
                self.pooling_weight_initilized = False
        elif head_type == 'mlp':
            raise NotImplementedError("MLP head not implemented yet")
        elif head_type == 'none' or head_type is None:
            assert output_size == self.final_out_dim, f"output_size must be the same as the final hidden dimension if head_type is 'none' or None. Currently, output_size is {output_size} and final hidden dimension is {self.final_out_dim}"
            self.head = nn.Identity()
            
        if activation != 'tanh':
            assert model_type == 'RNN', "only tanh activation is supported for LSTM and GRU"
        if norm_type is not None:
            raise NotImplementedError("norm_type  not implemented yet, as this would require not using nn.RNN and such (instead nn.RNNCell should be used to put nonlinearities between layers.. 일단은 고려안함")
        
        if model_type == "RNN":
            self.rnn = nn.RNN(d_input, d_hidden, num_layers, dropout = dropout, bidirectional=bidirectional, nonlinearity = activation)
            self.get_out_and_hidden = lambda output: (output[0], output[1])
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(d_input, d_hidden, num_layers, dropout = dropout, bidirectional=bidirectional)
            self.get_out_and_hidden = lambda output: (output[0], output[1][0])
        elif model_type == "GRU":
            self.rnn = nn.GRU(d_input, d_hidden, num_layers, dropout = dropout, bidirectional=bidirectional)
            self.get_out_and_hidden = lambda output: (output[0], output[1])
        else:
            raise ValueError("model_type must be one of 'RNN', 'LSTM', 'GRU'")

        #token_selection_strategy specific things 
        self.token_selection_strategy = token_selection_strategy
        self.token_selection_func = self._token_selection_func(token_selection_strategy)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model)) if token_selection_strategy == 'cls_token' else None
        # self.fixed_window = fixed_window
        
        #! implement norm_type!! and so on!
        self.fixed_window = fixed_window
        
    def forward(self, *args):
        src_list = list(args)
        src_total, src_lens, src_total_lens = self._process_sources(src_list)
        
        output, hidden = self.get_out_and_hidden(self.rnn(src_total))
        rnn_output = self.token_selection_func(output, hidden)
        return self.head(rnn_output)
    
    def _validate_lengths(self, *args):
        if not all(len(args[0]) == len(arg) for arg in args):
            raise ValueError("All input lists must be of the same length.")
        
    def _process_sources(self, src_list):
        assert isinstance(src_list, list), "src_list (input to the model (model(src_list))must be a list of tensors."
        src_total = []
        src_lens = [src.shape[0] for src in src_list]
        for i, src in enumerate(src_list):
            src = self.input_embeddings[i](src)
            src_total.append(src)
        src_total_lens = [len(i) for i in src_total]
        return torch.cat(src_total, dim=0), src_lens, src_total_lens 

    def _token_selection_func(self, token_selection): 
        if token_selection == 'cls_token' or token_selection == 'first':
            raise NotImplementedError("cls_token and first not implemented for RNNs!")
        elif token_selection == 'last':
            assert self.num_layers == 1, "last token selection only supported for single layer RNNs... or else, last layer hidden만 뽑아내는추가적인 함수를 봐야하는데, 귀찮음.."
            return lambda output, hidden : self._permute_and_flatten(hidden)
        elif token_selection == 'avg_pool':
            return lambda output, hidden  : torch.mean(output, dim=0)
        elif token_selection == 'max_pool':
            return lambda output, hidden : torch.max(output, dim=0)[0]
        elif token_selection == 'concat':
            return self._concat_selection
        elif token_selection == 'weighted_pool':
            return self._weighted_pool_selection
        elif token_selection == 'attention_weighted':
            raise NotImplementedError("attention_weighted not implemented yet")
            # return lambda output: self._attention_weighted(output)
        elif token_selection == 'custom':
            return lambda output: self._custom_selection(output)
        else:
            raise ValueError("token_selection_strategy must be one of 'cls_token', 'first', 'last', 'avg_pool', 'max_pool', 'attention_weighted', 'custom'")

    def _permute_and_flatten(self, output):
        return output.permute(1, 0, 2).flatten(1)
    
    def _concat_selection(self, output, hidden):
        r"""does 1. initialize the linear layer with the correct size 
        (by accounting for the window length) 2. permute and flatten 
        the output"""
        if not self.head_initialized:
            device = output.device
            # Initialize the linear layer based on the window length
            window_length = output.shape[0]  # Assuming the first dimension is the window length
            self.head = nn.Linear(self.final_out_dim * window_length, self.output_size).to(device)
            self.head_initialized = True
        
        # Permute and flatten the output
        flattened_output = self._permute_and_flatten(output)
        return flattened_output  
    
    def _weighted_pool_selection(self, output, hidden):
        r"""does 1. initialize the linear layer with the correct size 
        (by accounting for the window length) 2. permute and flatten 
        the output"""
        if not self.pooling_weight_initilized: #! change this name, or just define a new paremter name thing!
            # Initialize the WeightedRNNPoolying layer based on the window length
            device = output.device
            self.weighted_pooling_layer = WeightedRNNPooling(seq_len = output.shape[0], output_dim = 1).to(device) 
            self.pooling_weight_initilized = True
        return self.weighted_pooling_layer(output)
            
            
class WeightedRNNPooling(nn.Module) : 
    def __init__(self, seq_len=10, output_dim=1):
        super(WeightedRNNPooling, self).__init__()
        self.linear = nn.Linear(seq_len, output_dim)
    
    def forward(self, x):
        # Permute to shape [batch_size, feature_dim, seq_len]
        x = x.permute(1, 2, 0)
        
        # Apply linear layer on feature dimension [batch_size, feature_dim, 1 ]
        x = self.linear(x) 
        
        x = x.flatten(1) #flatten the last two dimensions
        
        return x