import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from utils_transformer import multimodal_mask, reverse_Cosine_PE, Learned_modality_embedding
import torch
import torch.nn as nn
from typing import List, Optional

class MultiPredTransformerDecoder(nn.Module): 
    r"""
    MultiPredTransformerDecoder is a multi-modal Transformer-based decoder for sequence prediction.

    Parameters:
    -----------
    input_sizes : list of int, Feature dimensions for each modality.
    d_model : int, default=1024, Model dimension.
    num_layers : int, default=1, Number of Transformer encoder layers.
    nhead : int, default=10, Number of attention heads.
    ff_hidden_size : int, default=2048, Feedforward network dimension.
    dropout : float, default=0.1, Dropout probability.
    activation : str, default='gelu', Activation function ('relu' or 'gelu').
    norm_type : str, default='layernorm', Normalization type ('layernorm' or 'batchnorm').
    norm_first : bool, default=True, Apply normalization before the feedforward network.
    linear_first : list of bool, Apply a linear transformation first for each modality.
    output_size : int, default=1024, Output dimension of the final linear layer. If None, no final linear transformation.
    masks : list of list of str, Attention mask settings.
    delays : list of int, Delays for each modality.
    pe_functions : list of callable, Positional encoding functions for each modality.
    modality_embedding : bool, default=True, Use learned modality embeddings.
    separate_modalities : bool, default=False, Insert a separator token between modalities.

    Methods:
    --------
    forward(src_list): Forward pass through the Transformer encoder.
    """
    def __init__(self, input_sizes=[1024, 1024], d_model=1024, num_layers=1,
                 nhead=10, ff_hidden_size=2048, dropout=0.1, activation='gelu', norm_type='layernorm', norm_first=True,  #encoder layer params
                 linear_first=[True, False], output_size=1024, masks=[['causal', None], ['full', 'causal']],  delays=[1, 0], #atten mask settings
                 pe_functions=[reverse_Cosine_PE, reverse_Cosine_PE], modality_embedding=True, separate_modalities=False): #PE/modality embedding settings
        #! modify output_size or sth! (cls token가지고 할대는 바꿔야할수도)
        #! inherit the cls_token and make version (modify certain methods and stuff)
        super().__init__()
        self._validate_lengths(input_sizes, masks, linear_first, delays, pe_functions)
        
        self.d_model = d_model
        self.dropout = dropout
        self.linear_first = linear_first
        self.output_size = output_size
        self.masks = masks
        self.delays = delays
        self.sep_between = separate_modalities
        self.src_mask = None
        
        self.sep_token = torch.full((1, 1, d_model), 100)  # 100 : arbitrary number for sep token (not learnable)
        
        self.input_embeddings = nn.ModuleList([
            nn.Linear(size, d_model) if linear_first[i] else nn.Identity()
            for i, size in enumerate(input_sizes)
        ])
        self.pos_encoders = nn.ModuleList([
            pe_functions[i](d_model, dropout=dropout) if pe_functions[i] else nn.Identity()
            for i in range(len(input_sizes))
        ])
        
        self.modality_embeddings = Learned_modality_embedding(d_model, len(input_sizes), dropout) if modality_embedding else None
        
        self.norm = nn.LayerNorm(d_model) if norm_type == 'layernorm' else None
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_hidden_size,
            dropout=dropout, activation=activation, norm_first=norm_first
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers, self.norm)
        self.decoder = nn.Linear(d_model, output_size) if output_size else nn.Identity() #TODO ! may need to be changed if doing cls token prediction? (i.e. make it more flexible, taking in a function or sth?)

    def forward(self, *args):
        src_list = list(args)
        """
        Forward pass through the Transformer decoder.

        Parameters:
        -----------
        src_list : list of tensors
            Input tensors for each modality.

        Returns:
        --------
        output_list : list of tensors
            Output tensors for each modality.
        """
        src_total, src_lens, src_total_lens = self._process_sources(src_list) #src_total_lens : along with sep token!
        device = src_total.device
        if self.src_mask is None:
            self.src_mask = multimodal_mask(src_lens, self.masks, self.delays, self.sep_between).to(device)
        
        output = self.transformer_encoder(src_total, mask=self.src_mask)
        output = self.decoder(output)
        return self._split_output(output, src_total_lens, sep_between=self.sep_between)

    def _validate_lengths(self, *args):
        if not all(len(args[0]) == len(arg) for arg in args):
            raise ValueError("All input lists must be of the same length.")
        
    def _process_sources(self, src_list):
        assert isinstance(src_list, list), "src_list (input to the model (model(src_list))must be a list of tensors."
        src_total = []
        src_lens = [src.shape[0] for src in src_list]
        for i, src in enumerate(src_list):
            src = self.input_embeddings[i](src)
            #TODO : implement the CLS token in front! (for when i==0) (or, inherit this class and prepend the CLS token in the forward method)
            if self.sep_between and i != 0:
                src = torch.cat([self.sep_token.expand(1, src.shape[1], src.shape[2]), src], dim=0)
            src = self.pos_encoders[i](src)
            if self.modality_embeddings:
                src = self.modality_embeddings(src, class_ids=torch.full((src.shape[0],), i, dtype=torch.long, device=src.device)) #returns x + modality embedding (not modality embedding itself), so don't worry (I think) (afterall it is an nn.Module )
            src_total.append(src)
        src_total_lens = [len(i) for i in src_total]
        return torch.cat(src_total, dim=0), src_lens, src_total_lens 

    def _split_output(self, output, src_total_lens, sep_between=False):
        # Initialize list to hold the output slices
        output_list = []

        # Initialize starting index of the first slice
        start_idx = 0

        # Iterate over each sequence length in src_total_list
        for i, length in enumerate(src_total_lens):
            if i == 0:
                # First sequence, take as is
                end_idx = start_idx + length
                output_list.append(output[start_idx:end_idx])
            else:
                if sep_between :
                    # Subsequent sequences with a separator token, skip the separator token at the start
                    start_idx += 1  # Skip one token (the separator)
                    end_idx = start_idx + length - 1  # Subtract one because we're skipping the separator
                else:
                    # Subsequent sequences without a separator token
                    end_idx = start_idx + length            
                output_list.append(output[start_idx:end_idx])

            # Print indices for debugging
            print(f"Segment {i}: Start index {start_idx}, End index {end_idx}")

            # Update start_idx for the next sequence
            start_idx = end_idx if sep_between else end_idx

        return output_list


def print_stats(tnsr, *args):
    print(args, tnsr.shape,tnsr.mean(), tnsr.std(), tnsr.min(), tnsr.max())
    
    
#TODO : to impelment CLs token thing ==> need to make it more general... 

#1. split_output should work when cls token is true or not 
#2. cls token : first modality 앞에 붙음! 
#3. positional embedding을 하나로 다 통일해도 되도록 하기? (like BERT)


class MultimodalTFVecPred(MultiPredTransformerDecoder):
    r""" 
    #!TODO : dataset 에 padding넣기?
    
    """
    def __init__(self, head_type = 'linear', token_selection_strategy = 'cls_token', fixed_window = True, *args, **kwargs):
        r""" 
        head_type : 'linear' or 'mlp'
        token_selection_strategy : 'first', 'cls_token', 'last', 'avg_pool', 'max_pool', 'attention_weighted', 'custom', + more!! 
        #! 주의 : 일단 위의 token_selection_strategy는 single modality일때만 가능! (i.e. 여러개의 modality일때는 아직 생각안함.. 어떤 modality의 first를 택할지 등등)=> 일단은 무시!
        #! TODO : device를 통일해야함!! 어떻게 device를 할지! 
        """
        super().__init__(*args, **kwargs)

        assert head_type in ['linear', 'mlp'], "head_type must be either 'linear' or 'mlp'"
        
        if head_type == 'linear':
            self.head = nn.Linear(self.d_model, self.output_size) #no dropout or activation cuz usually just linear is used
        elif head_type == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(), #fixed to gelu!
                nn.Linear(self.d_model, self.output_size)
            )
            
        #token_selection_strategy specific things 
        self.token_selection_strategy = token_selection_strategy
        self.token_selection_func = self._token_selection_func(token_selection_strategy)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model)) if token_selection_strategy == 'cls_token' else None
        self.fixed_window = fixed_window
        
    def forward(self, *args):
        src_list = list(args)
        # Prepend a CLS token to the first element of src_list
        src_total, src_lens, src_total_lens = self._process_sources(src_list)
        
        #add cls_token token to the things if needed
        src_total = torch.cat([self.cls_token.expand(1, src_total.shape[1], src_total.shape[2]), src_total], dim=0) if self.token_selection_strategy == 'cls_token' else src_total
        
        #if no src_mask (i.e. first forward) or not fixed window, create src_mask! 
        if self.src_mask is None or not self.fixed_window: 
            device = src_total.device
            self.src_mask = multimodal_mask(src_lens, self.masks, self.delays, sep_token=self.sep_between, 
                                            cls_token = (self.cls_token is not None)).to(device)        
        
        output = self.transformer_encoder(src_total, mask=self.src_mask)
        output = self.token_selection_func(output)
        # output = output[0] # Only return the CLS token output #! maybe change this?
        output = self.head(output)
        return output
    
    def _token_selection_func(self, token_selection): 
        if token_selection == 'cls_token' or token_selection == 'first':
            return lambda output: output[0]
        elif token_selection == 'last':
            return lambda output: output[-1]
        elif token_selection == 'avg_pool':
            return lambda output: torch.mean(output, dim=0)
        elif token_selection == 'max_pool':
            return lambda output: torch.max(output, dim=0)[0]
        elif token_selection == 'attention_weighted':
            raise NotImplementedError("attention_weighted not implemented yet")
            # return lambda output: self._attention_weighted(output)
        elif token_selection == 'custom':
            return lambda output: self._custom_selection(output)
        else:
            raise ValueError("token_selection_strategy must be one of 'cls_token', 'first', 'last', 'avg_pool', 'max_pool', 'attention_weighted', 'custom'")
