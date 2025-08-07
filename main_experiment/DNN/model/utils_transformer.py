import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

##############Positional Encoding##############
class Cosine_PE(nn.Module):
    def __init__(self, d_model, max_len=1000, exp_base=10000, phase_shift=0, dropout=0.0):
        super(Cosine_PE, self).__init__()
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        
        # Precompute the div_term for efficiency
        div_term = 1 / (exp_base ** (torch.arange(0.0, d_model, 2.0) / d_model))
        
        # Precompute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * div_term + phase_shift)
        pe[:, 1::2] = torch.cos(position * div_term + phase_shift)
        
        # Register positional encodings as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(0)
        x = x + self.pe[:seq_len].unsqueeze(1)
        return self.dropout(x)
    
class reverse_Cosine_PE(Cosine_PE):
    def __init__(self, d_model, max_len=1000, exp_base=10000, phase_shift=0, dropout=0.0):
        super(reverse_Cosine_PE, self).__init__(d_model, max_len, exp_base, phase_shift, dropout)
        
    def forward(self, x):
        seq_len = x.size(0)
        x = x + self.pe[:seq_len].flip(dims=[0]).unsqueeze(1)
        return self.dropout(x)
    
class Learned_PE(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.0):
        super(Learned_PE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create learned positional embeddings
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x, backward=False):
        seq_len = x.size(0)
        position = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        if backward:
            position = position.flip(dims=[0])
        
        # Get positional embeddings
        pe = self.pe(position).unsqueeze(1)
        
        x = x + pe
        return self.dropout(x)

class Learned_modality_embedding(nn.Module):
    def __init__(self, d_model, num_classes, dropout=0.0):
        super(Learned_modality_embedding, self).__init__()
        
        # Create learned class embeddings
        self.modality_embedding = nn.Embedding(num_classes, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, class_ids):
        modality_embedding = self.modality_embedding(class_ids)
        return self.dropout(x + modality_embedding.unsqueeze(1))
    
##############Attention Masking##############

def multimodal_mask(input_len_list, inter_causality, delay_list, sep_token=False, cls_token=False):
    r""" 
    #assumption : sequences들의 end는 같은 시각이구, delay가 sequences간의 delay를 의미함
    #below : with respect to the LAST element of input_len_list!! (i.e. audio, llm, resp) 면, resp기준으로!! 
    
    * input_len_list : list of input lengths, SHOULD NOT INCLUDE SEP TOKEN!
    * inter_causality : what the causlaity (or full connection) between the input sequences should be (i.e. between modalities)
        Should be a 2D list of types, 
        * None : no connection
        * 'full' : full connection
        * 'causal' : causal connection (with delay)
    * sep_token : whether the output attention mask should have a sep token in between the modalities
    * delay_list : the delay in the teacher forced mask (i.e. how many steps to look back), last element should be 0 ==> i.e. 
    
    * output atten mask shape : (sum(input_len_list) + alpha, sum(input_len_list) + alpha), where alpha = 0 if sep_token = False, len(input_len_list)-1 if sep_token = True
    """
    total_len = sum(input_len_list) 
    mask = torch.full((total_len, total_len), float('-inf'))

    cumulative_start_indices = [0]
    for i in range(1, len(input_len_list)):
        next_index = cumulative_start_indices[i-1] + input_len_list[i-1] 
        cumulative_start_indices.append(next_index)

    for i in range(len(input_len_list)):
        start_idx = cumulative_start_indices[i]
        end_idx = start_idx + input_len_list[i]
        mask[start_idx:end_idx, start_idx:end_idx] = 0
    mask = apply_inter_causality_mask(mask, inter_causality, cumulative_start_indices, input_len_list, delay_list)
    
    if sep_token :
        mask = add_sep_token_between(mask, cumulative_start_indices, input_len_list)
    if cls_token :
        mask = add_cls_token(mask)

    return mask

def add_cls_token(mask):
    mask = torch.cat((torch.full((1, mask.shape[1]), 0.), mask), dim=0) #
    mask = torch.cat((torch.full((mask.shape[0], 1), 0.), mask), dim=1)  #
    
    return mask

def apply_inter_causality_mask(mask, inter_causality, start_indices, input_len_list, delay_list):
    num_modalities = len(input_len_list)
    
    # Adjust delays based on position in the sequence
    adjusted_delays = [[-delay_list[i] + delay_list[j] for j in range(num_modalities)] for i in range(num_modalities)]
    for i in range(num_modalities):
        for j in range(num_modalities):
            start_i = start_indices[i]
            end_i = start_i + input_len_list[i]
            start_j = start_indices[j]
            end_j = start_j + input_len_list[j]
            # print(f"s_i : {start_i}, e_i : {end_i}, s_j : {start_j}, e_j : {end_j}")
            if inter_causality[i][j] is None:
                mask[start_i:end_i, start_j:end_j] = float('-inf')
                #continue  # No connection
            elif inter_causality[i][j] == 'full':
                mask[start_i:end_i, start_j:end_j] = 0
            elif inter_causality[i][j] == 'causal':
                causal_mask = torch_lower_triangle(end_i - start_i, end_j - start_j, delay=adjusted_delays[i][j])
                mask[start_i:end_i, start_j:end_j] = causal_mask
            else : 
                raise ValueError(f"Invalid inter_causality value: {inter_causality[i][j]}")
    return mask


def add_sep_token_between(mask, start_indices, input_len_list):
    #manually inserts a sep column/row to the mask at between the modalities
    # ... (since torch.tnsr is not mutable, it needs to split the mask and concat with the new row and col) #col : 0, row : '-inf'
    sep_indices = [start_indices[i] + input_len_list[i] for i in range(len(input_len_list))]
    sep_indices.pop() #remove the last sep index
    
    #row first, THAN columns cuz of the way the mask is constructed
    for i, sep_idx in enumerate(sep_indices) :
        split_idx = sep_idx + i 
        mask = torch.cat((mask[:split_idx, :], torch.full((1, mask.shape[1]), float('-inf')), mask[split_idx:, :]), dim=0) #-inf
    for i, sep_idx in enumerate(sep_indices) :
        split_idx = sep_idx + i 
        mask = torch.cat((mask[:, :split_idx], torch.full((mask.shape[0], 1), 0.), mask[:, split_idx:]), dim=1)  #0
    return mask
    

def torch_lower_triangle(sz_i, sz_j, delay=0):
    diag_adjusted = -sz_i +sz_j - delay
    mask = torch.tril(torch.ones(sz_i, sz_j),diagonal=diag_adjusted)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)) #* fill 0 with -inf, 1 with 0
    return mask