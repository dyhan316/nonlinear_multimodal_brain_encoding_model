import torch
import torch.nn as nn
import numpy as np
import time
import math

torch.manual_seed(0)
np.random.seed(0)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(
        #     torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        # )
        div_term = 1 / (10000 ** ((2 * np.arange(d_model)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])

        pe = pe.unsqueeze(0).transpose(0, 1) # [5000, 1, d_model],so need seq-len <= 5000
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(self.pe[:x.size(0), :].repeat(1,x.shape[1],1).shape ,'---',x.shape)
        # dimension 1 maybe inequal batchsize
        return x + self.pe[:x.size(0), :].repeat(1,x.shape[1],1)

###! for BRAIN DECODING
class single_pred_TF_decoder(nn.Module): #single (resp or stim)만 prediction하는 Transformer Decoder
    def __init__(self, input_size=250, num_layers=1, nhead = 10, ff_hidden_size = 512,
                 dropout=0.1,activation = 'gelu', norm = "layernorm", norm_first = True, **kwargs):
        super(single_pred_TF_decoder, self).__init__()
        self.model_type = 'Transformer'
        # self.input_embedding  = nn.Linear(1,input_size)
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(input_size) #TODO : add more options and stuff!! (like dropout, etc)(cuz positional encoding should have dropout!)
        
        if norm == "layernorm":
            self.norm = nn.LayerNorm(input_size)
        else : 
            raise ValueError("Only layernorm is supported implemented for now ")
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward = ff_hidden_size,
                                                        dropout=dropout, norm_first = norm_first, activation = activation) 
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers, norm=self.norm)
        # self.decoder = nn.Linear(input_size,1)
        # self.init_weights()

    def init_weights(self):
        pass #no need to do since the self.decoder is not used
        # initrange = 0.1    
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        # src with shape (input_window, batch_len, dim)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        # src = self.input_embedding(src) # linear transformation before positional embedding
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        # output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# class multi_pred_TF_decoder(nn.Module): #takes in both stim and resp, predicts resp!
#     def __init__(self, input_size=250, num_layers=1, nhead = 10, ff_hidden_size = 512,
#                  dropout=0.1,activation = 'gelu', norm = "layernorm", norm_first = True, **kwargs):
#         super(single_pred_TF_decoder, self).__init__()
#         self.model_type = 'Transformer'
#         # self.input_embedding  = nn.Linear(1,input_size)
#         self.src_mask = None

#         self.pos_encoder = PositionalEncoding(input_size) #TODO : add more options and stuff!! (like dropout, etc)(cuz positional encoding should have dropout!)
        
#         if norm == "layernorm":
#             self.norm = nn.LayerNorm(input_size)
#         else : 
#             raise ValueError("Only layernorm is supported implemented for now ")
        
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward = ff_hidden_size,
#                                                         dropout=dropout, norm_first = norm_first, activation = activation) 
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers, norm=self.norm)
#         # self.decoder = nn.Linear(input_size,1)
#         # self.init_weights()

#     def init_weights(self):
#         pass #no need to do since the self.decoder is not used
#         # initrange = 0.1    
#         # self.decoder.bias.data.zero_()
#         # self.decoder.weight.data.uniform_(-initrange, initrange)

#     def forward(self,src):
#         # src with shape (input_window, batch_len, dim)
#         if self.src_mask is None or self.src_mask.size(0) != len(src):
#             device = src.device
#             mask = self._generate_square_subsequent_mask(len(src)).to(device)
#             self.src_mask = mask
#         # src = self.input_embedding(src) # linear transformation before positional embedding
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
#         # output = self.decoder(output)
#         return output

#     def _generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask

