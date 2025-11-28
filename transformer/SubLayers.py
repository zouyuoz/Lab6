''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import flash_attn
from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_func
from transformer.utils import seqlen2cu_len
class MultiHeadSelfAttention_Flash(nn.Module):
    ''' Multi-Head self Attention module with Flash Attention '''

    def __init__(self, n_head, d_model, d_qkv, dropout=0.1, causal=False):
        super().__init__()
        self.n_head = n_head
        self.d_qkv = d_qkv
        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False) # QKV projection (d_model -> 3 * d_model)
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Output projection (d_model -> d_model)
        self.dropout_rate = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.causal = causal
    def forward(self, x, seq_lens):
        drop_rate = self.dropout_rate if self.training else 0.0
        residual = x
        ################# YOUR CODE HERE #################
        # 1. use self.w_qkv to get qkv and reshape to (total_len, 3, n_head, d_qkv)
        qkv = self.w_qkv(x)
        # Reshape: (B*L, 3, H, D) -> (total_tokens, 3, n_head, d_qkv)
        qkv_packed = qkv.view(-1, 3, self.n_head, self.d_qkv) 

        # 2. change seq_lens to cu_seqlens by using seqlen2cu_len function
        cu_seqlens = seqlen2cu_len(seq_lens)
        
        # 3. get max_len from seq_lens
        max_len = seq_lens.max().item()
        ##################################################
        
        output = flash_attn_varlen_qkvpacked_func(qkv_packed, cu_seqlens, max_len, dropout_p=drop_rate, causal=self.causal)
        output = output.reshape(-1, self.n_head * self.d_qkv)
        
        ################# YOUR CODE HERE ###################
        # 1. use self.w_o to project output back to d_model
        output = self.w_o(output)
        
        # 2. apply dropout, residual connection and layer norm
        output = self.dropout_layer(output)
        output += residual
        output = self.layer_norm(output)
        ##################################################
        return output

class MultiHeadCrossAttention_Flash(nn.Module):
    def __init__(self, n_head, d_model, d_qkv, dropout=0.1, causal=False):
        super().__init__()
        self.n_head = n_head
        self.d_qkv = d_qkv
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout_layer = nn.Dropout(dropout)
        self.dropout_rate = dropout
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.causal = causal
    def forward(self, x_q, x_kv, seq_lens_q, seq_lens_kv):
        drop_rate = self.dropout_rate if self.training else 0.0
        residual = x_q
        ################# YOUR CODE HERE #################
        # 1. use self.w_q, self.w_kv to get q, k, v
        q = self.w_q(x_q).view(-1, self.n_head, self.d_qkv)
        # kv shape: (total, 2, n_head, d_qkv)
        kv = self.w_kv(x_kv).view(-1, 2, self.n_head, self.d_qkv)
        k, v = kv.unbind(1)

        # 2. change seq_lens to cu_seqlens
        cu_seqlens_q = seqlen2cu_len(seq_lens_q)
        cu_seqlens_kv = seqlen2cu_len(seq_lens_kv)

        # 3. get max_len
        max_len_q = seq_lens_q.max().item()
        max_len_kv = seq_lens_kv.max().item()
        ##################################################
        
        output = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max_len_q,
            max_seqlen_k=max_len_kv,
            dropout_p=drop_rate,
            causal=self.causal,
        )
        
        ################# YOUR CODE HERE ###################
        # 1. Project back
        output = self.w_o(output.reshape(-1, self.n_head * self.d_qkv))
        
        # 2. apply dropout, residual, layer norm
        output = self.dropout_layer(output)
        output += residual
        output = self.layer_norm(output)
        ##################################################
        return output
    
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
