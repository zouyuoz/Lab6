''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # 1. 投影並轉換維度 (Batch, Head, Len, Dim)
        # 這裡不需要 transpose 回來，直接保持 (B, H, L, D) 給 SDPA 吃
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k).transpose(1, 2)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k).transpose(1, 2)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v).transpose(1, 2)
        
        # 2. 處理遮罩 (Mask)
        # SDPA 需要的 mask 形狀是 (B, 1, L, L) 以便廣播
        # 且最好轉換為 Float Mask: 0.0 代表保留, -inf 代表遮蔽
        attn_mask = None
        if mask is not None:
            # 如果 mask 是 3 維 (B, L, L)，先加一個維度變 (B, 1, L, L)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            
            # 建立一個全為 0 的矩陣，將 mask 為 False (原本是被遮住的地方) 的位置填入 -inf
            # 你的 mask 邏輯是: True=保留, False=Padding/Future (要遮掉)
            attn_mask = torch.zeros_like(mask, dtype=q.dtype)
            attn_mask.masked_fill_(~mask, float("-inf"))

        # 3. 使用 PyTorch 內建的高效 Attention (SDPA)
        # 這行會自動啟用 Flash Attention，速度提升數倍
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )

        # 4. 組合 Heads 並輸出
        # output: (Batch, Head, Len, Dim) -> (Batch, Len, Head, Dim) -> (Batch, Len, Dim_Model)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output += residual
        output = self.layer_norm(output)

        # 注意：SDPA 預設不回傳 attention weights (為了速度)。
        # 由於你的 Models.py 裡面是用 `dec_output, _, _ = dec_layer(...)` 接收，
        # 所以這裡回傳 None 是安全的，且能省下更多記憶體。
        return output, None


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