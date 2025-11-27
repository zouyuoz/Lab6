''' Define the Transformer model using Padding/Masking '''
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from transformers import ModernBertModel, AutoTokenizer
from transformer.Layers import DecoderLayer 
from transformer.Constants import *
import torch.nn.functional as F

# --- MASKING & UTILITIES (Padding 模式所需) ---

def get_pad_mask(seq, pad_idx):
    # 遮罩 shape: (B, 1, L)
    return (seq == pad_idx).unsqueeze(-2) 

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class PositionalEncoding(nn.Module):
    # n_position 必須是靜態常數，這裡我們使用 MAX_TARGET_LEN
    def __init__(self, d_hid, n_position=MAX_TARGET_LEN):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


# --- DECODER CLASS (標準 Transformer) ---

class Decoder(nn.Module):
    ''' A decoder model with standard self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec) 
        self.dropout = nn.Dropout(p=dropout)
        
        self.layer_stack = nn.ModuleList([
            DecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout
            )
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.pad_idx = pad_idx

    # 必須接收 Mask 作為輸入，並且參數名稱必須與 Layers.py 中的 DecoderLayer 匹配
    def forward(self, trg_seq, enc_output, src_mask, trg_mask):
        dec_output = self.trg_word_emb(trg_seq)
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            # 傳遞 slf_attn_mask 和 dec_enc_attn_mask
            dec_output, _, _ = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
        
        return dec_output


# --- 頂層 Seq2Seq 模型 (ModernBERT + Standard Decoder) ---

class Seq2SeqModelWithFlashAttn(nn.Module):
    def __init__(
        self,
        transformer_model_path: str = "answerdotai/ModernBERT-base",
        freeze_encoder: bool = True,
        weight_dtype: Optional[torch.dtype] = torch.bfloat16,
    ):
        super().__init__()
        
        encoder_kwargs = {}
        if weight_dtype is not None:
            encoder_kwargs["torch_dtype"] = weight_dtype
        self.encoder = ModernBertModel.from_pretrained(transformer_model_path, **encoder_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_path)
        
        n_trg_vocab = len(self.tokenizer)
        d_word_vec=768
        n_layers=12
        n_head=12
        d_k=768 // 12
        d_v=768 // 12
        d_model=768
        d_inner=768 * 2
        pad_idx=self.tokenizer.pad_token_id
        dropout=0.1

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, d_word_vec=d_word_vec, n_layers=n_layers, n_head=n_head,
            d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, pad_idx=pad_idx, dropout=dropout
        )
        self.output_projection = nn.Linear(d_model, n_trg_vocab, bias=False)
        self._cast_modules_to_dtype(weight_dtype)
        self._tie_decoder_embeddings()
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.pad_idx = pad_idx
        self.weight_dtype = weight_dtype
        
    # Forward 必須接收 Mask 作為參數
    def forward(self, src_input_ids, trg_input_ids, src_mask, trg_mask):
        # 1. Encoder (ModernBERT) - BERT 內核處理序列長度，我們只需傳遞注意力遮罩
        enc_output = self.encoder(
            src_input_ids, 
            attention_mask=(src_input_ids != self.pad_idx).long()
        ).last_hidden_state
        
        # 2. Decoder (標準 Transformer)
        dec_output = self.decoder(
            trg_input_ids, enc_output, src_mask=src_mask, trg_mask=trg_mask
        )

        # 3. 輸出投影
        logits = self.output_projection(dec_output)
        return logits
    
    # 這裡省略 generate 和 filtering 的完整實作，但它們也必須重寫以支持 Padding/Masking
    # ... (省略)
    
    def _cast_modules_to_dtype(self, dtype: Optional[torch.dtype]) -> None:
        if dtype is None:
            return
        self.encoder.to(dtype=dtype)
        self.decoder.to(dtype=dtype)
        self.output_projection.to(dtype=dtype)

    def _tie_decoder_embeddings(self) -> None:
        with torch.no_grad():
            self.decoder.trg_word_emb.weight.copy_(
                self.encoder.embeddings.tok_embeddings.weight
            )
        self.output_projection.weight = self.decoder.trg_word_emb.weight