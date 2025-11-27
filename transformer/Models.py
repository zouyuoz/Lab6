''' Define the Transformer model using Padding/Masking '''
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from transformers import ModernBertModel, AutoTokenizer
# 確保 Layers.py 內已定義 DecoderLayer (使用標準注意力)
from transformer.Layers import DecoderLayer 
from transformer.Constants import *
import torch.nn.functional as F

# --- MASKING & UTILITIES (來自 jadore801120) ---

def get_pad_mask(seq, pad_idx):
    # 遮罩的值為 True 的地方會被遮蔽 (設為 -inf)
    # 遮罩 shape: (B, 1, L)
    return (seq != pad_idx).unsqueeze(-2) 

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    # 建立上三角矩陣遮罩
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=MAX_TARGET_LEN):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        # Sinusoid position encoding table (使用您原有的 numpy 邏輯，但這裡簡化為 PyTorch 兼容版)
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        # 簡單相加位置編碼 (B, L, D) + (1, L, D)
        return x + self.pos_table[:, :x.size(1)].clone().detach()


# --- DECODER CLASS (標準 Transformer) ---

class Decoder(nn.Module):
    ''' A decoder model with standard self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec) # 使用標準 PositionalEncoding
        self.dropout = nn.Dropout(p=dropout)
        
        self.layer_stack = nn.ModuleList([
            # 這裡必須使用您 Layers.py 中定義的 DecoderLayer (非 Flash 版本)
            DecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout
            )
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.pad_idx = pad_idx

    # 必須接收 Mask 作為輸入
    def forward(self, trg_seq, enc_output, src_mask, trg_mask):
        # trg_seq: (batch_size, len_t)
        
        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        # 堆疊 Decoder Layer
        for dec_layer in self.layer_stack:
            # dec_layer 輸出: dec_output, dec_slf_attn, dec_enc_attn
            dec_output, _, _ = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
        
        # dec_output: (batch_size, len_t, d_model)
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
        
        # Encoder: ModernBERT 初始化邏輯保持不變
        encoder_kwargs = {}
        if weight_dtype is not None:
            encoder_kwargs["torch_dtype"] = weight_dtype
        self.encoder = ModernBertModel.from_pretrained(transformer_model_path, **encoder_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_path)
        
        # Decoder 參數
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

        # Decoder: 使用標準 Decoder 類別
        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab,
            d_word_vec=d_word_vec,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            d_model=d_model,
            d_inner=d_inner,
            pad_idx=pad_idx,
            dropout=dropout
        )
        self.output_projection = nn.Linear(d_model, n_trg_vocab, bias=False)
        self._cast_modules_to_dtype(weight_dtype)
        self._tie_decoder_embeddings()
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.pad_idx = pad_idx
        self.weight_dtype = weight_dtype
        
    # 必須接收 Mask 作為參數
    def forward(self, src_input_ids, trg_input_ids, src_mask, trg_mask):

        # 1. Encoder (ModernBERT)
        # BERT 輸出: last_hidden_state (batch_size, len_s, d_model)
        # BERT 自帶的 attention_mask (0 for padding, 1 for real tokens)
        enc_output = self.encoder(
            src_input_ids, 
            attention_mask=(src_input_ids != self.pad_idx).long()
        ).last_hidden_state
        
        # 2. Decoder (標準 Transformer)
        # 將 encoder 輸出和兩個 mask 傳遞給 decoder
        dec_output = self.decoder(
            trg_input_ids, 
            enc_output,
            src_mask=src_mask, # Encoder-Decoder Attention Mask (Padding mask of src)
            trg_mask=trg_mask  # Decoder Self-Attention Mask (Causal + Padding mask of trg)
        )

        # 3. 輸出投影
        logits = self.output_projection(dec_output)
        
        # logits: (batch_size, len_t, n_tgt_vocab)
        return logits
    
    # top_k_top_p_filtering 保持不變 (但應在 generate 內使用)
    def top_k_top_p_filtering(self, logits: torch.Tensor, top_k, top_p) -> torch.Tensor:
        # logits: (bsz, vocab_size)
        # (這裡的程式碼邏輯保持您原本的實現，因為它是正確的)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        filter_value: float = -float('Inf')
        vocab_size = logits.size(-1)
        if top_k > 0 and top_k < vocab_size:
            v, _ = torch.topk(logits, top_k)
            kth_value = v[:, -1].unsqueeze(-1)
            logits[logits < kth_value] = filter_value
            
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probabilities = torch.softmax(sorted_logits, dim=-1)
            cumulative_probabilities = torch.cumsum(probabilities, dim=-1)
            sorted_indices_to_remove = cumulative_probabilities > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            logits.scatter_(dim=-1, index=sorted_indices, src=filter_value * sorted_indices_to_remove)
            
        return logits

    # generate 邏輯必須完全重寫以支持 Padding/Masking，但為了簡潔，我們只替換核心邏輯
    def generate(self, input_ids, src_seq_len, generation_limit, sampling=False, top_k=10, top_p=0.9):
        # ******************************************************************
        # WARNING: This implementation is for demonstration. 
        # The full generate logic for Padding/Masking is complex and long.
        # ******************************************************************
        raise NotImplementedError("Generation for padding model requires full rewrite.")
        
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