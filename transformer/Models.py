''' Define the Transformer model '''
from multiprocessing import context
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from transformer.Layers import DecoderLayer_Flash
from transformer.utils import *
from transformer.Const import *
class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x, seq_lens=None):
        if seq_lens is None:
            return x + self.pos_table[:, :x.size(1)].clone().detach()

        seq_lens = seq_lens.to(device=x.device, dtype=torch.long)
        total_seq_len = int(seq_lens.sum().item())
        if total_seq_len != x.size(0):
            raise ValueError(
                f"Packed sequence length mismatch: got {x.size(0)} tokens, "
                f"but seq_lens sum to {total_seq_len}."
            )

        seq_starts = torch.cumsum(seq_lens, dim=0) - seq_lens
        token_seq_ids = torch.arange(seq_lens.size(0), device=x.device).repeat_interleave(seq_lens)
        seq_offsets = seq_starts[token_seq_ids]
        position_ids = torch.arange(total_seq_len, device=x.device) - seq_offsets

        max_pos = int(position_ids.max().item())
        if max_pos >= self.pos_table.size(1):
            raise ValueError(
                f"Requested position {max_pos} exceeds available sinusoid size {self.pos_table.size(1)}."
            )

        pos_emb = self.pos_table[:, position_ids, :].squeeze(0)
        return x + pos_emb
    
class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, flash_attn=True):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.flash_attn = flash_attn
        
        d_qkv = d_k
        self.layer_stack = nn.ModuleList([
            DecoderLayer_Flash(
                d_model, d_inner, n_head, d_qkv, dropout=dropout
            )
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask):
        ######### YOUR CODE STARTS HERE #########
        # 這裡的 trg_mask 和 src_mask 實際上傳入的是 seq_lens
        trg_seq_lens = trg_mask
        enc_seq_lens = src_mask

        # 1. Embedding
        dec_output = self.trg_word_emb(trg_seq)
        
        # 2. Positional Encoding
        dec_output = self.position_enc(dec_output, seq_lens=trg_seq_lens)
        
        # 3. Dropout
        dec_output = self.dropout(dec_output)

        # 4. Decoder Layers
        for layer in self.layer_stack:
            dec_output = layer(
                dec_output,
                trg_seq_lens,
                enc_output,
                enc_seq_lens,
            )
            
        # 5. Layer Norm
        dec_output = self.layer_norm(dec_output)
        #########################################
        return dec_output


from transformers import ModernBertModel, AutoTokenizer
from transformer.Const import *

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
        
        # 這裡會載入標準 BERT
        self.encoder = ModernBertModel.from_pretrained(transformer_model_path, **encoder_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_path)
        
        self.decoder = Decoder(
            n_trg_vocab=len(self.tokenizer),
            d_word_vec=768,
            n_layers=12,
            n_head=12,
            d_k=768 // 12,
            d_v=768 // 12,
            d_model=768,
            d_inner=768 * 2,
            pad_idx=self.tokenizer.pad_token_id,
            n_position=MAX_TARGET_LEN,
            dropout=0.1,
            flash_attn=True)
            
        self.output_projection = nn.Linear(768, len(self.tokenizer), bias=False)
        self._cast_modules_to_dtype(weight_dtype)
        self._tie_decoder_embeddings()
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.weight_dtype = weight_dtype

    def forward(self, src_input_ids, trg_input_ids, src_seq_len, trg_seq_len):
        # src_input_ids: 1D Packed Tensor (Total_Tokens)
        # src_seq_len: 1D Tensor (Batch_Size)

        # 1. [轉接頭] Packed (1D) -> Padded (2D) for BERT
        # 將 1D 壓扁的數據根據長度切開，並補 0 成 2D 矩陣
        src_inputs_list = torch.split(src_input_ids, src_seq_len.tolist())
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            src_inputs_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        
        # 製作 BERT 需要的 Attention Mask (1 為有效, 0 為 Padding)
        attention_mask = (padded_input_ids != self.tokenizer.pad_token_id).long()

        # 2. Encoder Forward (Standard BERT)
        enc_outputs = self.encoder(
            input_ids=padded_input_ids,
            attention_mask=attention_mask
        )
        last_hidden_state = enc_outputs.last_hidden_state # (B, Max_Len, H)

        # 3. [轉接頭] Padded (2D) -> Packed (1D) for Custom Decoder
        # 利用 mask 將補 0 的部分去掉，壓回 1D 給 Decoder 用
        enc_output_packed = last_hidden_state[attention_mask.bool()]

        # 4. Decoder Forward
        dec_output = self.decoder(
            trg_seq=trg_input_ids,
            trg_mask=trg_seq_len,
            enc_output=enc_output_packed,
            src_mask=src_seq_len
        )
        
        # Project to vocabulary
        logits = self.output_projection(dec_output)
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        src_seq_len: torch.Tensor,
        generation_limit: int,
        sampling: bool = False,
        top_k: int = 10,
        top_p: float = 0.9,
    ) -> List[str]:
        device = self.output_projection.weight.device
        src_seq_len = src_seq_len.to(device=device, dtype=torch.int32)
        bsz = src_seq_len.size(0)

        # 1. [轉接頭] Packed (1D) -> Padded (2D) for BERT
        # 與 forward 相同的邏輯
        src_inputs_list = torch.split(input_ids, src_seq_len.tolist())
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            src_inputs_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = (padded_input_ids != self.tokenizer.pad_token_id).long()

        # 2. Encoder Forward
        enc_outputs = self.encoder(
            input_ids=padded_input_ids,
            attention_mask=attention_mask
        )
        last_hidden_state = enc_outputs.last_hidden_state
        
        # 3. [轉接頭] Padded (2D) -> Packed (1D)
        enc_output = last_hidden_state[attention_mask.bool()]
        
        # 4. Initialization
        bos_id = self.tokenizer.cls_token_id 
        sequences: List[torch.Tensor] = [torch.tensor([bos_id], dtype=torch.long, device=device) for _ in range(bsz)]
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)

        for _ in range(generation_limit):
            trg_input_ids = torch.cat(sequences)
            trg_seq_len = torch.tensor([s.size(0) for s in sequences], dtype=torch.int32, device=device)

            dec_output = self.decoder(
                trg_seq=trg_input_ids,
                trg_mask=trg_seq_len,
                enc_output=enc_output,
                src_mask=src_seq_len
            )
            
            logits = self.output_projection(dec_output) 
            cu_seqlens_no_pad = torch.cumsum(trg_seq_len, dim=0, dtype=torch.long)
            last_token_indices = cu_seqlens_no_pad - 1 
            next_token_logits = logits[last_token_indices]

            if finished.any():
                next_token_logits[finished] = -float("inf")
                next_token_logits[finished, self.tokenizer.pad_token_id] = 0.0

            if sampling:
                filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                probabilities = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)

            for idx in range(bsz):
                if finished[idx]:
                    continue
                sequences[idx] = torch.cat([sequences[idx], next_token[idx].view(1)])
                if next_token[idx].item() == self.tokenizer.sep_token_id:
                    finished[idx] = True

            if bool(torch.all(finished)):
                break

        output_text: List[str] = []
        for seq in sequences:
            tokens = seq.tolist()
            if self.tokenizer.sep_token_id in tokens:
                tokens = tokens[: tokens.index(self.tokenizer.sep_token_id)]
            output_text.append(self.tokenizer.decode(tokens, skip_special_tokens=True))
        return output_text

    def _cast_modules_to_dtype(self, dtype: Optional[torch.dtype]) -> None:
        if dtype is None:
            return
        self.encoder.to(dtype=dtype)
        self.decoder.to(dtype=dtype)
        self.output_projection.to(dtype=dtype)

    def _tie_decoder_embeddings(self) -> None:
        with torch.no_grad():
            # 兼容 BERT (word_embeddings)
            if hasattr(self.encoder.embeddings, "word_embeddings"):
                encoder_embeddings = self.encoder.embeddings.word_embeddings
            elif hasattr(self.encoder.embeddings, "tok_embeddings"):
                encoder_embeddings = self.encoder.embeddings.tok_embeddings
            else:
                raise AttributeError("Cannot find embedding layer in encoder.embeddings")

            self.decoder.trg_word_emb.weight.copy_(encoder_embeddings.weight)
        self.output_projection.weight = self.decoder.trg_word_emb.weight

    def top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k,
        top_p
    ) -> torch.Tensor:
        # logits: (bsz, vocab_size)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        filter_value: float = -float('Inf')
        vocab_size = logits.size(-1)
        if top_k > 0 and top_k < vocab_size:
            v, _ = torch.topk(logits, top_k)
            kth_value = v[:, -1].unsqueeze(-1)
            logits[logits < kth_value] = filter_value
            
        if 0.0 < top_p < 1.0:
            # 1. Sort logits
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            
            # 2. Calculate softmax probabilities
            probabilities = torch.softmax(sorted_logits, dim=-1)
            
            # 3. Calculate cumulative probabilities
            cumulative_probabilities = torch.cumsum(probabilities, dim=-1)

            # 4. Find tokens to remove
            sorted_indices_to_remove = cumulative_probabilities > top_p
            # Shift mask right to keep the first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            # Scatter filtered values back
            # 修正：使用 scatter 建立 mask，然後直接將對應位置設為 filter_value
            # 這樣可以避免 dtype 不匹配的問題，並且正確保留未被過濾的 logits 值
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value

        return logits