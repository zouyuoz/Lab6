''' Define the Layers '''
import torch.nn as nn
from transformer.SubLayers import PositionwiseFeedForward, MultiHeadCrossAttention_Flash, MultiHeadSelfAttention_Flash
class DecoderLayer_Flash(nn.Module):
    ''' Compose with three layers using Flash Attention '''
    def __init__(self, d_model, d_inner, n_head, d_qkv, dropout=0.1):
        super(DecoderLayer_Flash, self).__init__()
        # Self-attention must be causal
        self.slf_attn = MultiHeadSelfAttention_Flash(
            n_head, d_model, d_qkv, dropout=dropout, causal=True
        )
        # Cross-attention is not causal
        self.enc_attn = MultiHeadCrossAttention_Flash(
            n_head, d_model, d_qkv, dropout=dropout, causal=False
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout
        )

    def forward(self, dec_input, dec_seq_lens, enc_output, enc_seq_lens):
        #################YOUR CODE HERE#################
        # 1. Self-Attention
        dec_output = self.slf_attn(dec_input, dec_seq_lens)
        
        # 2. Encoder-Decoder Attention
        dec_output = self.enc_attn(
            dec_output, enc_output, dec_seq_lens, enc_seq_lens
        )
        
        # 3. Position-wise Feed-Forward Network
        dec_output = self.pos_ffn(dec_output)
        ################################################
        return dec_output