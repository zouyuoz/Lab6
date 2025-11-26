''' Define the Layers '''
import torch.nn as nn
from transformer.SubLayers import PositionwiseFeedForward, MultiHeadCrossAttention_Flash, MultiHeadSelfAttention_Flash
class DecoderLayer_Flash(nn.Module):
    ''' Compose with three layers using Flash Attention '''
    def __init__(self, d_model, d_inner, n_head, d_qkv, dropout=0.1):
        super(DecoderLayer_Flash, self).__init__()
        self.slf_attn = ...
        self.enc_attn = ...
        self.pos_ffn = ...

    def forward(self, dec_input, dec_seq_lens, enc_output, enc_seq_lens):
        #################YOUR CODE HERE#################
        # 1. Self-Attention
        # 2. Encoder-Decoder Attention
        # 3. Position-wise Feed-Forward Network
        ################################################
        return dec_output