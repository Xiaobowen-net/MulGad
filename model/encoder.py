import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_activation_fn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="gelu",bias=True):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        # x [B, L, C]
        ## Multi-Head attention
        src = self.attention(x, x, x,attn_mask = attn_mask)
        ## Add & Norm
        src = x + self.dropout_attn(src) # Add: residual connection with residual dropout
        src = self.norm_attn(src)
        # Feed-forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        src = self.norm_ffn(src)

        return src

class Attention(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Attention, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, C]

        for attn_layer in self.attn_layers:
            x = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x